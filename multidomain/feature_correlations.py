"""Feature correlations within and across classes and domains (App. H, multi-domain correlation table).

Frozen pre-trained backbones, inference only: google/vit-base-patch16-224-in21k (vision)
and xlm-roberta-base (language), CLS token. At most 500 samples per (class, domain) cell.
Features are centred by the global mean, then for every pair of (class, domain) cells
the mean cosine similarity is computed exactly (bilinear identity) and the cell pairs
are grouped into same/different class x same/different domain. Reported: mean +- std
over cell pairs in each group.

Usage:
    python feature_correlations.py --benchmark pacs
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from benchmarks import get_dataset
from common import BENCHMARKS, ROOT

GROUPS = ["within_class_within_task", "within_class_cross_task", "cross_class_within_task", "cross_class_cross_task"]


def subsample_indices(labels, domains, max_per_cell, seed=0):
    rng = np.random.RandomState(seed)
    keep = []
    for c in np.unique(labels):
        for t in np.unique(domains):
            where = np.where((labels == c) & (domains == t))[0]
            if len(where) > max_per_cell:
                where = rng.choice(where, max_per_cell, replace=False)
            if len(where):
                keep.append(where)
    return np.sort(np.concatenate(keep))


@torch.no_grad()
def extract_vision(ds, idx, device, batch_size, num_workers):
    import torchvision.transforms as T
    from PIL import Image, ImageFile
    from transformers import AutoImageProcessor, AutoModel
    name = "google/vit-base-patch16-224-in21k"
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()
    tf = T.Compose([T.Resize((proc.size["height"], proc.size["width"])), T.ToTensor(),
                    T.Normalize(mean=proc.image_mean, std=proc.image_std)])
    field = "image" if "image" in ds.column_names else "image_path"
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    def as_pil(v):
        return (Image.open(v) if isinstance(v, str) else v).convert("RGB")

    def collate(batch):
        return torch.stack([tf(as_pil(b[field])) for b in batch])

    items = [ds[int(i)] for i in idx] if len(idx) < 20000 else Subset(ds, idx)
    feats = [model(pixel_values=x.to(device)).last_hidden_state[:, 0].float().cpu()
             for x in DataLoader(items, batch_size=batch_size, num_workers=num_workers, collate_fn=collate, pin_memory=True)]
    return torch.cat(feats)


@torch.no_grad()
def extract_text(handler, ds, idx, device, batch_size, num_workers):
    from transformers import AutoModel, AutoTokenizer
    name = "xlm-roberta-base"
    model = AutoModel.from_pretrained(name).to(device).eval()
    collate = handler.get_collate_fn(AutoTokenizer.from_pretrained(name))
    sub = [ds[int(i)] for i in idx]
    feats = [model(input_ids=b.input_ids.to(device), attention_mask=b.attention_mask.to(device)).last_hidden_state[:, 0].float().cpu()
             for b in DataLoader(sub, batch_size=batch_size, num_workers=num_workers, collate_fn=collate)]
    return torch.cat(feats)


def class_domain_groups(feats, labels, domains):
    """Mean cosine similarity for every pair of (class, domain) cells, grouped four ways."""
    f = feats - feats.mean(0, keepdim=True)          # global centring
    f = f / f.norm(dim=1).mean().clamp_min(1e-12)
    u = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
    S, n, cid, tid = [], [], [], []
    for c in np.unique(labels):
        for t in np.unique(domains):
            m = (labels == c) & (domains == t)
            if m.sum() < 2:
                continue
            S.append(u[torch.from_numpy(m)].sum(0).double())
            n.append(float(m.sum()))
            cid.append(c)
            tid.append(t)
    S, n = torch.stack(S), torch.tensor(n, dtype=torch.float64)
    G = S @ S.T
    groups = {g: [] for g in GROUPS}
    for i in range(len(n)):
        groups["within_class_within_task"].append(float((G[i, i] - n[i]) / (n[i] * (n[i] - 1))))
        for j in range(i + 1, len(n)):
            same_c, same_t = cid[i] == cid[j], tid[i] == tid[j]
            key = ("within_class_" if same_c else "cross_class_") + ("within_task" if same_t else "cross_task")
            groups[key].append(float(G[i, j] / (n[i] * n[j])))
    return {g: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n_cell_pairs": len(v)} for g, v in groups.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    p.add_argument("--max_per_cell", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    handler = get_dataset(args.benchmark)
    ds = handler.load_data()
    labels = np.array(ds["label"])
    domains = np.array([str(x) for x in ds["domain"]])
    idx = subsample_indices(labels, domains, args.max_per_cell)
    if handler.is_text:
        feats = extract_text(handler, ds, idx, device, args.batch_size, args.num_workers)
    else:
        feats = extract_vision(ds, idx, device, args.batch_size, args.num_workers)
    groups = class_domain_groups(feats, labels[idx], domains[idx])
    for g, v in groups.items():
        print(f"{args.benchmark:16s} {g:26s} {v['mean']:+.3f} +- {v['std']:.3f}  ({v['n_cell_pairs']} cell pairs)")
    out = Path(args.out) if args.out else ROOT / "outputs" / f"feature_correlations_{args.benchmark}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"benchmark": args.benchmark, "n_samples": int(len(idx)), "groups": groups}, f, indent=2)


if __name__ == "__main__":
    main()
