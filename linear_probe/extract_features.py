"""Stage 0: frozen-backbone features for every CIFAR-100 image (App. H, "Frozen feature extractors").

Writes features/{train,test}_{arch}.pt = {"features": (N, d) float32, "labels": (N,) fine labels}.

Post-processing: subtract the mean of each group of five consecutive fine-label ids
(fine_label // 5), then divide by the mean row norm.

Usage:
    python extract_features.py --arch vit
"""
import argparse
from pathlib import Path

import torch
import torchvision.models as tvm
import torchvision.transforms as T
from tqdm import tqdm

from common import ARCHS, DATA_DIR, DEVICE, FEAT_DIR, cifar100

TIMM_TAGS = {
    "vit": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "convnext": "convnext_tiny.in12k_ft_in1k",
}


def transform():
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def get_backbone(arch):
    """Returns a module with .forward_features(x) and its output dimension."""
    if arch in TIMM_TAGS:
        import timm
        m = timm.create_model(TIMM_TAGS[arch], pretrained=True)
        m.eval()
        return m, m.num_features
    if arch == "resnet":
        r50 = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        backbone = torch.nn.Sequential(*(list(r50.children())[:-1]))
        backbone.eval()
        backbone.forward_features = lambda x: backbone(x).squeeze(-1).squeeze(-1)
        return backbone, 2048
    if arch == "clip":
        import open_clip
        clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        clip.eval()

        class ClipWrapper(torch.nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.clip = clip_model

            def forward_features(self, x):
                return self.clip.encode_image(x)

        return ClipWrapper(clip), clip.visual.output_dim
    raise ValueError(f"unknown arch {arch!r}; choose from {ARCHS}")


@torch.no_grad()
def extract(arch, data_dir=DATA_DIR, out_dir=FEAT_DIR, batch_size=512, save_raw=False):
    backbone, D = get_backbone(arch)
    backbone.to(DEVICE)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, train_flag in [("train", True), ("test", False)]:
        ds = cifar100(train_flag, transform=transform(), data_dir=data_dir)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        N, ptr = len(ds), 0
        feats = torch.empty(N, D)
        fine = torch.empty(N, dtype=torch.long)
        for imgs, lbls in tqdm(loader, desc=f"{arch} {split_name}"):
            z = backbone.forward_features(imgs.to(DEVICE))
            if arch == "vit":
                z = z[:, 0]  # CLS token
            if z.dim() == 4:
                z = z.mean(dim=(-2, -1))  # global average pool
            b = len(imgs)
            feats[ptr:ptr + b] = z.cpu()
            fine[ptr:ptr + b] = lbls
            ptr += b
        if save_raw:
            torch.save({"features": feats.clone(), "labels": fine}, out_dir / f"{split_name}_{arch}_raw.pt")

        group = fine // 5
        means = torch.zeros(20, D)
        for c in range(20):
            means[c] = feats[group == c].mean(0)
        feats -= means[group]
        feats /= feats.norm(dim=1).mean()

        out_file = out_dir / f"{split_name}_{arch}.pt"
        torch.save({"features": feats, "labels": fine}, out_file)
        print(f"[OK] {arch} {split_name} features -> {out_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=ARCHS, nargs="+", default=ARCHS)
    p.add_argument("--data_dir", default=str(DATA_DIR))
    p.add_argument("--out_dir", default=str(FEAT_DIR))
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--save_raw", action="store_true", help="also save features before centring/scaling")
    args = p.parse_args()
    for a in args.arch:
        extract(a, args.data_dir, args.out_dir, args.batch_size, args.save_raw)
