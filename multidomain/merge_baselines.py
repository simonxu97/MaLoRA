"""TIES, DARE-TIES and DARE-Linear baselines on the same dense task vectors as merge.py.

Each method is swept over density d in {0.2, ..., 0.9} x global scale lambda in {0.1, ..., 1.0}
(80 configurations, density-major) and written to
<run_dir>/<method>_lamsweep/merge_results.csv; aggregate.py picks the best one on validation.

DARE masks are drawn from a generator seeded with seed * 1000 + i, where i is the
configuration's 1-based position in its method's sweep. Parameters are visited in sorted
order, so reruns are deterministic.

Usage:
    python merge_baselines.py --benchmark pacs --seed 1
"""
import argparse
import gc
import os

import pandas as pd
import torch

from common import BENCHMARKS, evaluate_domains, load_config, load_model, make_dataloader, run_dir, set_seed
from benchmarks import get_dataset
from merge import load_run, materialize_adapter_delta, result_row

DENSITIES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
LAMBDAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
METHODS = ["ties", "dare_ties", "dare_linear"]


def trim_top_density(t, density):
    """Keep the top density fraction of |t| (per tensor), zero the rest."""
    if density >= 1.0 or t.numel() == 0:
        return t
    flat = t.abs().flatten()
    k = max(1, int(round(density * flat.numel())))
    threshold = torch.kthvalue(flat, flat.numel() - k + 1).values
    return t * (t.abs() >= threshold).to(t.dtype)


def random_drop(t, drop_rate, generator):
    """DARE: Bernoulli keep-mask with p = 1 - drop_rate, then rescale by 1 / p."""
    keep = 1.0 - drop_rate
    if keep <= 0:
        return torch.zeros_like(t)
    if keep >= 1:
        return t
    mask = torch.bernoulli(torch.full(t.shape, keep, dtype=torch.float32), generator=generator)
    return (t.to(torch.float32) * mask / keep).to(t.dtype)


def ties(stacked, density, lam):
    """Trim each task vector, elect the dominant sign per entry, average the agreeing entries."""
    trimmed = torch.stack([trim_top_density(stacked[i], density) for i in range(stacked.shape[0])], dim=0)
    elected = torch.sign(trimmed.sum(dim=0))
    agree = (torch.sign(trimmed) == elected.unsqueeze(0)).to(trimmed.dtype)
    agree = agree * (elected != 0).unsqueeze(0).to(trimmed.dtype)
    return lam * (trimmed * agree).sum(dim=0) / agree.sum(dim=0).clamp(min=1)


def merged_delta(method, deltas, domains_to_merge, density, lam, generator):
    names = sorted(set().union(*[deltas[d].keys() for d in domains_to_merge]))
    out = {}
    for name in names:
        tensors = [deltas[d].get(name) for d in domains_to_merge]
        ref = next(t for t in tensors if t is not None)
        stacked = torch.stack([t if t is not None else torch.zeros(ref.shape, dtype=ref.dtype) for t in tensors], dim=0)
        if method == "ties":
            out[name] = ties(stacked, density, lam)
        else:
            dropped = torch.stack([random_drop(stacked[i], 1.0 - density, generator) for i in range(stacked.shape[0])], dim=0)
            out[name] = ties(dropped, 1.0, lam) if method == "dare_ties" else lam * dropped.mean(dim=0)
    return out


def apply_delta(base_path, delta, is_text, device):
    base = load_model(base_path, is_text)
    state = base.state_dict()
    for name, d in delta.items():
        if name in state and state[name].shape == d.shape:
            state[name] = (state[name].to(torch.float32) + d.to(torch.float32)).to(state[name].dtype)
    base.load_state_dict(state, strict=False)
    return base.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run_dir", default=None)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    args = p.parse_args()

    cfg = load_config(args.benchmark)
    rdir = args.run_dir or str(run_dir(args.benchmark, args.seed))
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adapter_dirs, domain_splits, pretrain_domain, base_path = load_run(rdir)
    domains = sorted(adapter_dirs)
    domains_to_merge = sorted(d for d in adapter_dirs if d != pretrain_domain)

    handler = get_dataset(cfg["dataset"])
    ds = handler.load_data()
    collate_fn = handler.get_collate_fn(handler.get_processor(f"{rdir}/base_with_head"))
    is_text = handler.is_text

    def loader(idx):
        return make_dataloader(ds, idx, collate_fn, cfg["baseline_batch_size"], cfg["baseline_num_workers"])

    deltas = {d: materialize_adapter_delta(base_path, adapter_dirs[d], is_text) for d in domains_to_merge}
    test_loaders = {d: loader(domain_splits[d][2]) for d in domains}
    val_loaders = {d: loader(domain_splits[d][1]) for d in domains}

    for method in args.methods:
        out_dir = f"{rdir}/{method}_lamsweep"
        os.makedirs(out_dir, exist_ok=True)
        rows = []
        configs = [(d, lam) for d in DENSITIES for lam in LAMBDAS]
        for i, (density, lam) in enumerate(configs, 1):
            name = f"{method}_density{density}" + (f"_lam{lam}" if lam != 1.0 else "")
            gen = torch.Generator(device="cpu").manual_seed(args.seed * 1000 + i)
            merged = apply_delta(base_path, merged_delta(method, deltas, domains_to_merge, density, lam, gen), is_text, device)
            test_m = evaluate_domains(merged, test_loaders, domains, device)
            val_m = evaluate_domains(merged, val_loaders, domains, device)
            print(f"[{method} {i}/{len(configs)}] {name}: val={val_m['avg']:.4f} test={test_m['avg']:.4f}", flush=True)
            row = result_row(name, domains, test_m, val_m)
            row["Density"] = density
            rows.append(row)
            del merged
            gc.collect()
            torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(f"{out_dir}/merge_results.csv", index=False)
        print(f"Saved {out_dir}/merge_results.csv")


if __name__ == "__main__":
    main()
