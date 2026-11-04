"""Measure merge cost for one (benchmark, seed); input for analyze_timing.py.

Times, with CUDA synchronisation and a fixed budget of 4 CPU threads (3 repeats each):
  t_val_eval  one pass over every domain's validation split
  t_coef      method-specific work (MaLoRA: the margin pass, reported also as t_gamma;
              TIES / DARE: trim / sign election / dropout)
  t_apply     building base + merged delta on the device
Each method is timed at the hyperparameters it selects on validation (full grids).
Requires train.py, merge.py and merge_baselines.py outputs. Writes <run_dir>/merge_timing.json.

Usage:
    python timing.py --benchmark pacs --seed 1
"""
import argparse
import gc
import json
import re
import time

import pandas as pd
import torch

from benchmarks import get_dataset
from common import BENCHMARKS, evaluate_domains, load_config, load_model, make_dataloader, run_dir, set_seed
from merge import apply_weighted_deltas, compute_margins, load_run, malora_coefficients, materialize_adapter_delta
from merge_baselines import DENSITIES, LAMBDAS, apply_delta, merged_delta


class Timer:
    def __enter__(self):
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        self.dt = time.perf_counter() - self.t0


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def best_method(csv, prefix):
    df = pd.read_csv(csv)
    sub = df[df["Method"] == prefix] if prefix == "simple_average" else df[df["Method"].str.startswith(prefix)]
    return sub.sort_values("Val_Avg_Accuracy", ascending=False).iloc[0]["Method"]


def parse(name, key, default=1.0):
    m = re.search(rf"_{key}([0-9.]+)", name)
    return float(m.group(1)) if m else default


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run_dir", default=None)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.num_threads)

    cfg = load_config(args.benchmark)
    rdir = args.run_dir or str(run_dir(args.benchmark, args.seed))
    set_seed(args.seed)
    device = torch.device("cuda")
    adapter_dirs, splits, pretrain_domain, base_path = load_run(rdir)
    domains = sorted(adapter_dirs)
    to_merge = sorted(d for d in adapter_dirs if d != pretrain_domain)

    handler = get_dataset(cfg["dataset"])
    ds = handler.load_data()
    collate_fn = handler.get_collate_fn(handler.get_processor(f"{rdir}/base_with_head"))
    is_text = handler.is_text

    def loader(idx):
        return make_dataloader(ds, idx, collate_fn, args.batch_size, 2)

    load_model(base_path, is_text).to(device)  # warm-up
    cleanup()
    t_deltas = []
    for _ in range(args.repeats):
        with Timer() as t:
            deltas = {d: materialize_adapter_delta(base_path, adapter_dirs[d], is_text) for d in to_merge}
        t_deltas.append(t.dt)

    ta_lam = parse(best_method(f"{rdir}/merge_fair/merge_results.csv", "custom_linear_uniform_scaled_lam"), "lam")
    mr_lam = parse(best_method(f"{rdir}/merge_fair/merge_results.csv", "custom_linear_margin_ratio_drop_lam"), "lam")
    baseline_hp = {}
    for label, method in [("TIES", "ties"), ("DARE-TIES", "dare_ties"), ("DARE-Linear", "dare_linear")]:
        name = best_method(f"{rdir}/{method}_lamsweep/merge_results.csv", f"{method}_density")
        baseline_hp[label] = (method, parse(name, "density"), parse(name, "lam"))

    val_loaders = {d: loader(splits[d][1]) for d in domains}
    probe = apply_weighted_deltas(base_path, deltas, {d: 1.0 / len(to_merge) for d in to_merge}, is_text, device)
    t_val = []
    for _ in range(args.repeats):
        with Timer() as t:
            evaluate_domains(probe, val_loaders, domains, device)
        t_val.append(t.dt)
    del probe
    cleanup()

    results = {}

    def timed(label, make_recipe, build):
        coef, apply_, gamma = [], [], []
        for _ in range(args.repeats):
            with Timer() as tc:
                recipe, tg = make_recipe()
            with Timer() as ta:
                model = build(recipe)
            coef.append(tc.dt)
            apply_.append(ta.dt)
            gamma.append(tg)
            del model
            cleanup()
        results[label] = {"t_coef": coef, "t_apply": apply_, "t_total": [c + a for c, a in zip(coef, apply_)]}
        if label == "MaLoRA":
            results[label]["t_gamma"] = gamma
        print(f"{label:<12} merge {sum(results[label]['t_total']) / args.repeats:.3f}s", flush=True)

    def dense(recipe):
        return apply_weighted_deltas(base_path, deltas, recipe, is_text, device)

    def malora_recipe():
        with Timer() as tg:
            margins = compute_margins(base_path, adapter_dirs, splits, pretrain_domain, is_text, loader, device)
        beta = malora_coefficients(margins, pretrain_domain, to_merge)
        return {d: mr_lam * beta[d] for d in to_merge}, tg.dt

    timed("Soup", lambda: ({d: 1.0 / len(to_merge) for d in to_merge}, 0.0), dense)
    timed("TA", lambda: ({d: ta_lam for d in to_merge}, 0.0), dense)
    timed("MaLoRA", malora_recipe, dense)
    for label, (method, density, lam) in baseline_hp.items():
        idx = [(d, l) for d in DENSITIES for l in LAMBDAS].index((round(density, 1), round(lam, 1))) + 1
        timed(label, lambda: (merged_delta(method, deltas, to_merge, density, lam,
                                           torch.Generator().manual_seed(args.seed * 1000 + idx)), 0.0),
              lambda r: apply_delta(base_path, r, is_text, device))

    out = {"benchmark": args.benchmark, "seed": args.seed, "gpu": torch.cuda.get_device_name(0),
           "repeats": args.repeats, "num_threads": args.num_threads, "t_deltas": t_deltas, "t_val_eval": t_val,
           "methods": results}
    with open(f"{rdir}/merge_timing.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
