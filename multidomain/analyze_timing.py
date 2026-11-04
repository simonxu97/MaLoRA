"""Merge-selection runtime table (App. H, "Computational overhead").

Reads <runs_dir>/<benchmark>_seed<s>/merge_timing.json (timing.py). For each run the
median over repeats is taken; per benchmark, the mean over seeds. End-to-end cost of
selecting a merge under the matched budget of n configurations per method:
    MaLoRA      t_gamma + n * (t_apply + t_val_eval)    (margins computed once)
    all others  n * (t_total + t_val_eval)
with n = 49 (the size of the TA / MaLoRA scale grid).

Usage:
    python analyze_timing.py
    python analyze_timing.py --benchmarks pacs vlcs --seeds 1 2 3   # a subset of the runs
"""
import argparse
import json
import os
import sys

import numpy as np

from common import BENCHMARKS, RUNS_DIR, SEEDS

METHODS = ["Soup", "TA", "DARE-Linear", "DARE-TIES", "TIES", "MaLoRA"]
LABELS = {"pacs": "PACS", "vlcs": "VLCS", "terra_incognita": "Terra Inc.", "mnli": "MNLI",
          "amazon_reviews": "Amazon", "civil_comments": "CivilComm."}
MERGE_OUTPUTS = ["merge_fair", "ties_lamsweep", "dare_ties_lamsweep", "dare_linear_lamsweep"]  # inputs of timing.py


def missing_inputs_message(runs_dir, benchmarks, seeds):
    """None if every run has merge_timing.json, else what to run (from multidomain/) to produce it."""
    timing_only, full_run = [], []
    for b in benchmarks:
        for s in seeds:
            rdir = f"{runs_dir}/{b}_seed{s}"
            if os.path.exists(f"{rdir}/merge_timing.json"):
                continue
            if all(os.path.exists(f"{rdir}/{sub}/merge_results.csv") for sub in MERGE_OUTPUTS):
                timing_only.append(f"python timing.py --benchmark {b} --seed {s}")
            else:
                full_run.append(f"bash ../scripts/run_multidomain.sh {b} {s}")
    if not timing_only and not full_run:
        return None
    lines = [f"Missing timing results for {len(timing_only) + len(full_run)} of "
             f"{len(benchmarks) * len(seeds)} runs in {runs_dir}."]
    if timing_only:
        lines += ["Merged but not timed; from multidomain/ run:"] + [f"  {c}" for c in timing_only]
    if full_run:
        lines += ["Not trained or merged yet; from multidomain/ run:"] + [f"  {c}" for c in full_run]
    lines.append("Or restrict the table to finished runs with --benchmarks / --seeds.")
    return "\n".join(lines)


def mean_timings(runs_dir, bench, method, seeds):
    """Per-seed medians over repeats, averaged over seeds."""
    per_seed = []
    for s in seeds:
        with open(f"{runs_dir}/{bench}_seed{s}/merge_timing.json") as f:
            d = json.load(f)
        t = {k: float(np.median(v)) for k, v in d["methods"][method].items() if k.startswith("t_")}
        t["t_val_eval"] = float(np.median(d["t_val_eval"]))
        per_seed.append(t)
    return {k: float(np.mean([t[k] for t in per_seed])) for k in per_seed[0]}


def end_to_end(t, method, n_configs):
    if method == "MaLoRA":
        return t["t_gamma"] + n_configs * (t["t_apply"] + t["t_val_eval"])
    return n_configs * (t["t_total"] + t["t_val_eval"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default=str(RUNS_DIR))
    p.add_argument("--n_configs", type=int, default=49)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=BENCHMARKS)
    args = p.parse_args()
    benchmarks = [b for b in BENCHMARKS if b in args.benchmarks]  # keep the table's column order

    message = missing_inputs_message(args.runs_dir, benchmarks, args.seeds)
    if message:
        sys.exit(message)
    timings = {(m, b): mean_timings(args.runs_dir, b, m, args.seeds) for m in METHODS for b in benchmarks}
    rows = {m: [end_to_end(timings[m, b], m, args.n_configs) for b in benchmarks] for m in METHODS}
    malora_avg, ta_avg = np.mean(rows["MaLoRA"]), np.mean(rows["TA"])
    print(f"End-to-end merge-selection runtime (s), {args.n_configs} configurations per method, "
          f"mean over seeds {args.seeds}")
    print(f"{'Method':<12}" + "".join(f"{LABELS[b]:>11}" for b in benchmarks) + f"{'Avg':>9}{'Ratio':>8}")
    for m in METHODS:
        avg = np.mean(rows[m])
        print(f"{m:<12}" + "".join(f"{v:>11.1f}" for v in rows[m]) + f"{avg:>9.1f}{avg / malora_avg:>7.2f}x")
    print(f"MaLoRA overhead over TA: {malora_avg - ta_avg:.1f} s ({100 * (malora_avg / ta_avg - 1):.2f}%)")
    print("margin pass / one validation pass: " + ", ".join(
        f"{LABELS[b]} {timings['MaLoRA', b]['t_gamma'] / timings['MaLoRA', b]['t_val_eval']:.2f}" for b in benchmarks))


if __name__ == "__main__":
    main()
