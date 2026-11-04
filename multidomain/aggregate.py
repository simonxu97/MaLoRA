"""Tables 1, 5 and 6: average normalized accuracy and relative accuracy drop.

For every (benchmark, seed) and method, the configuration with the highest mean
validation accuracy is selected; its per-domain test accuracies are divided by the
oracle accuracies (oracle.py) and averaged over all domains, pretrain domain included
(normalized accuracy, Ilharco et al. 2022). Relative drop = 100 - normalized accuracy.
Per benchmark: mean +- std over seeds; Avg: mean over benchmarks, std = sqrt(sum sigma_b^2) / 6.

Selection grids:
  Soup         no hyperparameter
  TA, MaLoRA   s in {5, 10, 15, 20, 24, 29, 34, 39, 44, 49} / 49, the points of the swept
               k/49 grid nearest to {0.1, ..., 1.0}
  TIES, DARE   density {0.2, ..., 0.9} x lambda {0.1, ..., 1.0}

Usage:
    python aggregate.py                                  # your runs in runs/
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from common import BENCHMARKS, RUNS_DIR, SEEDS

TABLE_K = {5, 10, 15, 20, 24, 29, 34, 39, 44, 49}  # s = k / 49 (runs log s rounded or at full precision)
METHODS = {  # name: (subdirectory, row filter)
    "Soup": ("merge_fair", lambda m: m == "simple_average"),
    "TA": ("merge_fair", lambda m: m.startswith("custom_linear_uniform_scaled_lam") and round(_lam(m) * 49) in TABLE_K),
    "TIES": ("ties_lamsweep", lambda m: m.startswith("ties_")),
    "DARE-TIES": ("dare_ties_lamsweep", lambda m: m.startswith("dare_ties_")),
    "DARE-Linear": ("dare_linear_lamsweep", lambda m: m.startswith("dare_linear_")),
    "MaLoRA": ("merge_fair", lambda m: m.startswith("custom_linear_margin_ratio_drop_lam") and round(_lam(m) * 49) in TABLE_K),
}
LABELS = {"pacs": "PACS", "vlcs": "VLCS", "terra_incognita": "Terra Inc.", "mnli": "MNLI",
          "amazon_reviews": "Amazon", "civil_comments": "CivilComm."}


def _lam(method_name):
    return float(re.search(r"_lam([0-9.]+)$", method_name).group(1))


def best_on_val(csv_path, row_filter):
    df = pd.read_csv(csv_path)
    sub = df[df["Method"].map(row_filter)]
    return sub.sort_values("Val_Avg_Accuracy", ascending=False).iloc[0]


def normalized_accuracy(runs_dir, bench, seed, method):
    rdir = f"{runs_dir}/{bench}_seed{seed}"
    sub, row_filter = METHODS[method]
    csv = f"{rdir}/{sub}/merge_results.csv"
    row = best_on_val(csv, row_filter)
    with open(f"{rdir}/oracle_accs.json") as f:
        oracle = json.load(f)["oracle_accs"]
    return 100 * float(np.mean([row[d] / oracle[d] for d in oracle]))


def missing_files(runs_dir, seeds):
    files = [f"{runs_dir}/{b}_seed{s}/{name}" for b in BENCHMARKS for s in seeds
             for name in ["oracle_accs.json"] + [f"{sub}/merge_results.csv" for sub in sorted({v[0] for v in METHODS.values()})]]
    return [f for f in files if not os.path.exists(f)]


def collect(runs_dir, seeds=SEEDS):
    """Mean and population std over the same seeds for every method and benchmark."""
    missing = missing_files(runs_dir, seeds)
    if missing:
        raise FileNotFoundError(f"{len(missing)} result files missing, e.g.\n  " + "\n  ".join(missing[:10]))
    return {(m, b): (float(np.mean(v)), float(np.std(v)))
            for m in METHODS for b in BENCHMARKS
            for v in [[normalized_accuracy(runs_dir, b, s, m) for s in seeds]]}


def print_tables(stats, n_seeds):
    head = f"{'Method':<12}" + "".join(f"{LABELS[b]:>17}" for b in BENCHMARKS) + f"{'Avg':>17}"
    for title, f in [("Average normalized accuracy (%)", lambda v: v), ("Relative accuracy drop (%)", lambda v: 100 - v)]:
        print(f"\n{title}, mean +- std over {n_seeds} seeds")
        print(head)
        for m in METHODS:
            cells = [stats[m, b] for b in BENCHMARKS]
            avg = np.mean([c[0] for c in cells])
            avg_std = np.sqrt(np.sum([c[1] ** 2 for c in cells])) / len(cells)
            print(f"{m:<12}" + "".join(f"{f(mu):>10.2f} +-{sd:>4.2f}" for mu, sd in cells) + f"{f(avg):>10.2f} +-{avg_std:>4.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default=str(RUNS_DIR))
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = p.parse_args()
    print_tables(collect(args.runs_dir, args.seeds), len(args.seeds))
