"""Regularization-sweep figure (reg_sweep_all_architectures.png) and the Emp/Thm lambda table.

Everything is reported in the paper's lambda units (objective L + (lambda/2)(||B||^2 + ||A||^2)):
the sweep runs over `reg` = lambda / 2, so empirical values are converted with lambda = 2 * reg,
and the theoretical optimum (`optimal_reg` in the sweep output) is already in lambda units.

Usage:
    python plot_reg_sweep.py                       # reads outputs/reg_sweep
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import OUT_DIR, ROOT

ARCHS = ["vit", "clip", "resnet", "convnext"]
K_LIST = [5, 10, 20]


def load(input_dir, arch, k):
    """Swept lambda values (= 2 * reg), accuracies, and the theoretical optimal lambda."""
    data = json.loads((Path(input_dir) / f"reg_sweep_{arch}_K{k}.json").read_text())
    theory = data.pop("optimal_reg")
    regs = sorted(float(r) for r in data)
    acc = [data[str(r)]["accuracy"] for r in regs]
    return [2 * r for r in regs], acc, theory


def main(input_dir, out_file):
    plt.rcParams.update({"font.size": 35, "axes.titlesize": 40, "axes.labelsize": 35, "xtick.labelsize": 31, "ytick.labelsize": 31})
    colors = plt.cm.tab10.colors[:len(K_LIST)]
    fig, axs = plt.subplots(1, len(ARCHS), figsize=(9 * len(ARCHS), 7.5), constrained_layout=True)
    handles = [Line2D([], [], linestyle="none", label="Number of fine-tuning classes")]
    rows = {}
    for col, arch in enumerate(ARCHS):
        ax = axs[col]
        for color, k in zip(colors, K_LIST):
            lams, acc, theory = load(input_dir, arch, k)
            best = int(np.argmax(acc))
            line, = ax.plot(lams, acc, color=color, linewidth=5, zorder=5, label=rf"$\bar{{K}}={k}$")
            ax.plot(lams[best], acc[best], marker="*", markersize=40, color=color, zorder=5)
            ax.axvline(theory, color=color, linestyle="--", linewidth=5)
            if col == 0:
                handles.append(line)
            rows[arch, k] = (lams[best], theory)
        ax.set_xscale("log")
        ax.set_xlabel(r"Regularization effect $\lambda$")
        ax.set_title(arch.upper())
        if col == 0:
            ax.set_ylabel("Accuracy")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=len(K_LIST) + 1, frameon=False, fontsize=35)
    plt.suptitle("Accuracy vs Regularization λ across Architectures", y=1.1, fontsize=50)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"saved {out_file}\n\nEmpirical (Emp) and theoretical (Thm) optimal lambda")
    print(f"{'Model':<10}" + "".join(f"{'Kbar=' + str(k):>24}" for k in K_LIST))
    for arch in ["resnet", "vit", "convnext", "clip"]:
        print(f"{arch:<10}" + "".join(f"   Emp {rows[arch, k][0]:.6f} Thm {rows[arch, k][1]:.6f}" for k in K_LIST))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(OUT_DIR / "reg_sweep"))
    p.add_argument("--out", default=str(OUT_DIR / "reg_sweep_all_architectures.png"))
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    main(ROOT / args.input if not Path(args.input).is_absolute() else args.input, args.out)
