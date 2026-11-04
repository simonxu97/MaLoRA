"""Merged-model accuracy heatmaps: Fig. 1 (one Kbar, 1 x 4) and the appendix 3 x 4 figure.

Red star: (alpha1, alpha2) from merge_grid.py; red triangle: grid argmax.

Usage:
    python plot_merge_grid.py --num_classes 10        # Fig. 1
    python plot_merge_grid.py                         # 3 x 4 figure (Kbar = 5, 10, 20)
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from common import OUT_DIR, ROOT

ARCHS = ["vit", "resnet", "convnext", "clip"]


def load(input_dir, arch, k):
    d = json.loads((Path(input_dir) / f"merge_grid_{arch}_K{k}.json").read_text())
    acc = np.array(d["grid"]["accuracy"])
    A1, A2 = np.array(d["grid"]["alpha1"]), np.array(d["grid"]["alpha2"])
    gi, gj = np.unravel_index(acc.argmax(), acc.shape)
    return d, acc, A1, A2, A1[gi], A2[gj]


def main(input_dir, k_list, out_file):
    data = {(k, a): load(input_dir, a, k) for k in k_list for a in ARCHS}
    norm = plt.Normalize(min(v[1].min() for v in data.values()), max(v[1].max() for v in data.values()))
    single = len(k_list) == 1
    plt.rcParams.update({"font.size": 28, "axes.titlesize": 32, "axes.labelsize": 28, "xtick.labelsize": 24, "ytick.labelsize": 24})
    fig, axes = plt.subplots(len(k_list), len(ARCHS), figsize=(7 * len(ARCHS), 6 * len(k_list)), squeeze=False)
    im = None
    for row, k in enumerate(k_list):
        for col, arch in enumerate(ARCHS):
            d, acc, A1, A2, b1, b2 = data[k, arch]
            ax = axes[row, col]
            im = ax.imshow(acc, extent=[A2.min(), A2.max(), A1.min(), A1.max()], origin="lower", aspect="auto", norm=norm)
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set_yticks(np.linspace(0, 1, 6))
            if row == 0:
                ax.set_title(arch.upper())
            if col == 0:
                ax.set_ylabel(r"$\alpha_1$" if single else rf"$\alpha_1$ ($\bar{{K}}={k}$)")
            if row == len(k_list) - 1:
                ax.set_xlabel(r"$\alpha_2$")
            ax.plot(d["alpha2"], d["alpha1"], marker="*", markersize=20, markeredgecolor="k", markerfacecolor="red")
            ax.plot(b2, b1, marker="^", markersize=15, markeredgecolor="k", markerfacecolor="red")
            print(f"Kbar={k:2d} {arch:8s} star=({d['alpha1']:.3f}, {d['alpha2']:.3f})  grid argmax=({b1:.3f}, {b2:.3f})  "
                  f"acc at argmax {acc.max():.2f}%")
    cb = fig.colorbar(im, ax=axes[:, -1], fraction=0.046, pad=0.02)
    cb.ax.set_ylabel("Accuracy (%)", rotation=90, fontsize=32)
    star = mlines.Line2D([], [], marker="*", markersize=18, markeredgecolor="k", markerfacecolor="red", linestyle="none",
                         label=r"optimal $(\alpha_1,\alpha_2)$ by Theorem 3.3")
    tri = mlines.Line2D([], [], marker="^", markersize=16, markeredgecolor="k", markerfacecolor="red", linestyle="none",
                        label=r"optimal $(\alpha_1,\alpha_2)$ by grid search")
    fig.legend(handles=[star, tri], bbox_to_anchor=(0.5, 0.02 if not single else -0.02), loc="upper center", ncol=2,
               frameon=False, fontsize=28)
    plt.subplots_adjust(top=0.88 if not single else 0.80, bottom=0.10 if not single else 0.18, left=0.08, right=0.85,
                        hspace=0.35, wspace=0.25)
    title = (rf"Merged-model Accuracy across Architectures when $\bar{{K}}={k_list[0]}$" if single
             else r"Merged-model Accuracy across Architectures and $\bar{K}$ Values")
    plt.suptitle(title, fontsize=38, y=0.94 if not single else 0.97)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"saved {out_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(OUT_DIR / "merge_grid"))
    p.add_argument("--num_classes", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    name = f"unified_model_accuracy_{args.num_classes[0]}.png" if len(args.num_classes) == 1 else "unified_model_accuracy_3x4.png"
    out = Path(args.out) if args.out else OUT_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    main(ROOT / args.input if not Path(args.input).is_absolute() else args.input, args.num_classes, out)
