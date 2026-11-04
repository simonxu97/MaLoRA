"""CIFAR-100 feature geometry (App. H, "Empirical motivation for the task geometry assumption").

Uses the training features from extract_features.py (no further mean subtraction).
  * heatmap figure + intra/inter table: 100 x 100 matrix of mean cosine similarity between
    classes, each entry estimated from 1,000 random sample pairs; intra-class = diagonal,
    inter-class = off-diagonal; mean and std over entries.
  * histogram figure: 100 random samples per class, 100,000 random pairs.

Both passes are seeded (heatmap: seed 0, histogram: seed 42).

Usage:
    python cosine_similarity.py
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUT_DIR, load_features

ARCH_ORDER = [("resnet", "ResNet"), ("vit", "ViT"), ("clip", "CLIP"), ("convnext", "ConvNeXt")]


def class_features(arch):
    feats, labels = load_features(arch)
    feats, labels = feats.numpy(), labels.numpy()
    return {c: feats[labels == c] for c in range(100)}


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return np.nan if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


def mean_pair_similarity(f1, f2, n_pairs=1000):
    sims = []
    for _ in range(n_pairs):
        if f1 is f2:
            i, j = np.random.choice(len(f1), 2, replace=False)
            s = cosine(f1[i], f1[j])
        else:
            s = cosine(f1[np.random.randint(len(f1))], f2[np.random.randint(len(f2))])
        if np.isfinite(s):
            sims.append(s)
    return float(np.mean(sims))


def similarity_matrix(cf):
    M = np.zeros((100, 100))
    for i in range(100):
        for j in range(i, 100):
            M[i, j] = M[j, i] = mean_pair_similarity(cf[i], cf[j])
    return M


def pairwise_similarities(cf, samples_per_class=100, max_pairs=100000):
    sampled = np.array([x for c in range(100)
                        for x in cf[c][np.random.choice(len(cf[c]), samples_per_class, replace=len(cf[c]) < samples_per_class)]])
    norms = np.linalg.norm(sampled, axis=1)
    pairs = set()
    while len(pairs) < max_pairs:
        i, j = np.random.choice(len(sampled), 2, replace=False)
        if norms[i] > 0 and norms[j] > 0:
            pairs.add((min(i, j), max(i, j)))
    return np.array([np.dot(sampled[i], sampled[j]) / (norms[i] * norms[j]) for i, j in pairs])


def main(heatmap_seed=0, out_dir=OUT_DIR / "cosine_similarity"):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 18, "axes.titlesize": 24, "axes.labelsize": 20, "xtick.labelsize": 16, "ytick.labelsize": 16})

    np.random.seed(heatmap_seed)
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    fig.suptitle("CIFAR-100 Feature Cosine-Similarity Analysis Across Architectures", fontsize=32, y=0.95)
    table = {}
    for k, (arch, title) in enumerate(ARCH_ORDER):
        M = similarity_matrix(class_features(arch))
        intra, inter = np.diag(M), M[np.triu_indices(100, k=1)]
        table[arch] = (intra.mean(), intra.std(), inter.mean(), inter.std())
        ax = axes[k // 2, k % 2]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_title(f"{title} Features", fontsize=24, pad=15)
        ax.set_xlabel("Class Index", fontsize=18)
        ax.set_ylabel("Class Index", fontsize=18)
        ax.set_xticks(np.arange(0, 100, 20))
        ax.set_yticks(np.arange(0, 100, 20))
        for b in range(5, 100, 5):
            ax.axhline(y=b - 0.5, color="white", linewidth=0.5, alpha=0.7)
            ax.axvline(x=b - 0.5, color="white", linewidth=0.5, alpha=0.7)
    cbar = fig.colorbar(im, cax=fig.add_axes([0.92, 0.15, 0.02, 0.7]))
    cbar.set_label("Cosine Similarity", rotation=270, labelpad=25, fontsize=22)
    plt.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.90, hspace=0.25, wspace=0.08)
    plt.savefig(out_dir / "cifar100_cosine_similarity_all_architectures.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    np.random.seed(42)
    plt.rcParams.update({"axes.titlesize": 26, "xtick.labelsize": 18, "ytick.labelsize": 18})
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    fig.suptitle("Pairwise Feature Cosine-Similarity Distributions Across Architectures", fontsize=36, y=0.95)
    hist = {}
    for k, (arch, title) in enumerate(ARCH_ORDER):
        sims = pairwise_similarities(class_features(arch))
        hist[arch] = (sims.mean(), sims.std(), sims.min(), sims.max())
        mu, sd = hist[arch][:2]
        ax = axes[k // 2, k % 2]
        ax.hist(sims, bins=50, alpha=0.7, density=True, color="skyblue", edgecolor="black", linewidth=0.5)
        ax.axvline(mu, color="red", linestyle="--", linewidth=5, label=f"Mean: {mu:.3f}")
        ax.axvline(mu + sd, color="orange", linestyle=":", linewidth=5, label=f"Mean + Std: {mu + sd:.3f}")
        ax.axvline(mu - sd, color="orange", linestyle=":", linewidth=5, label=f"Mean - Std: {mu - sd:.3f}")
        ax.set_title(f"{title} Feature Cosine Similarities", fontsize=26, pad=15)
        ax.set_xlabel("Cosine Similarity", fontsize=24)
        ax.set_ylabel("Density", fontsize=24)
        ax.tick_params(labelsize=24)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=22)
        ax.text(0.02, 0.98, f"Mean: {mu:.4f}\nStd: {sd:.4f}\nMin: {sims.min():.4f}\nMax: {sims.max():.4f}", transform=ax.transAxes,
                fontsize=22, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.35, wspace=0.25)
    plt.savefig(out_dir / "cifar100_feature_cosine_similarity_histograms.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    lines = ["Intra/inter-class cosine similarity, mean (std) over class (pair) entries"]
    lines += [f"  {a:9s} intra {t[0]:.4f} ({t[1]:.4f})   inter {t[2]:+.4f} ({t[3]:.4f})" for a, t in table.items()]
    lines += ["Pairwise sample cosine similarity (histogram figure)"]
    lines += [f"  {a:9s} mean {h[0]:+.4f}  std {h[1]:.4f}  min {h[2]:.4f}  max {h[3]:.4f}" for a, h in hist.items()]
    (out_dir / "cosine_similarity_stats.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
