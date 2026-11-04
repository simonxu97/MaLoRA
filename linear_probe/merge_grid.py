"""Optimal mixing coefficients: theory vs. grid search (Fig. 1 and the 3x4 appendix figure).

For one backbone and Kbar: train LoRA adapters (B_1 A_1) on D_1 and (B_2 A_2) on D_2
(first Kbar superclasses), then evaluate W_pre + a1 B_1A_1 + a2 B_2A_2 on
D_pre u D_1 u D_2 over a 50 x 50 grid of (a1, a2) in [0, 1]^2.

`alpha1`, `alpha2` are the coefficients plotted as the red star:
    alpha_i = ||W_pre|| gamma(W_pre; D_pre) / (gamma(W_pre + B_iA_i; D_i) ||B_iA_i||).

Usage:
    python merge_grid.py --arch vit --num_classes 10 --rank 10 --reg 1e-7 --epochs 200 --lr 0.1
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from common import (ARCHS, CKPT_DIR, DEVICE, N_SUPER, OUT_DIR, LoRALinear, load_features, normalized_min_margin,
                    seed_everything, select)


def finetune(W_pre, x, y, rank, reg, epochs, lr):
    model = LoRALinear(W_pre, rank).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = ce(model(x), y) + reg * (torch.norm(model.B, p="fro") ** 2 + torch.norm(model.A, p="fro") ** 2)
        loss.backward()
        opt.step()
    return model.B.detach(), model.A.detach()


def run(arch, num_classes, rank, reg, epochs, lr, seed=0, out_dir=OUT_DIR / "merge_grid"):
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    W_pre = torch.load(CKPT_DIR / f"W_pre_{arch}.pt")["W"].to(DEVICE)
    feats, labs = load_features(arch)
    data = {name: [t.to(DEVICE) for t in select(feats, labs, name, num_classes if name != "pretrain" else N_SUPER)]
            for name in ["pretrain", "tune1", "tune2"]}

    deltas = {}
    for stage, split in [(1, "tune1"), (2, "tune2")]:
        B, A = finetune(W_pre, *data[split], rank, reg, epochs, lr)
        torch.save({"B": B.cpu(), "A": A.cpu(), "rank": rank}, out_dir / f"B{stage}_A{stage}_{arch}_K{num_classes}.pt")
        deltas[stage] = (B.cpu() @ A.cpu()).to(DEVICE)  # B @ A is formed on the CPU

    x_pre, y_pre = data["pretrain"]
    gamma_pre = normalized_min_margin(x_pre @ W_pre.T, y_pre, W_pre)
    rho_pre = torch.norm(W_pre, p="fro").item()
    alpha, margins = {}, {"P": gamma_pre}
    for stage, split in [(1, "tune1"), (2, "tune2")]:
        x, y = data[split]
        W = W_pre + deltas[stage]
        gamma_i = normalized_min_margin(x @ W.T, y, W)
        margins[f"F{stage}"] = gamma_i
        alpha[stage] = rho_pre * gamma_pre / (gamma_i * torch.norm(deltas[stage], p="fro").item())

    X = torch.cat([data[s][0] for s in ["pretrain", "tune1", "tune2"]])
    Y = torch.cat([data[s][1] for s in ["pretrain", "tune1", "tune2"]])
    A1 = np.linspace(0, 1, 50)
    A2 = np.linspace(0, 1, 50)
    acc_grid = np.zeros((len(A1), len(A2)))
    marg_grid = np.zeros_like(acc_grid)
    with torch.no_grad():
        for i, a1 in enumerate(A1):
            for j, a2 in enumerate(A2):
                W = W_pre + a1 * deltas[1] + a2 * deltas[2]
                logits = X @ W.T
                acc_grid[i, j] = (logits.argmax(1) == Y).float().mean().item() * 100
                marg_grid[i, j] = normalized_min_margin(logits, Y, W)

    gi, gj = np.unravel_index(acc_grid.argmax(), acc_grid.shape)
    result = {
        "alpha1": alpha[1], "alpha2": alpha[2],
        "margins": margins,
        "hyperparameters": {"rank": rank, "reg": reg, "epochs": epochs, "lr": lr, "seed": seed},
        "grid": {"alpha1": A1.tolist(), "alpha2": A2.tolist(), "margin": marg_grid.tolist(), "accuracy": acc_grid.tolist()},
    }
    (out_dir / f"merge_grid_{arch}_K{num_classes}.json").write_text(json.dumps(result, indent=2))
    print(f"[{arch} K={num_classes}] star (alpha1, alpha2) = ({alpha[1]:.4f}, {alpha[2]:.4f})  "
          f"grid argmax = ({A1[gi]:.4f}, {A2[gj]:.4f}) acc {acc_grid[gi, gj]:.2f}%", flush=True)
    return result


# Settings for the paper's figures.
PAPER_SETTINGS = {
    5: dict(rank=5, reg=1e-7, epochs=200, lr=0.1),
    10: dict(rank=20, reg=1e-6, epochs=1000, lr=0.1),
    20: dict(rank=20, reg=1e-6, epochs=1000, lr=0.1),
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=ARCHS, nargs="+", default=ARCHS)
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--reg", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    hp = dict(PAPER_SETTINGS.get(args.num_classes, {}))
    hp.update({k: v for k, v in vars(args).items() if k in ("rank", "reg", "epochs", "lr") and v is not None})
    for a in args.arch:
        run(a, args.num_classes, seed=args.seed, **hp)
