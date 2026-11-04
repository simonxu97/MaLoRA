"""Effect of regularization (App. H, "Effect of regularization": reg-sweep figure and lambda table).

For each backbone and Kbar in {5, 10, 20}: fine-tune a rank-Kbar LoRA on the first Kbar
superclasses of D_1 (the "tune1" split) with full-batch SGD, for 50 log-spaced values of
`reg` in [1e-5, 1/20], and record accuracy and normalized min-margin on D_pre u D_ft.
Also records the theoretical optimum `optimal_reg` (paper lambda units; see common.py).

Note: the loss is CE + reg (||B||^2 + ||A||^2), so paper lambda = 2 * reg.

Usage:
    python reg_sweep.py --arch vit --lr 0.5
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from common import (ARCHS, CKPT_DIR, DEVICE, N_PER_CLASS, N_SUPER, OUT_DIR, LoRALinear, etf_classifier, load_features,
                    normalized_min_margin, seed_everything, split_indices, superclass_ids, theory_optimal_lambda)

REG_GRID = np.logspace(np.log10(1e-5), np.log10(1 / N_SUPER), 50).tolist()


def sweep(arch, lr, epochs=3000, k_list=(5, 10, 20), split="tune1", seed=0, out_dir=OUT_DIR / "reg_sweep"):
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    W_pre = torch.load(CKPT_DIR / f"W_pre_{arch}.pt")["W"].to(DEVICE)
    feats, labs = load_features(arch)

    for m in k_list:
        idx = split_indices(labs, split)
        super_ids = superclass_ids(labs[idx])
        keep = super_ids < m
        idx, super_ids = idx[keep], super_ids[keep]
        order = super_ids.argsort()
        x = feats[idx[order]].to(DEVICE)
        y = super_ids[order].to(DEVICE)

        idx_pre = split_indices(labs, "pretrain")
        x_pre = feats[idx_pre].to(DEVICE)
        y_pre = superclass_ids(labs[idx_pre]).to(DEVICE)

        W_etf = etf_classifier(x, m)
        gamma_ft = normalized_min_margin(x @ W_etf.T, y, W_etf)
        gamma_pre = normalized_min_margin(x_pre @ W_pre.T, y_pre, W_pre)
        optimal_reg = theory_optimal_lambda(torch.norm(W_pre, "fro"), gamma_pre, gamma_ft, K=N_SUPER, m=m, n=N_PER_CLASS)

        results = {}
        for reg in REG_GRID:
            model = LoRALinear(W_pre, m).to(DEVICE)
            opt = torch.optim.SGD(model.parameters(), lr=lr)
            ce = nn.CrossEntropyLoss()
            for _ in range(epochs):
                opt.zero_grad()
                loss = ce(model(x), y) + reg * (torch.norm(model.B, p="fro") ** 2 + torch.norm(model.A, p="fro") ** 2)
                loss.backward()
                opt.step()

            # Accuracy and normalized min-margin on D_pre u D_ft
            with torch.no_grad():
                X, Y = torch.cat([x_pre, x]), torch.cat([y_pre, y])
                lg = model(X)
                W_full = model.W_pre + model.delta()
                results[str(reg)] = {"margin": normalized_min_margin(lg, Y, W_full),
                                     "accuracy": 100 * (lg.argmax(1) == Y).sum().item() / len(Y)}
        results["optimal_reg"] = optimal_reg
        best = max((k for k in results if k != "optimal_reg"), key=lambda k: results[k]["accuracy"])
        print(f"[{arch} K={m}] empirical lambda={2 * float(best):.6f} (reg {float(best):.6f}, acc {results[best]['accuracy']:.2f}%)  "
              f"theoretical lambda={optimal_reg:.6f}  gamma_pre={gamma_pre:+.4f} gamma_ft={gamma_ft:+.4f}", flush=True)
        (out_dir / f"reg_sweep_{arch}_K{m}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=ARCHS, nargs="+", default=ARCHS)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    for a in args.arch:
        sweep(a, args.lr, args.epochs, seed=args.seed)
