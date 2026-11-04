"""Pre-train the linear head W_pre (20 superclass outputs, no bias) on D_pre with full-batch Adam.

Writes checkpoints/W_pre_{arch}.pt, the head that reg_sweep.py and merge_grid.py fine-tune.

Usage:
    python pretrain.py               # all four backbones
    python pretrain.py --arch vit
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from common import ARCHS, CKPT_DIR, DEVICE, N_SUPER, load_features, seed_everything, select


def pretrain(arch, epochs=600, lr=0.2, seed=0):
    seed_everything(seed)
    feats, labs = load_features(arch)
    x, y = [t.to(DEVICE) for t in select(feats, labs, "pretrain")]
    head = nn.Linear(x.shape[1], N_SUPER, bias=False).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = ce(head(x), y)
        loss.backward()
        opt.step()
    acc = (head(x).argmax(1) == y).float().mean().item() * 100
    print(f"[{arch}] loss {loss.item():.4f}  train acc {acc:.2f}%  ||W_pre||_F {head.weight.norm().item():.1f}")
    return head.weight.detach().cpu()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=ARCHS, nargs="+", default=ARCHS)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=str(CKPT_DIR))
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for a in args.arch:
        torch.save({"W": pretrain(a, args.epochs, args.lr, args.seed)}, out_dir / f"W_pre_{a}.pt")
