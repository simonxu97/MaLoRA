"""Training / evaluation utilities shared by the multi-domain scripts."""
import json
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForImageClassification, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
CONFIG_DIR = ROOT / "configs"
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 42, 123]
BENCHMARKS = ["pacs", "vlcs", "terra_incognita", "mnli", "amazon_reviews", "civil_comments"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(benchmark):
    return json.loads((CONFIG_DIR / f"{benchmark}.json").read_text())


def run_dir(benchmark, seed):
    return RUNS_DIR / f"{benchmark}_seed{seed}"


def load_model(path, is_text, num_labels=None):
    cls = AutoModelForSequenceClassification if is_text else AutoModelForImageClassification
    kwargs = {"ignore_mismatched_sizes": True}
    if num_labels is not None:
        kwargs["num_labels"] = num_labels
    return cls.from_pretrained(str(path), **kwargs)


def make_dataloader(ds, indices, collate_fn, batch_size, num_workers=4, shuffle=False):
    return DataLoader(Subset(ds, indices), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_fn, pin_memory=torch.cuda.is_available())


def stratified_split_indices(labels, rng, val_ratio=0.1, test_ratio=0.1):
    """Per-class shuffle, then the first test_ratio / next val_ratio of each class."""
    labels = np.asarray(labels)
    all_idx = np.arange(len(labels))
    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(labels):
        c_idx = all_idx[labels == c]
        rng.shuffle(c_idx)
        n_test = int(round(len(c_idx) * test_ratio))
        n_val = int(round(len(c_idx) * val_ratio))
        test_idx.extend(c_idx[:n_test].tolist())
        val_idx.extend(c_idx[n_test:n_test + n_val].tolist())
        train_idx.extend(c_idx[n_test + n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def forward_logits(model, batch, device):
    if hasattr(batch, "pixel_values"):
        return model(pixel_values=batch.pixel_values.to(device)).logits
    return model(input_ids=batch.input_ids.to(device), attention_mask=batch.attention_mask.to(device)).logits


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    correct, total = 0, 0
    for batch in dl:
        labels = batch.labels.to(device)
        correct += (forward_logits(model, batch, device).argmax(dim=-1) == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def evaluate_domains(model, loaders, domains, device):
    accs = {d: float(evaluate(model, loaders[d], device)) for d in domains if d in loaders}
    vals = list(accs.values())
    return {"domain_accuracies": accs, "avg": float(np.mean(vals)), "worst": float(np.min(vals)),
            "best": float(np.max(vals))}


def train_one_phase(model, train_dl, val_dl, device, lr, epochs, weight_decay, warmup_ratio=0.05, max_grad_norm=1.0):
    """AdamW + cosine schedule with 5% warmup and gradient clipping at norm 1.

    Runs all `epochs` and restores the parameters of the epoch with the best
    validation accuracy (no early stopping).
    """
    model.to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    total_steps = max(1, len(train_dl) * epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * warmup_ratio), total_steps)

    best_val, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_dl:
            labels = batch.labels.to(device)
            if hasattr(batch, "pixel_values"):
                out = model(pixel_values=batch.pixel_values.to(device), labels=labels)
            else:
                out = model(input_ids=batch.input_ids.to(device), attention_mask=batch.attention_mask.to(device),
                            labels=labels)
            optimizer.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            scheduler.step()

        val_acc = evaluate(model, val_dl, device)
        if val_acc > best_val + 1e-4:
            best_val, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch:02d} val_acc={val_acc:.4f} (best={best_val:.4f}@ep{best_epoch})", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val
