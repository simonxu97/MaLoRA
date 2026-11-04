"""Soup, Task Arithmetic and MaLoRA (Alg. 1) on dense task vectors.

For each merged domain i, Delta_i = merge_and_unload(base_with_pretrain + adapter_i) - base_with_pretrain
(the LoRA product and the classifier update). The merged model is
    W = base_with_pretrain + sum_i c_i Delta_i,
with c_i = 1/N (Soup), c_i = s (TA) or c_i = s * beta_i (MaLoRA), where
    beta_i = Cbar_pre / Cbar_i  if Cbar_i > 0 else 0,
and Cbar is the average logit margin (true-class logit minus the largest other logit)
on the validation split: Cbar_pre for base_with_pretrain on the pretrain domain and
Cbar_i for base_with_pretrain + adapter_i on domain i.

Every configuration is evaluated on the val and test split of every domain and
written to <run_dir>/merge_fair/merge_results.csv; the scale s is picked on validation
by aggregate.py.

Usage:
    python merge.py --benchmark pacs --seed 1
"""
import argparse
import gc
import json
import os

import pandas as pd
import torch
from peft import PeftModel

from common import BENCHMARKS, evaluate_domains, forward_logits, load_config, load_model, make_dataloader, run_dir, set_seed
from benchmarks import get_dataset

# Scale grid s = k / 49, k = 1..49 (rounded to 6 decimals).
LAMBDA_GRID = [round(k / 49, 6) for k in range(1, 50)]


def materialize_adapter_delta(base_path, adapter_path, is_text):
    base_state = {k: v.detach().cpu().clone() for k, v in load_model(base_path, is_text).state_dict().items()}
    merged = PeftModel.from_pretrained(load_model(base_path, is_text), adapter_path).merge_and_unload()
    delta = {}
    for k, v in merged.state_dict().items():
        if k in base_state and base_state[k].shape == v.shape:
            d = v.detach().cpu() - base_state[k]
            if torch.any(d != 0):
                delta[k] = d
    del base_state, merged
    gc.collect()
    return delta


def apply_weighted_deltas(base_path, deltas, coefficients, is_text, device):
    base = load_model(base_path, is_text)
    state = base.state_dict()
    touched = set().union(*[d.keys() for d in deltas.values()])
    for name in touched:
        if name not in state:
            continue
        acc = torch.zeros_like(state[name], dtype=torch.float32)
        for domain, coef in coefficients.items():
            d = deltas[domain].get(name)
            if d is not None:
                acc = acc + float(coef) * d.to(acc.device, dtype=torch.float32)
        state[name] = (state[name].to(torch.float32) + acc).to(state[name].dtype)
    base.load_state_dict(state, strict=False)
    return base.to(device).eval()


@torch.no_grad()
def average_margin(model, dl, device):
    """Cbar(W; D): mean over samples of (true-class logit - largest other logit)."""
    model.eval()
    total, n = 0.0, 0
    for batch in dl:
        labels = batch.labels.to(device)
        logits = forward_logits(model, batch, device)
        idx = torch.arange(logits.size(0), device=device)
        true_logit = logits[idx, labels]
        other = logits.clone()
        other[idx, labels] = float("-inf")
        total += (true_logit - other.max(dim=-1).values).sum().item()
        n += logits.size(0)
    return total / max(n, 1)


def compute_margins(base_path, adapter_dirs, domain_splits, pretrain_domain, is_text, loader, device):
    margins = {pretrain_domain: average_margin(load_model(base_path, is_text).to(device),
                                               loader(domain_splits[pretrain_domain][1]), device)}
    for d, path in adapter_dirs.items():
        if d == pretrain_domain:
            continue
        model = PeftModel.from_pretrained(load_model(base_path, is_text), path).to(device)
        margins[d] = average_margin(model, loader(domain_splits[d][1]), device)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return margins


def malora_coefficients(margins, pretrain_domain, domains_to_merge):
    gamma_pre = margins[pretrain_domain]
    return {d: (gamma_pre / margins[d]) if margins[d] > 0 else 0.0 for d in domains_to_merge}


def result_row(name, domains, test_m, val_m):
    row = {"Method": name, "Density": "", "Rank": "", "Pretrain_Task": "",
           "Avg_Accuracy": test_m["avg"], "Worst_Accuracy": test_m["worst"], "Best_Accuracy": test_m["best"],
           "Val_Avg_Accuracy": val_m["avg"], "Val_Worst_Accuracy": val_m["worst"], "Val_Best_Accuracy": val_m["best"]}
    row.update({d: test_m["domain_accuracies"][d] for d in domains})
    row.update({f"val_{d}": val_m["domain_accuracies"][d] for d in domains})
    return row


def load_run(rdir):
    def read(name):
        with open(f"{rdir}/{name}") as f:
            return json.load(f)
    adapter_dirs, domain_splits, info = read("adapters.json"), read("domain_splits.json"), read("pretrain_info.json")
    pretrain_domain = info["pretrain_domain"]
    return adapter_dirs, domain_splits, pretrain_domain, info["base_with_pretrain"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run_dir", default=None, help="trained run (default: runs/<benchmark>_seed<seed>)")
    p.add_argument("--lambdas", type=float, nargs="+", default=LAMBDA_GRID)
    args = p.parse_args()

    cfg = load_config(args.benchmark)
    rdir = args.run_dir or str(run_dir(args.benchmark, args.seed))
    out_dir = f"{rdir}/merge_fair"
    os.makedirs(out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adapter_dirs, domain_splits, pretrain_domain, base_path = load_run(rdir)
    domains = sorted(adapter_dirs)
    domains_to_merge = sorted(d for d in adapter_dirs if d != pretrain_domain)

    handler = get_dataset(cfg["dataset"])
    ds = handler.load_data()
    collate_fn = handler.get_collate_fn(handler.get_processor(f"{rdir}/base_with_head"))
    is_text = handler.is_text

    def loader(idx):
        return make_dataloader(ds, idx, collate_fn, cfg["merge_batch_size"], cfg["merge_num_workers"])

    deltas = {d: materialize_adapter_delta(base_path, adapter_dirs[d], is_text) for d in domains_to_merge}
    margins = compute_margins(base_path, adapter_dirs, domain_splits, pretrain_domain, is_text, loader, device)
    beta = malora_coefficients(margins, pretrain_domain, domains_to_merge)
    with open(f"{out_dir}/malora_coefficients.json", "w") as f:
        json.dump({"pretrain_domain": pretrain_domain, "margins": margins, "beta": beta}, f, indent=2)
    print(f"margins: {margins}\nbeta: {beta}")

    test_loaders = {d: loader(domain_splits[d][2]) for d in domains}
    val_loaders = {d: loader(domain_splits[d][1]) for d in domains}

    experiments = [("simple_average", {d: 1.0 / len(domains_to_merge) for d in domains_to_merge})]
    for lam in args.lambdas:
        experiments.append((f"custom_linear_uniform_scaled_lam{lam}", {d: lam for d in domains_to_merge}))
        experiments.append((f"custom_linear_margin_ratio_drop_lam{lam}", {d: lam * beta[d] for d in domains_to_merge}))

    rows = []
    for i, (name, coefs) in enumerate(experiments, 1):
        merged = apply_weighted_deltas(base_path, deltas, coefs, is_text, device)
        test_m = evaluate_domains(merged, test_loaders, domains, device)
        val_m = evaluate_domains(merged, val_loaders, domains, device)
        print(f"[{i}/{len(experiments)}] {name}: val={val_m['avg']:.4f} test={test_m['avg']:.4f}", flush=True)
        rows.append(result_row(name, domains, test_m, val_m))
        del merged
        gc.collect()
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(f"{out_dir}/merge_results.csv", index=False)
    print(f"Saved {out_dir}/merge_results.csv")


if __name__ == "__main__":
    main()
