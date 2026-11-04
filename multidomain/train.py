"""Train the per-domain LoRA adapters of one (benchmark, seed) (paper Sec. 4.2, App. H).

Pipeline:
  1. Stratified 80/10/10 train/val/test split of every domain (numpy rng seeded with `seed`).
  2. base_with_head  = backbone + randomly initialized classification head (not trained).
  3. Pretrain-domain adapter: LoRA on query/value of layers 8-11, trained on the pretrain
     domain, then merged into the backbone -> base_with_pretrain (= W_pre in Alg. 1).
     Text models also train the classification head here (PEFT SEQ_CLS modules_to_save);
     for ViT the head stays at its random initialization.
  4. One adapter per remaining domain on top of base_with_pretrain: LoRA on query/value of
     all 12 layers plus the classifier (ViT: LoRA on the classifier; XLM-R: a trainable
     copy of the head). These are the adapters that get merged.
Every adapter: AdamW, cosine schedule with 5% warmup, grad-norm clip 1, best-val epoch kept.

Usage:
    python train.py --benchmark pacs --seed 1
"""
import argparse
import json
import os

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model

from common import BENCHMARKS, load_config, load_model, make_dataloader, run_dir, set_seed, stratified_split_indices, train_one_phase
from benchmarks import get_dataset


def lora_configs(cfg, is_text):
    task_type = TaskType.SEQ_CLS if is_text else getattr(TaskType, "IMAGE_CLASSIFICATION", TaskType.FEATURE_EXTRACTION)
    pretrain = {
        "task_type": task_type,
        "r": cfg["lora_r"],
        "lora_alpha": cfg["lora_alpha"],
        "lora_dropout": cfg["lora_dropout"],
        "target_modules": cfg["lora_target_modules"],
        "bias": "none",
        "layers_to_transform": cfg["lora_layers_to_transform"],
    }
    # Adapters that get merged also adapt the classifier, which lives outside the
    # transformer layers, so the layers_to_transform restriction is dropped for them.
    finetune = dict(pretrain)
    finetune["target_modules"] = pretrain["target_modules"] + (
        ["classifier.dense", "classifier.out_proj"] if is_text else ["classifier"])
    del finetune["layers_to_transform"]
    return pretrain, finetune


def train_adapter(base_path, lora_cfg, num_labels, is_text, cfg, tr_dl, va_dl, device):
    model = load_model(base_path, is_text, num_labels=num_labels)
    model = get_peft_model(model, LoraConfig(**lora_cfg))
    for n, p in model.named_parameters():
        if "lora" not in n and "modules_to_save" not in n:
            p.requires_grad = False
    train_one_phase(model, tr_dl, va_dl, device, lr=cfg["lora_lr"], epochs=cfg["lora_epochs"],
                    weight_decay=cfg["lora_weight_decay"])
    return model


def run(benchmark, seed, out_dir=None):
    cfg = load_config(benchmark)
    out_dir = str(out_dir or run_dir(benchmark, seed))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    handler = get_dataset(cfg["dataset"])
    ds = handler.load_data()
    domains, num_labels, is_text = handler.domains, handler.num_labels, handler.is_text
    processor = handler.get_processor(cfg["base_model"])
    collate_fn = handler.get_collate_fn(processor)

    rng = np.random.default_rng(seed)
    labels_column = ds["label"]
    domain_splits = {}
    for d, gidx in handler.build_domain_to_indices(ds).items():
        tr, va, te = stratified_split_indices([labels_column[i] for i in gidx], rng, cfg["val_ratio"], cfg["test_ratio"])
        domain_splits[d] = ([gidx[j] for j in tr], [gidx[j] for j in va], [gidx[j] for j in te])
        print(f"  {d}: train={len(tr)} val={len(va)} test={len(te)}")
    with open(f"{out_dir}/domain_splits.json", "w") as f:
        json.dump(domain_splits, f, indent=2)

    def loader(indices, shuffle):
        return make_dataloader(ds, indices, collate_fn, cfg["batch_size"], cfg["num_workers"], shuffle)

    base_dir = f"{out_dir}/base_with_head"
    base = load_model(cfg["base_model"], is_text, num_labels=num_labels)
    base.save_pretrained(base_dir)
    processor.save_pretrained(base_dir)
    del base

    lora_pretrain, lora_finetune = lora_configs(cfg, is_text)
    pretrain_domain = cfg["pretrain_task"]
    adapter_dirs = {}

    print(f"\n=== pretrain-domain adapter: {pretrain_domain} ===")
    tr, va, _ = domain_splits[pretrain_domain]
    model = train_adapter(base_dir, lora_pretrain, num_labels, is_text, cfg, loader(tr, True), loader(va, False), device)
    adapter_dirs[pretrain_domain] = f"{out_dir}/adapter_{pretrain_domain}"
    model.save_pretrained(adapter_dirs[pretrain_domain])
    merged_base_dir = f"{out_dir}/base_with_pretrain"
    model = model.merge_and_unload()
    model.save_pretrained(merged_base_dir)
    processor.save_pretrained(merged_base_dir)
    del model
    torch.cuda.empty_cache()

    for d in [d for d in domains if d != pretrain_domain]:
        print(f"\n=== domain adapter: {d} ===")
        tr, va, _ = domain_splits[d]
        model = train_adapter(merged_base_dir, lora_finetune, num_labels, is_text, cfg, loader(tr, True), loader(va, False), device)
        adapter_dirs[d] = f"{out_dir}/adapter_{d}"
        model.save_pretrained(adapter_dirs[d])
        del model
        torch.cuda.empty_cache()

    with open(f"{out_dir}/adapters.json", "w") as f:
        json.dump(adapter_dirs, f, indent=2)
    with open(f"{out_dir}/pretrain_info.json", "w") as f:
        json.dump({"pretrain_domain": pretrain_domain, "base_with_pretrain": merged_base_dir}, f, indent=2)
    with open(f"{out_dir}/train_config.json", "w") as f:
        json.dump({**cfg, "seed": seed}, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()
    run(args.benchmark, args.seed, args.out_dir)
