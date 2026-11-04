"""Per-domain "oracle" test accuracy, the denominator of normalized accuracy (Ilharco et al.).

oracle[pretrain domain] = accuracy of base_with_pretrain on its test split;
oracle[d]               = accuracy of base_with_pretrain + adapter_d (unmerged) on d's test split.
Writes <run_dir>/oracle_accs.json.

Usage:
    python oracle.py --benchmark pacs --seed 1
"""
import argparse
import gc
import json

import torch
from peft import PeftModel

from benchmarks import get_dataset
from common import BENCHMARKS, evaluate, load_config, load_model, make_dataloader, run_dir, set_seed
from merge import load_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run_dir", default=None)
    args = p.parse_args()

    cfg = load_config(args.benchmark)
    rdir = args.run_dir or str(run_dir(args.benchmark, args.seed))
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adapter_dirs, domain_splits, pretrain_domain, base_path = load_run(rdir)
    handler = get_dataset(cfg["dataset"])
    ds = handler.load_data()
    collate_fn = handler.get_collate_fn(handler.get_processor(f"{rdir}/base_with_head"))

    def test_loader(d):
        return make_dataloader(ds, domain_splits[d][2], collate_fn, cfg["oracle_batch_size"], cfg["merge_num_workers"])

    oracle = {pretrain_domain: float(evaluate(load_model(base_path, handler.is_text).to(device), test_loader(pretrain_domain), device))}
    for d, path in adapter_dirs.items():
        if d == pretrain_domain:
            continue
        model = PeftModel.from_pretrained(load_model(base_path, handler.is_text), path).to(device)
        oracle[d] = float(evaluate(model, test_loader(d), device))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print(oracle)
    with open(f"{rdir}/oracle_accs.json", "w") as f:
        json.dump({"pretrain_domain": pretrain_domain, "oracle_accs": oracle}, f, indent=2)


if __name__ == "__main__":
    main()
