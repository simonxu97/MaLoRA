# MaLoRA: Margin-based LoRA Merging

Code for **"A Margin Perspective on LoRA: Robustness to Catastrophic Forgetting and Adapter Merging"** (NeurIPS 2026).

MaLoRA (Algorithm 1 of the paper) merges LoRA adapters trained on separate tasks into one model,

```
W = W_pre + s * sum_i beta_i * B_i A_i,      beta_i = Cbar_pre / Cbar_i   (0 if Cbar_i <= 0),
```

where `Cbar` is the average logit margin on a task's validation split (`Cbar_pre` for the pre-trained
model on its own task, `Cbar_i` for `W_pre + B_i A_i` on task `i`), and the single scale `s` is
chosen on validation. The implementation is `multidomain/merge.py`.

This repository contains the code behind every figure and table of the paper's experiments:

| Paper result | What it shows | Code |
|---|---|---|
| Figure 1, Figure 5 | Merged-model accuracy over (alpha_1, alpha_2): theory vs. grid search, CIFAR-100 linear probing | `linear_probe/merge_grid.py`, `plot_merge_grid.py` |
| Figure 4, Table 4 | Accuracy vs. regularization; empirical vs. theoretical optimal lambda | `linear_probe/reg_sweep.py`, `plot_reg_sweep.py` |
| Figures 2, 3, Table 3 | Cosine similarity of CIFAR-100 features | `linear_probe/cosine_similarity.py` |
| Table 1, Tables 5, 6 | MaLoRA vs. Soup / TA / TIES / DARE-TIES / DARE-Linear on 6 benchmarks x 10 seeds | `multidomain/` (`train.py`, `merge.py`, `merge_baselines.py`, `oracle.py`, `aggregate.py`) |
| Table 9 | Feature correlations within / across classes and domains | `multidomain/feature_correlations.py` |
| Table 10 | Merge-selection runtime | `multidomain/timing.py`, `analyze_timing.py` |

## Setup

```bash
conda create -n malora python=3.12 -y && conda activate malora
pip install -r requirements.txt
```

The paper runs used one NVIDIA RTX A5000 (24 GB) per job. If importing pandas or scipy after
torch fails with `GLIBCXX_3.4.29 not found` (older system `libstdc++`), preload the
environment's copy: `export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6`.

## Data

| Data | Source | Notes |
|---|---|---|
| CIFAR-100 | torchvision (downloaded to `linear_probe/data/`) | |
| PACS | HF `flwrlabs/pacs` | revision pinned in `multidomain/benchmarks.py` |
| MNLI | HF `multi_nli`, train split | genres as domains |
| Amazon Reviews | HF `amazon_polarity`, train split | four keyword-defined pseudo-domains |
| CivilComments | HF `google/civil_comments`, train split | domain = dominant toxicity attribute |
| VLCS, TerraIncognita | [DomainBed](https://github.com/facebookresearch/DomainBed) | `python -m domainbed.scripts.download --data_dir <dir>`; then `export MALORA_DATA=<dir>` (expects `<dir>/VLCS`, `<dir>/terra_incognita`) |

Every domain is split 80/10/10 (train/val/test), stratified by class and seeded by the run's
seed. For VLCS and TerraIncognita the images are read in the order listed in
`multidomain/manifests/`, which is the order of the paper runs; with that order the splits
are identical to the paper's.

## 1. Linear probing on CIFAR-100 (Sec. 4.1, App. H)

Frozen ResNet-50, ViT-B/16, ConvNeXt-T and CLIP ViT-B/32 features; LoRA on the final 20-way
linear classifier; pre-training task `D_pre` = 3 fine classes per superclass, fine-tuning tasks
`D_1`, `D_2` = one more fine class per superclass each (`linear_probe/splits/`). The
pre-trained head `W_pre` of each backbone is trained on `D_pre` by `pretrain.py`.

```bash
cd linear_probe
python extract_features.py                 # ~15 min, writes features/{train,test}_{arch}.pt
python pretrain.py                         # W_pre per backbone (seconds), writes checkpoints/W_pre_{arch}.pt
python cosine_similarity.py                # Figures 2, 3 and Table 3 (CPU, ~10 min)
python reg_sweep.py --lr 0.5               # Figure 4 / Table 4 data (~30 min per backbone)
python plot_reg_sweep.py                   # Figure 4 + Table 4
python merge_grid.py --num_classes 5       # Figure 5, first row (seconds per backbone)
python merge_grid.py --num_classes 10      # Figure 1 / Figure 5, second row
python merge_grid.py --num_classes 20      # Figure 5, third row
python plot_merge_grid.py --num_classes 10 # Figure 1
python plot_merge_grid.py                  # Figure 5
```

`bash scripts/run_linear_probe.sh` runs all of the above. Outputs go to `linear_probe/outputs/`.

## 2. Multi-domain adapter merging (Sec. 4.2, App. H)

ViT-B/16 (`google/vit-base-patch16-224-in21k`) on PACS, VLCS, TerraIncognita and XLM-R Base on
MNLI, Amazon Reviews, CivilComments; seeds 1-8, 42, 123. Hyperparameters are in
`multidomain/configs/<benchmark>.json`.

One (benchmark, seed):

```bash
bash scripts/run_multidomain.sh pacs 1
```

runs, in `multidomain/`:

| Step | Script | Output in `runs/<benchmark>_seed<seed>/` |
|---|---|---|
| train the pretrain-domain adapter and one adapter per other domain | `train.py` | `base_with_pretrain/`, `adapter_<domain>/` |
| Soup, TA and MaLoRA over the scale grid `s = k/49`, k = 1..49 | `merge.py` | `merge_fair/merge_results.csv` |
| TIES, DARE-TIES, DARE-Linear over density x lambda (80 configs each) | `merge_baselines.py` | `<method>_lamsweep/merge_results.csv` |
| per-domain accuracy before merging (normalization) | `oracle.py` | `oracle_accs.json` |
| merge-selection timing (4 CPU threads, 3 repeats) | `timing.py` | `merge_timing.json` |

All runs of the paper (one GPU job per line is the natural unit on a cluster):

```bash
for b in pacs vlcs terra_incognita mnli amazon_reviews civil_comments; do
  for s in 1 2 3 4 5 6 7 8 42 123; do bash scripts/run_multidomain.sh $b $s; done
done
```

Compute per (benchmark, seed) on one A5000, all five steps: about 1.5 h for PACS and Amazon
Reviews, 3-4 h for VLCS and TerraIncognita, and about 9 h for MNLI (whose genres have ~63k
training examples each); roughly 200 GPU-hours for all 60 runs. The TIES / DARE sweeps
(240 configurations) take more than half of it.

Then, once all 60 runs have finished, build the tables:

```bash
bash scripts/make_multidomain_tables.sh
```

which runs, in `multidomain/`, `aggregate.py` (Table 1, Tables 5 and 6; it stops with a list of
missing files if any run is incomplete), `analyze_timing.py` (Table 10) and
`feature_correlations.py` for each benchmark (Table 9; inference only, no training runs needed).

## Citation

```bibtex
@inproceedings{xu2026margin,
  title     = {A Margin Perspective on {LoRA}: Robustness to Catastrophic Forgetting and Adapter Merging},
  author    = {Xu, Ziqing and Min, Hancheng and MacDonald, Lachlan Ewen and Tarmoun, Salma and Mallada, Enrique and Su, Weijie and Vidal, Ren{\'e}},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
