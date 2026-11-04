#!/bin/bash
# All CIFAR-100 linear-probing experiments (Figures 1-5, Tables 3 and 4).
#   bash scripts/run_linear_probe.sh
set -euo pipefail
cd "$(dirname "$0")/../linear_probe"

python extract_features.py
python pretrain.py
python cosine_similarity.py
python reg_sweep.py --lr 0.5
python plot_reg_sweep.py
for k in 5 10 20; do python merge_grid.py --num_classes "$k"; done
python plot_merge_grid.py --num_classes 10
python plot_merge_grid.py
