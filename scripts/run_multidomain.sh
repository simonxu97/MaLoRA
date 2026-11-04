#!/bin/bash
# Train, merge (Soup / TA / MaLoRA), run the TIES / DARE baselines, measure oracle
# accuracies and time the merges for one (benchmark, seed).
#   bash scripts/run_multidomain.sh pacs 1
# Benchmarks: pacs vlcs terra_incognita mnli amazon_reviews civil_comments
# Seeds used in the paper: 1 2 3 4 5 6 7 8 42 123
set -euo pipefail
BENCH=$1
SEED=$2
cd "$(dirname "$0")/../multidomain"

python train.py --benchmark "$BENCH" --seed "$SEED"
python merge.py --benchmark "$BENCH" --seed "$SEED"
python merge_baselines.py --benchmark "$BENCH" --seed "$SEED"
python oracle.py --benchmark "$BENCH" --seed "$SEED"
python timing.py --benchmark "$BENCH" --seed "$SEED"
