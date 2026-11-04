#!/bin/bash
# Tables 1, 5, 6, 9 and 10 from the runs of scripts/run_multidomain.sh (all 6 benchmarks x 10 seeds).
#   bash scripts/make_multidomain_tables.sh
set -euo pipefail
cd "$(dirname "$0")/../multidomain"

python aggregate.py        # Table 1, Tables 5 and 6
python analyze_timing.py   # Table 10
for b in pacs vlcs terra_incognita mnli amazon_reviews civil_comments; do
  python feature_correlations.py --benchmark "$b"   # Table 9
done
