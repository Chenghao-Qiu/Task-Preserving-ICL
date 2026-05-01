#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASETS=(
    "sst2_checklist"
    "proverqa_easy"
    "proverqa_medium"
    "proverqa_hard"
    "problemathic_simple"
    "problemathic_complex"
)

for dataset in "${DATASETS[@]}"; do
    python overlap.py \
        --input "../exemplars/${dataset}.csv" \
        --output "overlap/${dataset}_overlap.csv" \
        --original-col text \
        --replaced-col adv_text \
        --lowercase
done
