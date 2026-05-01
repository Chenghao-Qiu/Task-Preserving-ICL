#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="sentence-transformers/all-roberta-large-v1"
POOLING="mean"
BATCH_SIZE=16

DATASETS=(
    "sst2_checklist:128"
    "proverqa_easy:512"
    "proverqa_medium:512"
    "proverqa_hard:512"
    "problemathic_simple:256"
    "problemathic_complex:256"
)

for item in "${DATASETS[@]}"; do
    dataset="${item%%:*}"
    max_length="${item##*:}"

    python bert.py \
        --input "../exemplars/${dataset}.csv" \
        --output "bert/${dataset}_roberta_similarity.csv" \
        --model "$MODEL" \
        --pooling "$POOLING" \
        --batch-size "$BATCH_SIZE" \
        --max-length "$max_length" \
        --original-col text \
        --replaced-col adv_text \
        --show-worst 5
done
