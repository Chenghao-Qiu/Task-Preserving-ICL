#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python similarity/retrival.py \
    --datasets \
        sst2_checklist \
        proverqa_easy \
        proverqa_medium \
        proverqa_hard \
        problemathic_simple \
        problemathic_complex \
    --methods tfidf bm25 \
    --top-k 4 8 16 32 \
    --query-col text \
    --original-col text \
    --replaced-col adv_text \
    --output-dir similarity/retrival \
    --lowercase
