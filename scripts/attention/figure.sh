#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

export CUDA_VISIBLE_DEVICES=0,1

# Default model list.
MODELS=(
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-2-70b-chat-hf"
)
Dataset="sst2_checklist_matched"
Method="checklist"
EXEMPLAR_POOL="exemplars/${Dataset}.csv"
N_VALUES=(32)
NUM_ADV_EXEMPLARS_LIST=(28)
NUM_RUNS=5
SAMPLING="random"
ADV_PLACEMENT="${ADV_PLACEMENT:-head}"
BACKEND="transformers"
ATTENTION_MAP_MODE="unit"

if [ "$#" -gt 0 ]; then
    MODELS=("$@")
fi

for Model in "${MODELS[@]}"; do
    OutName="${Model##*/}"
    mkdir -p "results/${ADV_PLACEMENT}/${OutName}"

    echo "=============================="
    echo "Running model: ${Model}"
    echo "Output dir: results/${ADV_PLACEMENT}/${OutName}"
    echo "=============================="

    for NumExemplars in "${N_VALUES[@]}"; do
        for NumAdvExemplars in "${NUM_ADV_EXEMPLARS_LIST[@]}"; do
            if [ "$NumAdvExemplars" -gt "$NumExemplars" ]; then
                continue
            fi

            echo "---- num examples: ${NumExemplars}, num adv examples: ${NumAdvExemplars} ----"

            OutputSuffix="${NumExemplars}_${NumAdvExemplars}"
            if [ "$NumAdvExemplars" -eq 0 ]; then
                OutputName="${Dataset}_${Method}_${OutputSuffix}.json"
            else
                OutputName="${Dataset}_a_${Method}_${OutputSuffix}.json"
            fi

            python icl.py \
                --model "$Model" \
                --backend "$BACKEND" \
                --dataset "$Dataset" \
                --exemplar-pool "$EXEMPLAR_POOL" \
                --num-exemplars "$NumExemplars" \
                --num-adv-exemplars "$NumAdvExemplars" \
                --sampling "$SAMPLING" \
                --adv-placement "$ADV_PLACEMENT" \
                --num-runs "$NUM_RUNS" \
                --test-data "data/${Dataset}/validation.csv" \
                --max-eval-samples 1 \
                --output "/dev/null" \
                --device "auto" \
                --batch-size 16 \
                --attention-map \
                --attention-map-mode ${ATTENTION_MAP_MODE} \
                --attention-map-publication \
                --attention-map-dir "results/${ADV_PLACEMENT}/${OutName}/${Dataset}_${Method}_${OutputSuffix}_attention_maps_publication"

        done
    done
done
