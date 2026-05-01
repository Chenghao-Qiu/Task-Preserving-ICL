#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Use two GPUs for large-model inference (mapped as cuda:0 and cuda:1 in-process).
export CUDA_VISIBLE_DEVICES=0,1

# Default model list.
MODELS=(
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-2-13b-chat-hf"
    "meta-llama/Llama-2-70b-chat-hf"
)
DATASETS=(
    "problemathic_simple"
    "problemathic_complex"
)
N_VALUES=(16)
NUM_RUNS=10
SAMPLING="random"
ADV_PLACEMENT="${ADV_PLACEMENT:-random}"
BACKEND="vllm"
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.95

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

    for Dataset in "${DATASETS[@]}"; do
        EXEMPLAR_POOL="exemplars/${Dataset}.csv"

        echo "------------------------------"
        echo "Dataset: ${Dataset}"
        echo "Exemplar pool: ${EXEMPLAR_POOL}"
        echo "------------------------------"

        python icl.py \
            --model "$Model" \
            --backend "$BACKEND" \
            --dataset "$Dataset" \
            --exemplar-pool "$EXEMPLAR_POOL" \
            --num-adv-exemplars 0 \
            --num-exemplars 0 \
            --adv-placement "$ADV_PLACEMENT" \
            --num-runs 1 \
            --test-data "data/${Dataset}/validation.csv" \
            --output "results/${ADV_PLACEMENT}/${OutName}/${Dataset}_0.json" \
            --device "auto" \
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --batch-size 16

        for N in "${N_VALUES[@]}"; do
            echo "---- dataset: ${Dataset}, num examples: ${N} ----"

            python icl.py \
                --model "$Model" \
                --backend "$BACKEND" \
                --dataset "$Dataset" \
                --exemplar-pool "$EXEMPLAR_POOL" \
                --num-exemplars "$N" \
                --num-adv-exemplars 0 \
                --sampling "$SAMPLING" \
                --adv-placement "$ADV_PLACEMENT" \
                --num-runs "$NUM_RUNS" \
                --test-data "data/${Dataset}/validation.csv" \
                --output "results/${ADV_PLACEMENT}/${OutName}/${Dataset}_${N}.json" \
                --device "auto" \
                --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
                --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
                --batch-size 16

        done
    done
done
