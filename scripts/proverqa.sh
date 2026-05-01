#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Use two GPUs for large-model inference (mapped as cuda:0 and cuda:1 in-process).
export CUDA_VISIBLE_DEVICES=0,1

# Default model list.
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.1-70B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
    "Qwen/Qwen2.5-32B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
)
DATASETS=(
    "proverqa_easy"
    "proverqa_medium"
    "proverqa_hard"
)
N_VALUES=(8)
NUM_RUNS=5
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
            --batch-size 16 \
            --max-eval-samples 400

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
                --batch-size 16 \
                --max-eval-samples 400

        done
    done
done
