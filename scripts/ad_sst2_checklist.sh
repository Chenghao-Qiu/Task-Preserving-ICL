#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

export CUDA_VISIBLE_DEVICES=0,1

# Default model list.
MODELS=(
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-2-13b-chat-hf"
    "meta-llama/Llama-2-70b-chat-hf"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.1-70B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
    "Qwen/Qwen2.5-32B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
)
Dataset="sst2_checklist"
Method="checklist"
EXEMPLAR_POOL="exemplars/${Dataset}.csv"
N_VALUES=(32)
NUM_ADV_EXEMPLARS_LIST=(8 16 24 32)
NUM_RUNS=100
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
                --output "results/${ADV_PLACEMENT}/${OutName}/${OutputName}" \
                --device "auto" \
                --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
                --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
                --batch-size 16

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
                --test-data "data/${Dataset}/validation_adv.csv" \
                --output "results/${ADV_PLACEMENT}/${OutName}/adv_${OutputName}" \
                --device "auto" \
                --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
                --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
                --batch-size 16
        done
    done
done
