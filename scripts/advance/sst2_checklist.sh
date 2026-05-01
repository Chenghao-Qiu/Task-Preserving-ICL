#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets

export CUDA_DRIVER_LIB=/usr/lib/x86_64-linux-gnu
export LIBRARY_PATH=$CUDA_DRIVER_LIB:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_DRIVER_LIB:$LD_LIBRARY_PATH
export LDFLAGS="-L$CUDA_DRIVER_LIB $LDFLAGS"

# Use two GPUs for large-model inference (mapped as cuda:0 and cuda:1 in-process).
export CUDA_VISIBLE_DEVICES=2,3
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0

# Default model list.
MODELS=(
    "google/gemma-4-E2B-it"
    "google/gemma-4-E4B-it"
    "google/gemma-4-31B-it"
    "Qwen/Qwen3.5-2B"
    "Qwen/Qwen3.5-9B"
    "Qwen/Qwen3.5-27B"
)
Dataset="sst2_checklist"
EXEMPLAR_POOL="exemplars/${Dataset}.csv"
N_VALUES=(32)
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

    python icl_vlm.py \
        --model "$Model" \
        --backend "$BACKEND" \
        --dataset "$Dataset" \
        --exemplar-pool "$EXEMPLAR_POOL" \
        --num-exemplars 0 \
        --num-adv-exemplars 0 \
        --adv-placement "$ADV_PLACEMENT" \
        --num-runs 1 \
        --test-data "data/${Dataset}/validation.csv" \
        --output "results/${ADV_PLACEMENT}/${OutName}/${Dataset}_0.json" \
        --device "auto" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --batch-size 16

    python icl_vlm.py \
        --model "$Model" \
        --backend "$BACKEND" \
        --dataset "$Dataset" \
        --exemplar-pool "$EXEMPLAR_POOL" \
        --num-exemplars 0 \
        --num-adv-exemplars 0 \
        --adv-placement "$ADV_PLACEMENT" \
        --num-runs 1 \
        --test-data "data/${Dataset}/validation_adv.csv" \
        --output "results/${ADV_PLACEMENT}/${OutName}/${Dataset}_adv_0.json" \
        --device "auto" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --batch-size 16

    for N in "${N_VALUES[@]}"; do
        echo "---- num examples: ${N} ----"

        python icl_vlm.py \
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
        
        python icl_vlm.py \
            --model "$Model" \
            --backend "$BACKEND" \
            --dataset "$Dataset" \
            --exemplar-pool "$EXEMPLAR_POOL" \
            --num-exemplars "$N" \
            --num-adv-exemplars 0 \
            --sampling "$SAMPLING" \
            --adv-placement "$ADV_PLACEMENT" \
            --num-runs "$NUM_RUNS" \
            --test-data "data/${Dataset}/validation_adv.csv" \
            --output "results/${ADV_PLACEMENT}/${OutName}/${Dataset}_adv_${N}.json" \
            --device "auto" \
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --batch-size 16
    done
done
