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

export CUDA_VISIBLE_DEVICES=0,1,2,3
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
DATASETS=(
    "proverqa_easy"
    "proverqa_medium"
    "proverqa_hard"
)
Method="noise"
N_VALUES=(8)
NUM_ADV_EXEMPLARS_LIST=(2 4 6 8)
NUM_RUNS=5
SAMPLING="random"
ADV_PLACEMENT="${ADV_PLACEMENT:-random}"
BACKEND="vllm"
TENSOR_PARALLEL_SIZE=4
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

        for NumExemplars in "${N_VALUES[@]}"; do
            for NumAdvExemplars in "${NUM_ADV_EXEMPLARS_LIST[@]}"; do
                if [ "$NumAdvExemplars" -gt "$NumExemplars" ]; then
                    continue
                fi

                echo "---- dataset: ${Dataset}, num examples: ${NumExemplars}, num adv examples: ${NumAdvExemplars} ----"

                OutputSuffix="${NumExemplars}_${NumAdvExemplars}"
                if [ "$NumAdvExemplars" -eq 0 ]; then
                    OutputName="${Dataset}_${Method}_${OutputSuffix}.json"
                else
                    OutputName="${Dataset}_a_${Method}_${OutputSuffix}.json"
                fi

                python icl_vlm.py \
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
                    --batch-size 16 \
                    --max-eval-samples 400
            done
        done
    done
done
