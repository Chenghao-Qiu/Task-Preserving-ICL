#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

datasets=("sst2_checklist")
ADV_PLACEMENT_LIST=("head" "tail" "medium" "random" "custom")

for Dataset in "${datasets[@]}"; do
    echo "Running dataset: $Dataset"

    for AdvPlacement in "${ADV_PLACEMENT_LIST[@]}"; do
        echo "  ADV_PLACEMENT: $AdvPlacement"
        ADV_PLACEMENT="$AdvPlacement" bash "scripts/attention/ad_${Dataset}.sh"
    done
done
