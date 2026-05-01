SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

LEVEL=hard
CNOISE1=0
CNOISE2=0

NOISE1_LIST=(0 0.25 0.50 0.75 1)
NOISE2_LIST=(0 1)

for NOISE1 in "${NOISE1_LIST[@]}"; do
    for NOISE2 in "${NOISE2_LIST[@]}"; do
        python create.py \
            --input-file "outputs/final_data_${LEVEL}_${NOISE1}_${NOISE2}_${CNOISE1}_${CNOISE2}/test-0_2000.json" \
            --output-file "outputs/exemplars/proverqa_${LEVEL}_${NOISE1}_${NOISE2}_${CNOISE1}_${CNOISE2}.csv"
    done
done
