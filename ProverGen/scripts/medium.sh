SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

LEVEL=medium

# python3 logic_skeleton_generator_multiprocess.py --mode $LEVEL --num $NUM --output_dir outputs/logic_data

# python3 logic_skeleton_translator_resume.py \
#     --model_name meta-llama/Llama-3.1-70B-Instruct \
#     --data_dir outputs/logic_data \
#     --num $NUM --start 0 --end $NUM \
#     --output_dir outputs/translated_data \
#     --mode $LEVEL \
#     --base_url http://localhost:6417/v1 --api_key EMPTY

python3 fol_problem_generator.py \
    --model_name meta-llama/Llama-3.1-70B-Instruct \
    --filepath outputs/translated_data/$LEVEL-${NUM}-0_${NUM}.json \
    --start 0 --end $NUM \
    --output_dir outputs/final_data_${LEVEL}_${NOISE1}_${NOISE2}_${CNOISE1}_${CNOISE2} \
    --mode normal_generation \
    --noise1 $NOISE1 --noise2 $NOISE2 \
    --cot_noise1 $CNOISE1 --cot_noise2 $CNOISE2\
    --base_url http://localhost:6417/v1 --api_key EMPTY
