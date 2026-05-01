CUDA_VISIBLE_DEVICES=0,1,2,3
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --host 0.0.0.0 \
  --port 6417 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95
