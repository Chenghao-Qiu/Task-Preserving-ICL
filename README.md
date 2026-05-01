# When Correct Demonstrations Hurt: Rethinking the Role of Exemplars in In-Context Learning

This repository contains the code and data processing utilities for the paper **"When Correct Demonstrations Hurt: Rethinking the Role of Exemplars in In-Context Learning"**.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Main Arguments](#main-arguments)
- [Example Usage](#example-usage)
- [Supported Tasks](#supported-tasks)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Repository Structure

- `icl.py`: main ICL evaluation script.
- `utils/`: dataset configuration, exemplar sampling, and prompt utilities.
- `scripts/`: experiment launch scripts.
- `data/`: evaluation datasets.
- `exemplars/`: exemplar pools used for in-context demonstrations.

## Installation

Install vLLM before running experiments with the vLLM backend:

```bash
pip install vllm
```

## Main Arguments

Common arguments for `icl.py`:

| Argument | Meaning |
| --- | --- |
| `--dataset` | Dataset name or alias. The script uses this to find the default test data, exemplar pool, labels, and task type from `utils/dataset_config.py`. |
| `--model` | Hugging Face model name or local model path. |
| `--backend` | Inference backend. Use `transformers` for Hugging Face inference or `vllm` for vLLM inference. |
| `--num-exemplars` | Number of in-context exemplars sampled for each run. |
| `--num-adv-exemplars` | Number of adversarial exemplars mixed into the sampled exemplars for each run. Use `0` for normal ICL without adversarial exemplars. |
| `--adv-placement` | Where to place adversarial exemplars in the prompt. Options: `random`, `head`, `medium`, `tail`, or `custom`. |

## Example Usage

Run a basic ICL evaluation:

```bash
python icl.py \
  --dataset sst2_checklist \
  --model Qwen/Qwen2.5-7B-Instruct \
  --backend transformers \
  --num-exemplars 16 \
  --num-adv-exemplars 0 \
  --num-runs 100 \
  --output results/sst2_checklist.json
```

Run with adversarial exemplars:

```bash
python icl.py \
  --dataset sst2_checklist \
  --model Qwen/Qwen2.5-7B-Instruct \
  --backend transformers \
  --num-exemplars 16 \
  --num-adv-exemplars 4 \
  --adv-placement random \
  --num-runs 100 \
  --output results/sst2_checklist_adv.json
```

For larger models, use the vLLM backend:

```bash
python icl.py \
  --dataset proverqa_easy \
  --model meta-llama/Llama-2-70b-chat-hf \
  --backend vllm \
  --tensor-parallel-size 2 \
  --num-exemplars 16 \
  --num-runs 100 \
  --output results/proverqa_easy.json
```

## Supported Tasks

The repository includes configurations for sentiment classification, natural language inference, paraphrase detection, mathematical reasoning, and logical reasoning datasets. Dataset aliases and default paths are defined in `utils/dataset_config.py`.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgements

This work builds on the following datasets and repositories:

- **AdvGLUE:** https://huggingface.co/datasets/AI-Secure/adv_glue
- **ProverQA:** https://huggingface.co/datasets/opendatalab/ProverQA
- **PROBLEMATHIC:** https://huggingface.co/datasets/him1411/problemathic
