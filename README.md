<div align="center">

# When Correct Demonstrations Hurt: Rethinking the Role of Exemplars in In-Context Learning

[![arXiv](https://img.shields.io/badge/arXiv-ff0000.svg?style=for-the-badge)](https://arxiv.org/abs/2605.26350)  [![Github](https://img.shields.io/badge/Github-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Chenghao-Qiu/Task-Preserving-ICL)
</div>

This repository contains the official implementation of **"When Correct Demonstrations Hurt: Rethinking the Role of Exemplars in In-Context Learning"**

## Table of Contents

- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Main Arguments](#main-arguments)
- [Example Usage](#example-usage)
- [Supported Tasks](#supported-tasks)
- [Citation](#citation)

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

## Citation

If you have any question regarding our paper or code, please feel free to start an issue or email Chenghao Qiu (chenghaoqiu@tamu.edu).

If you use this work in your research, please kindly cite our paper:

**When Correct Demonstrations Hurt: Rethinking the Role of Exemplars in In-Context Learning**

```
@article{qiu2026correct,
  title={When Correct Demonstrations Hurt: Rethinking the Role of Exemplars in In-Context Learning},
  author={Qiu, Chenghao and Peng, Chunli and Yang, Yufeng and Huang, Kuan-Hao and Zhou, Yi},
  journal={arXiv preprint arXiv:2605.26350},
  year={2026}
}
```
