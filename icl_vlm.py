import os
import json
import random
import argparse
import warnings
import re
from typing import Any, List, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except Exception:
    AutoModelForImageTextToText = None
    AutoProcessor = None
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams
except Exception:
    LLM = None
    SamplingParams = None

from utils.dataprocess import TestDataset
from utils.attention_map import export_attention_map_for_run
from utils.dataset_config import (
    DATASET_CHOICES,
    DATASET_CONFIGS,
    is_alias_dataset_name,
    resolve_dataset_name,
)
from utils.exemplar_sampling import load_exemplar_candidates, sample_mixed_exemplars
from utils.prompt_format import format_prompt_exemplar

DEBUG_PROMPT = True
DEBUG_GENERATIONS = True
DEBUG_GENERATION_SAMPLES = 2
QA_COT_SYSTEM_PROMPT = (
    "Given a problem statement as contexts, the task is to answer a logical reasoning "
    "question. Your answer should be in JSON format with keys: reasoning, answer."
)
MATH_COT_SYSTEM_PROMPT = (
    "Solve the math word problem. "
    "Your answer should be in JSON format with keys: reasoning, answer. "
    "The answer value should be numeric."
)


class ICLTester:
    """In-Context Learning Accuracy Tester for Causal LLMs"""

    def __init__(
        self,
        model_name: str,
        labels: Tuple[str, ...] = None,
        task_type: str = 'classification',
        device: str = 'cuda',
        backend: str = 'transformers',
        include_instruction: bool = False,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        enforce_eager: bool = False,
        max_new_tokens: int = 1024,
        enable_attention_export: bool = False,
        attention_map_mode: str = 'token',
        attention_map_publication: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.backend = backend
        self.task_type = task_type
        self.is_qa_task = task_type == 'qa'
        self.is_math_task = task_type == 'math'
        self.is_generative_task = self.is_qa_task or self.is_math_task
        self.include_instruction = include_instruction
        self.max_model_len = max_model_len
        self.max_new_tokens = max_new_tokens
        self.enable_attention_export = enable_attention_export
        self.attention_map_mode = attention_map_mode
        self.attention_map_publication = attention_map_publication
        self.is_vlm_text_only_model = self._is_text_only_vlm_model(model_name)
        self.system_prompt = MATH_COT_SYSTEM_PROMPT if self.is_math_task else QA_COT_SYSTEM_PROMPT
        self.qa_system_prompt = self.system_prompt
        if not self.is_math_task and (not labels or len(labels) < 2):
            raise ValueError("labels must contain at least 2 classes")
        self.labels = tuple(labels or ())
        if self.task_type == 'qa':
            self.instruction = self.system_prompt
            self.max_model_len = max_model_len = 8192  # QA tasks often require more context for reasoning, so we set a higher default max_model_len.
            self.max_new_tokens = max_new_tokens = 2048  # Allow more tokens for the model to generate the answer and reasoning.
            print("Note: For QA tasks, max_model_len is set to 8192 and max_new_tokens is set to 2048 by default. Adjust these if your model or task requires different settings.")
        elif self.task_type == 'math':
            self.instruction = self.system_prompt
            self.max_model_len = max_model_len = 4096
            self.max_new_tokens = max_new_tokens = 256
        elif self.task_type == 'nli':
            self.instruction = (
                f"Read the paired inputs carefully and decide the correct relation. "
                f"Your answer must be exactly one of: {', '.join(self.labels)}. "
                f"Output only the label and nothing else."
            )
        else:
            self.instruction = (
                f"Your answer must be exactly one of the following labels: {', '.join(self.labels)}. "
                f"Output only the label and nothing else."
            )
        
        print(f"Loading model: {model_name}")
        print(f"Inference backend: {backend}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.processor = None
        self.chat_template_renderer = self.tokenizer

        if self.is_vlm_text_only_model:
            if AutoProcessor is not None:
                try:
                    self.processor = AutoProcessor.from_pretrained(model_name)
                    processor_tokenizer = getattr(self.processor, "tokenizer", None)
                    if processor_tokenizer is not None:
                        self.tokenizer = processor_tokenizer
                    self.chat_template_renderer = self.processor
                    print("Using VLM processor chat template in text-only mode.")
                except Exception as exc:
                    warnings.warn(
                        f"Failed to load AutoProcessor for VLM text-only mode; "
                        f"falling back to tokenizer chat template. Error: {exc}"
                    )
            else:
                warnings.warn(
                    "AutoProcessor is unavailable in this transformers installation; "
                    "falling back to tokenizer chat template for VLM text-only mode."
                )
        
        # [CRITICAL] Set padding side to 'left' for batch inference with decoder-only models.
        # If set to 'right', the last token position would be a padding token, not the input.
        self.tokenizer.padding_side = 'left'
        
        # Handle padding token if missing
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"Set tokenizer pad_token to eos_token: {self.tokenizer.eos_token}")
            else:
                raise ValueError("Tokenizer has no pad_token or eos_token to set as pad_token.")

        if backend == 'transformers':
            # Use CausalLM for generation/next-token prediction tasks.
            # device=auto lets HF/Accelerate shard a large model across visible GPUs.
            model_kwargs = {
                "dtype": torch.bfloat16,
                "device_map": 'auto' if device == 'auto' else None,
            }
            model_cls = (
                AutoModelForImageTextToText
                if self.is_vlm_text_only_model and AutoModelForImageTextToText is not None
                else AutoModelForCausalLM
            )
            try:
                self.model = model_cls.from_pretrained(model_name, **model_kwargs)
            except Exception as exc:
                if model_cls is AutoModelForCausalLM:
                    raise
                warnings.warn(
                    f"Failed to load VLM model with AutoModelForImageTextToText; "
                    f"falling back to AutoModelForCausalLM. Error: {exc}"
                )
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

            self.model.config.pad_token_id = self.tokenizer.pad_token_id

            if device != 'auto':
                self.model.to(device)

            if self.enable_attention_export and hasattr(self.model, "set_attn_implementation"):
                self.model.set_attn_implementation("eager")
                print("Set transformers attention implementation to eager for attention-map export.")

            # In model-parallel mode (device_map=auto), inputs should be sent to the
            # first shard device rather than a single global device string.
            self.input_device = next(self.model.parameters()).device
            print(f"Model input device: {self.input_device}")

            self.model.eval()
        elif backend == 'vllm':
            if LLM is None or SamplingParams is None:
                raise ImportError(
                    "vLLM is not installed or failed to import. Please install vllm first."
                )

            self.llm = LLM(
                model=model_name,
                tokenizer=model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                enforce_eager=enforce_eager,
            )
            print(f"vLLM tensor_parallel_size: {tensor_parallel_size}")
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        self.exemplars = ''
        self.exemplar_attention_metadata: List[Dict[str, str]] = []
        self._debug_prompt_printed = False
        self._prompt_prefix_checked = False
        self._debug_generation_count = 0

        # Pre-compute first-token IDs for candidate labels.
        # Prompt ends with a space, so model predicts label token directly.
        self.label_token_ids: List[int] = []
        if not self.is_math_task:
            for label in self.labels:
                token_ids = self.tokenizer.encode(label, add_special_tokens=False)
                if not token_ids:
                    raise ValueError(f"Label '{label}' cannot be tokenized into non-empty token ids")
                self.label_token_ids.append(token_ids[0])

            print("Monitoring labels:")
            for label, token_id in zip(self.labels, self.label_token_ids):
                real_token = self.tokenizer.decode([token_id])
                print(f"  - {label}: [{token_id}] -> '{real_token}'")

    @staticmethod
    def _sanitize_path_component(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "run"

    def export_attention_map_for_run(
        self,
        test_csv: str,
        output_dir: str,
        run_idx: int,
        test_example_idx: int = 0,
    ) -> None:
        export_attention_map_for_run(
            backend=self.backend,
            is_qa_task=self.is_qa_task,
            test_csv=test_csv,
            output_dir=output_dir,
            run_idx=run_idx,
            test_example_idx=test_example_idx,
            tokenizer=self.tokenizer,
            model=self.model,
            input_device=self.input_device,
            max_model_len=self.max_model_len,
            label_token_ids=self.label_token_ids,
            labels=self.labels,
            create_prompt=self.create_prompt,
            create_attention_focus_content=self.create_attention_focus_content,
            attention_map_mode=self.attention_map_mode,
            exemplar_metadata=self.exemplar_attention_metadata,
            publication_mode=self.attention_map_publication,
        )

    def _extract_label_from_generation(self, generated_text: str) -> str:
        if self.is_math_task:
            return self._extract_math_response(generated_text)

        if self.is_qa_task:
            parsed = self._extract_qa_response(generated_text)
            if parsed is not None:
                answer = parsed.get('answer')
                if answer is not None:
                    normalized = self._normalize_prediction(str(answer).strip())
                    if normalized is not None:
                        return normalized

        matches = re.findall(r"####\s*([^\s.,;:!?]+)", generated_text)
        if matches:
            candidate = matches[-1].strip()
            candidate = re.search(r"[A-Za-z]", candidate).group(0) # Llama-3.1-8B-Instruct sometimes generates "#### A)" instead of "#### A", so we extract just the letter. You can adjust this regex if your labels are different.
            normalized = self._normalize_prediction(candidate)
            if normalized is not None:
                return normalized
        return None

    @staticmethod
    def _parse_numeric_value(value: Any) -> float | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        text = text.replace(",", "")
        if re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", text):
            numerator, denominator = re.split(r"\s*/\s*", text)
            denominator_value = float(denominator)
            if denominator_value == 0:
                return None
            return float(numerator) / denominator_value
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    @staticmethod
    def _format_numeric_prediction(value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.12g}"

    def _extract_math_response(self, generated_text: str) -> str | None:
        parsed = self._extract_qa_response(generated_text)
        if parsed is not None and parsed.get('answer') is not None:
            numeric = self._parse_numeric_value(parsed.get('answer'))
            if numeric is not None:
                return self._format_numeric_prediction(numeric)

        stripped = generated_text.strip()
        patterns = [
            r"####\s*([-+]?(?:[\d,]+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:\s*/\s*[-+]?\d+)?)",
            r"(?:final\s+answer|answer)\s*(?:is|=|:)?\s*([-+]?(?:[\d,]+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:\s*/\s*[-+]?\d+)?)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, stripped, flags=re.IGNORECASE)
            if matches:
                numeric = self._parse_numeric_value(matches[-1])
                if numeric is not None:
                    return self._format_numeric_prediction(numeric)

        matches = re.findall(
            r"[-+]?(?:[\d,]+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:\s*/\s*[-+]?\d+)?",
            stripped,
        )
        if matches:
            numeric = self._parse_numeric_value(matches[-1])
            if numeric is not None:
                return self._format_numeric_prediction(numeric)
        return None

    def _extract_qa_response(self, generated_text: str) -> Dict[str, Any] | None:
        candidates: List[str] = []
        stripped = generated_text.strip()
        if not stripped:
            return None

        fenced_blocks = re.findall(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            stripped,
            flags=re.DOTALL | re.IGNORECASE,
        )
        candidates.extend(fenced_blocks)
        candidates.extend(re.findall(r"\{.*?\}", stripped, flags=re.DOTALL))
        candidates.append(stripped)

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                answer = parsed.get('answer')
                if answer is not None:
                    answer = str(answer).strip()
                    if answer:
                        for label in self.labels:
                            if answer.lower() == label.lower():
                                parsed['answer'] = label
                                break
                        else:
                            first_char = answer[0]
                            for label in self.labels:
                                if first_char.lower() == label.lower():
                                    parsed['answer'] = label
                                    break
                return parsed

        answer_match = re.search(
            r'"answer"\s*:\s*"([^"]+)"',
            stripped,
            flags=re.IGNORECASE,
        )
        if answer_match:
            answer = answer_match.group(1).strip()
            for label in self.labels:
                if answer.lower() == label.lower():
                    return {'answer': label}
            if answer:
                first_char = answer[0]
                for label in self.labels:
                    if first_char.lower() == label.lower():
                        return {'answer': label}
            return {'answer': answer}
        return None

    def _debug_generation_output(self, generated_text: str) -> None:
        if not DEBUG_GENERATIONS:
            return
        if self._debug_generation_count >= DEBUG_GENERATION_SAMPLES:
            return

        print("\n" + "=" * 60)
        print(f"DEBUG GENERATION #{self._debug_generation_count + 1}")
        print("=" * 60)
        print("RAW OUTPUT:")
        print(generated_text if generated_text else "<EMPTY>")
        print("=" * 60)

        self._debug_generation_count += 1

    def _normalize_prediction(self, prediction: str) -> str | None:
        if self.is_math_task:
            numeric = self._parse_numeric_value(prediction)
            if numeric is None:
                return None
            return self._format_numeric_prediction(numeric)

        candidate = str(prediction).strip()
        for label in self.labels:
            if candidate.lower() == label.lower():
                return label
        return None

    def _is_prediction_correct(self, predicted_label: str | None, true_label: str) -> bool:
        if self.is_math_task:
            pred_value = self._parse_numeric_value(predicted_label)
            true_value = self._parse_numeric_value(true_label)
            if pred_value is None or true_value is None:
                return False
            return abs(pred_value - true_value) <= max(1e-6, 1e-6 * abs(true_value))

        normalized_pred = self._normalize_prediction(predicted_label)
        normalized_true = self._normalize_prediction(true_label)
        if normalized_pred is None or normalized_true is None:
            return False

        if self.task_type == 'qa':
            return normalized_pred == normalized_true
        if self.task_type == 'nli':
            return normalized_pred == normalized_true
        return normalized_pred == normalized_true
    
    def set_exemplars_from_text(self, exemplar_text: str, exemplar_metadata: List[Dict[str, str]] = None):
        """Set exemplars directly from text string"""
        self.exemplars = exemplar_text
        self.exemplar_attention_metadata = list(exemplar_metadata or [])
        self._prompt_prefix_checked = False

    @staticmethod
    def _is_pair_text(text: str) -> bool:
        normalized = text.strip().lower()
        return (
            normalized.startswith('premise:') and 'hypothesis:' in normalized
        ) or (
            normalized.startswith('question1:') and 'question2:' in normalized
        )

    @staticmethod
    def _should_use_chat_template(model_name: str) -> bool:
        normalized_name = model_name.lower()
        if "gemma-4" in normalized_name:
            return "it" in normalized_name
        if "qwen3.5" in normalized_name:
            return not "base" in normalized_name
        return False

    @staticmethod
    def _is_text_only_vlm_model(model_name: str) -> bool:
        normalized_name = model_name.lower()
        return "gemma-4" in normalized_name or "qwen3.5" in normalized_name

    def _should_disable_thinking(self) -> bool:
        normalized_name = self.model_name.lower()
        return "qwen3.5" in normalized_name

    def _format_chat_content(self, text: str) -> Any:
        if self.is_vlm_text_only_model:
            return [{"type": "text", "text": text}]
        return text

    def _apply_chat_template(self, messages: List[Dict[str, Any]]) -> str:
        renderer = self.chat_template_renderer
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self._should_disable_thinking():
            kwargs["enable_thinking"] = False

        try:
            return renderer.apply_chat_template(messages, **kwargs)
        except TypeError:
            if "enable_thinking" not in kwargs:
                raise
            warnings.warn(
                "This chat template renderer does not accept enable_thinking=False; "
                "Qwen3.5 NLI thinking could not be disabled through the template API."
            )
            kwargs.pop("enable_thinking")
            return renderer.apply_chat_template(messages, **kwargs)

    def _get_chat_template(self) -> str | None:
        renderer_template = getattr(self.chat_template_renderer, "chat_template", None)
        if renderer_template:
            return renderer_template
        renderer_tokenizer = getattr(self.chat_template_renderer, "tokenizer", None)
        return getattr(renderer_tokenizer, "chat_template", None)

    def _create_user_content(self, text: str) -> str:
        """Create the raw user content before optional chat templating."""
        if self.is_generative_task:
            prompt_parts = []
            if self.include_instruction:
                prompt_parts.append(self.instruction)
            if self.exemplars:
                prompt_parts.append(self.exemplars.strip())
            prompt_parts.append(text.strip())
            return "\n\n".join(part for part in prompt_parts if part)

        if self._is_pair_text(text):
            query_text = text
        else:
            query_text = f"sentence: {text}"
        prompt_parts = []
        if self.include_instruction:
            prompt_parts.append(self.instruction)
        if self.exemplars:
            prompt_parts.append(self.exemplars)
        # Add trailing space after 'is' so the model predicts the label token directly (not space token)
        prompt_parts.append(f"{query_text}\nThe answer is ")
        return "\n\n".join(prompt_parts)

    def create_attention_focus_content(self, text: str) -> str:
        """Create the View E span: exemplars plus the actual query and answer stub."""
        if self.is_generative_task:
            return self._create_user_content(text)

        if self._is_pair_text(text):
            query_text = text
        else:
            query_text = f"sentence: {text}"

        prompt_parts = []
        if self.exemplars:
            prompt_parts.append(self.exemplars)
        prompt_parts.append(f"{query_text}\nThe answer is ")
        return "\n\n".join(part for part in prompt_parts if part)

    def create_prompt(self, text: str) -> str:
        """Create the final model input, wrapped as a user chat message when supported."""
        user_content = self._create_user_content(text)
        if self.is_generative_task and self._should_use_chat_template(self.model_name):
            messages = [
                {"role": "system", "content": self._format_chat_content(self.system_prompt)},
                {"role": "user", "content": self._format_chat_content(user_content)},
            ]
            chat_template = self._get_chat_template()
        elif self._should_use_chat_template(self.model_name):
            messages = [
                {"role": "user", "content": self._format_chat_content(user_content)},
            ]
            chat_template = self._get_chat_template()
        else:
            messages = None
            chat_template = None

        if messages is not None and hasattr(self.chat_template_renderer, "apply_chat_template") and chat_template:
            return self._apply_chat_template(messages)
        if self.is_generative_task:
            return (
                f"System: {self.system_prompt}\n\n"
                f"User: {user_content}\n\n"
                f"Assistant:"
            )
        return user_content

    def _validate_prompt_prefix_length(self) -> None:
        """Fail fast if the fixed prompt prefix already consumes too much context."""
        if self._prompt_prefix_checked:
            return

        prompt_prefix = self.create_prompt(".")
        prefix_token_ids = self.tokenizer(
            prompt_prefix,
            add_special_tokens=True,
            truncation=False,
        )['input_ids']
        prefix_len = len(prefix_token_ids)
        max_prefix_len = int(self.max_model_len * 0.9)

        if prefix_len > max_prefix_len:
            warning_msg = (
                f"Prompt prefix uses {prefix_len} tokens, exceeding 90% of "
                f"--max-model-len={self.max_model_len} (limit: {max_prefix_len}). "
                f"Refusing to continue."
            )
            warnings.warn(warning_msg)
            raise ValueError(warning_msg)
        # else:
        #     print(f"Prompt prefix uses {prefix_len} tokens, within safe limit.")

        self._prompt_prefix_checked = True
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Batch predict labels by comparing candidate-label next-token probabilities"""
        prompts = [self.create_prompt(text) for text in texts]
        self._validate_prompt_prefix_length()

        # Debug: print the first full ICL prompt once so you can inspect the exact model input.
        # Comment out this block after debugging if you no longer need it.
        if DEBUG_PROMPT and not self._debug_prompt_printed and prompts:
            print("\n" + "=" * 60)
            print("DEBUG ICL PROMPT")
            print("=" * 60)
            print(prompts[0])
            print("=" * 60)
            self._debug_prompt_printed = True

        if self.is_generative_task:
            return self._predict_batch_generative(prompts)

        if self.backend == 'transformers':
            inputs = self.tokenizer(
                prompts,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_model_len,
                padding=True,
            )

            input_ids = inputs['input_ids'].to(self.input_device)
            attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids)).to(self.input_device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]

                results = []
                for i in range(len(texts)):
                    candidate_logits = next_token_logits[i, self.label_token_ids]
                    probs = torch.softmax(candidate_logits, dim=0)
                    best_idx = int(torch.argmax(probs).item())
                    results.append((self.labels[best_idx], float(probs[best_idx].item())))

            return results

        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            allowed_token_ids=sorted(set(self.label_token_ids)),
            logprobs=max(len(self.label_token_ids), len(self.labels)),
        )
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        results = []
        for out in outputs:
            if not out.outputs or not out.outputs[0].token_ids:
                results.append((self.labels[0], 0.0))
                continue

            pred_token_id = out.outputs[0].token_ids[0]
            label_idx = 0
            for idx, tid in enumerate(self.label_token_ids):
                if tid == pred_token_id:
                    label_idx = idx
                    break

            confidence = 1.0
            step_logprobs = out.outputs[0].logprobs
            if step_logprobs and len(step_logprobs) > 0 and step_logprobs[0] is not None:
                lp_map = step_logprobs[0]
                candidate_lps = []
                for tid in self.label_token_ids:
                    info = lp_map.get(tid)
                    candidate_lps.append(float(info.logprob) if info is not None else -1e30)
                lp_tensor = torch.tensor(candidate_lps, dtype=torch.float32)
                probs = torch.softmax(lp_tensor, dim=0)
                confidence = float(probs[label_idx].item())

            results.append((self.labels[label_idx], confidence))

        return results

    def _predict_batch_generative(self, prompts: List[str]) -> List[Tuple[str, float]]:
        if self.backend == 'transformers':
            inputs = self.tokenizer(
                prompts,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_model_len,
                padding=True,
            )

            input_ids = inputs['input_ids'].to(self.input_device)
            attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids)).to(self.input_device)

            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            prompt_length = input_ids.shape[1]
            results = []
            for i in range(len(prompts)):
                new_tokens = generated[i, prompt_length:]
                generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                predicted_label = self._extract_label_from_generation(generated_text)
                self._debug_generation_output(generated_text)
                confidence = 1.0 if predicted_label is not None else 0.0
                results.append((predicted_label, confidence))
            return results

        sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=0.0,
        )
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        results = []
        for prompt, out in zip(prompts, outputs):
            generated_text = ""
            if out.outputs and out.outputs[0].text:
                generated_text = out.outputs[0].text.strip()
            predicted_label = self._extract_label_from_generation(generated_text)
            self._debug_generation_output(generated_text)
            confidence = 1.0 if predicted_label is not None else 0.0
            results.append((predicted_label, confidence))

        return results
    
    def test_accuracy(self, test_csv: str, batch_size: int = 8, max_eval_samples: int = None) -> Dict:
        """Test accuracy on test dataset using batch processing"""
        print(f"\nLoading test data from: {test_csv}")
        test_dataset = TestDataset(test_csv, max_samples=max_eval_samples)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        correct = 0
        total = 0
        results = {
            'total': len(test_dataset),
            'correct': 0,
            'accuracy': 0.0,
            'predictions': [],
            'batch_size': batch_size
        }
        
        print(f"Testing on {len(test_dataset)} samples with batch size {batch_size}...")
        
        for batch in tqdm(test_dataloader):
            texts = batch['text']
            true_labels = batch['label']
            
            # Batch prediction
            predictions = self.predict_batch(texts)
            
            # Process results
            for text, true_label, (predicted_label, confidence) in zip(texts, true_labels, predictions):
                is_correct = self._is_prediction_correct(predicted_label, true_label)
                if is_correct:
                    correct += 1
                total += 1
                
                results['predictions'].append({
                    'text': text,
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'confidence': float(confidence),
                    'correct': is_correct
                })
        
        accuracy = correct / total if total > 0 else 0.0
        results['correct'] = correct
        results['accuracy'] = accuracy
        
        return results


def main():
    default_device = 'auto' if torch.cuda.device_count() > 1 else ('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='In-Context Learning Accuracy Tester')
    parser.add_argument('--dataset', type=str, default='sst2',
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='HuggingFace model name')
    parser.add_argument('--backend', type=str, default='transformers', choices=['transformers', 'vllm'],
                        help='Inference backend to use')
    parser.add_argument('--test-data', type=str, default='',
                        help='Path to test data CSV file')
    parser.add_argument('--exemplar-pool', type=str, default='',
                        help='Path to exemplar candidate CSV; expected columns: text, adv_text, label, label_id')
    parser.add_argument('--num-exemplars', type=int, default=16,
                        help='Number of exemplars to sample per run')
    parser.add_argument('--num-adv-exemplars', type=int, default=0,
                        help='Number of adversarial exemplars to mix into each run')
    parser.add_argument('--adv-placement', type=str, default='random', choices=['random', 'head', 'medium', 'tail', 'custom'],
                        help='Placement of adversarial exemplars in the final prompt: random, head (front), medium (center), tail (back), or custom (first two in front, remaining at the back)')
    parser.add_argument('--num-runs', type=int, default=100,
                        help='Number of random runs to average')
    parser.add_argument('--sampling', type=str, default='balanced', choices=['balanced', 'random'],
                        help='Sampling strategy: balanced (equal per label) or random (fully random)')
    parser.add_argument('--output', type=str, default='',
                        help='Path to save results')
    parser.add_argument('--device', type=str, default=default_device,
                        help='Device to use (cuda, cpu, or auto for multi-GPU sharding)')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                        help='vLLM tensor parallel size (number of GPUs)')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                        help='vLLM GPU memory utilization ratio')
    parser.add_argument('--max-model-len', type=int, default=4096,
                        help='Maximum prompt/model length')
    parser.add_argument('--max-new-tokens', type=int, default=1024,
                        help='Maximum number of new tokens to generate for generative tasks')
    parser.add_argument('--enforce-eager', action='store_true',
                        help='Disable vLLM torch.compile and CUDA graphs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for testing')
    parser.add_argument('--max-eval-samples', type=int, default=1000,
                        help='Maximum number of evaluation samples to use; if the eval set is larger, only the first N rows are used')
    parser.add_argument('--include-instruction', action='store_true',
                        help='Include the label instruction in the prompt prefix')
    parser.add_argument('--attention-map', action='store_true',
                        help='Export attention heatmaps for the first 5 test examples (transformers backend only)')
    parser.add_argument('--attention-map-dir', type=str, default='',
                        help='Directory to save attention heatmaps; defaults to a folder derived from --output')
    parser.add_argument('--attention-map-mode', type=str, default='token', choices=['token', 'word', 'unit'],
                        help='Granularity for attention heatmaps: token exports View A/B over subword tokens, word exports View A/B over merged words, unit exports View C with one unit per in-context exemplar')
    parser.add_argument('--attention-map-publication', action='store_true',
                        help='Also export raw attention map data (.npz plus .json metadata) for publication plotting')
    
    args = parser.parse_args()

    resolved_dataset = resolve_dataset_name(args.dataset)
    if resolved_dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Unsupported dataset: {args.dataset}. "
            f"Known base datasets/aliases: {', '.join(DATASET_CHOICES)}"
        )
    cfg = DATASET_CONFIGS[resolved_dataset]
    if is_alias_dataset_name(args.dataset):
        args.test_data = args.test_data or f"data/{args.dataset}/validation.csv"
        args.exemplar_pool = args.exemplar_pool or f"exemplars/{args.dataset}.csv"
    else:
        args.test_data = args.test_data or cfg['test_data']
        args.exemplar_pool = args.exemplar_pool or cfg['exemplar_pool']
    labels = cfg['labels']
    task_type = cfg.get('task_type', 'classification')
    
    # Create results directory
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    if args.attention_map and args.backend != 'transformers':
        raise ValueError("--attention-map requires --backend transformers")
    
    print(f"\n{'=' * 60}")
    print(f"RUNNING {args.num_runs} exemplar trials")
    print(f"Sampling {args.num_exemplars} exemplars per run from {args.exemplar_pool}")
    print(f"Mixing {args.num_adv_exemplars} adversarial exemplars per run")
    print(f"Adversarial exemplar placement: {args.adv_placement}")
    print(f"Max Eval Samples: {args.max_eval_samples}")
    print(f"Include Instruction: {args.include_instruction}")
    print(f"{'=' * 60}")

    exemplar_candidates = load_exemplar_candidates(
        args.exemplar_pool,
        dataset_name=args.dataset,
        labels=labels,
    )
    tester = ICLTester(
        args.model,
        labels=labels,
        task_type=task_type,
        device=args.device,
        backend=args.backend,
        include_instruction=args.include_instruction,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        max_new_tokens=args.max_new_tokens,
        enable_attention_export=args.attention_map,
        attention_map_mode=args.attention_map_mode,
        attention_map_publication=args.attention_map_publication,
    )

    all_accuracies = []
    all_run_results = []
    for run_idx in range(args.num_runs):
        random.seed(run_idx)
        sampled = sample_mixed_exemplars(
            exemplar_candidates,
            num_exemplars=args.num_exemplars,
            num_adv_exemplars=args.num_adv_exemplars,
            sampling=args.sampling,
            adv_placement=args.adv_placement,
            task_type=task_type,
        )
        exemplar_metadata = [
            {
                'text': format_prompt_exemplar(
                    s['text'],
                    s['label'],
                    s.get('reference'),
                    task_type=task_type,
                ),
                'source': s.get('source', 'orig'),
            }
            for s in sampled
        ]
        exemplar_text = '\n\n'.join(item['text'] for item in exemplar_metadata)
        tester.set_exemplars_from_text(exemplar_text, exemplar_metadata=exemplar_metadata)

        if args.attention_map and run_idx < 5:
            if args.attention_map_dir:
                attention_map_dir = args.attention_map_dir
            elif args.output:
                output_root, _ = os.path.splitext(args.output)
                attention_map_dir = f"{output_root}_attention_maps"
            else:
                model_tag = tester._sanitize_path_component(args.model.split("/")[-1])
                dataset_tag = tester._sanitize_path_component(args.dataset)
                attention_map_dir = os.path.join(
                    "results",
                    model_tag,
                    f"{dataset_tag}_attention_maps",
                )
            tester.export_attention_map_for_run(
                args.test_data,
                output_dir=attention_map_dir,
                run_idx=run_idx,
                test_example_idx=0,
            )

        run_results = tester.test_accuracy(
            args.test_data,
            batch_size=args.batch_size,
            max_eval_samples=args.max_eval_samples,
        )
        acc = run_results['accuracy']
        all_accuracies.append(acc)
        all_run_results.append({
            'run_idx': run_idx,
            'accuracy': float(acc),
        })
        print(f"  Run {run_idx + 1}/{args.num_runs}: accuracy = {acc:.4f}")

    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)

    print(f"\n{'=' * 60}")
    print("ICL RESULTS")
    print(f"{'=' * 60}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Exemplar Pool: {args.exemplar_pool}")
    print(f"Test Data: {args.test_data}")
    print(f"Num Exemplars per run: {args.num_exemplars}")
    print(f"Num Adversarial Exemplars per run: {args.num_adv_exemplars}")
    print(f"Adversarial Exemplar Placement: {args.adv_placement}")
    print(f"Max Eval Samples: {args.max_eval_samples}")
    print(f"Include Instruction: {args.include_instruction}")
    print(f"Num Runs: {args.num_runs}")
    print(f"Mean Accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%)")
    print(f"Std Accuracy:  {std_acc:.4f} ({std_acc*100:.2f}%)")
    print(f"Min Accuracy:  {np.min(all_accuracies):.4f}")
    print(f"Max Accuracy:  {np.max(all_accuracies):.4f}")
    print(f"{'=' * 60}")

    results = {
        'exemplar_pool': args.exemplar_pool,
        'num_runs': args.num_runs,
        'num_exemplars': args.num_exemplars,
        'num_adv_exemplars': args.num_adv_exemplars,
        'adv_placement': args.adv_placement,
        'max_eval_samples': args.max_eval_samples,
        'include_instruction': args.include_instruction,
        'attention_map': args.attention_map,
        'attention_map_dir': args.attention_map_dir,
        'attention_map_mode': args.attention_map_mode,
        'attention_map_publication': args.attention_map_publication,
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'min_accuracy': float(np.min(all_accuracies)),
        'max_accuracy': float(np.max(all_accuracies)),
        'all_accuracies': [float(a) for a in all_accuracies],
        'run_summaries': all_run_results,
    }

    if args.output:
        print(f"\nSaving results to {args.output}...")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("✓ Done!")
    else:
        print("\nNo --output provided; results were not written to disk.")


if __name__ == '__main__':
    main()
