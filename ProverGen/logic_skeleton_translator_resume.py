import argparse
import json
import logging
import os
import pickle
import random
import time
from types import MethodType, SimpleNamespace

import numpy as np
from tqdm.auto import tqdm

from utils.logic_translator import generator as generator_module
from utils.logic_translator import noise as noise_module
from utils.logic_translator import translator as translator_module
from utils.logic_translator.generator import ProblemGenerator
from utils.logic_translator.noise import NoiseTranslator
from utils.logic_translator.translator import Translator


class _NullProgressBar:
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def update(self, _n=1):
        return None


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _resolve_project_path(script_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(script_dir, path)


def _build_stage_args(args):
    stage_args = SimpleNamespace(**vars(args).copy())
    stage_args.start = 0
    stage_args.end = 1
    return stage_args


def _disable_inner_progress_bars():
    translator_module.tqdm = _NullProgressBar
    noise_module.tqdm = _NullProgressBar
    generator_module.tqdm = _NullProgressBar


class ErrorThresholdExceeded(RuntimeError):
    def __init__(self, component_name: str, error_count: int):
        super().__init__(
            f"Cumulative request errors reached {error_count} in {component_name}; restarting pipeline."
        )
        self.component_name = component_name
        self.error_count = error_count


def _atomic_write_json(path: str, data):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _parse_response_payload(answer_str: str):
    cleaned_answer = answer_str.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned_answer)
    except json.JSONDecodeError:
        return eval(cleaned_answer)


def _make_guarded_send_request(component_name: str, logger, error_state, error_threshold: int):
    def _guarded_send_request(self, message):
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message,
                    temperature=0.7,
                )
                answer_str = completion.choices[0].message.content
                return _parse_response_payload(answer_str)
            except Exception as exc:
                self.err_cnt += 1
                error_state["count"] += 1
                logger.warning(
                    "%s request failed. component_errors=%d, cumulative_errors=%d/%d. Retrying in 2 seconds. Error: %s",
                    component_name,
                    self.err_cnt,
                    error_state["count"],
                    error_threshold,
                    exc,
                )
                if error_state["count"] >= error_threshold:
                    raise ErrorThresholdExceeded(component_name, error_state["count"]) from exc
                time.sleep(2)

    return _guarded_send_request


def _build_stage_components(stage_args, logger, error_threshold: int):
    error_state = {"count": 0}

    translator = Translator(stage_args)
    noise_translator = NoiseTranslator(stage_args, translated_data=[])
    problem_generator = ProblemGenerator(stage_args, translated_data=[])

    translator._Translator__send_request = MethodType(
        _make_guarded_send_request("Translator", logger, error_state, error_threshold),
        translator,
    )
    noise_translator._NoiseTranslator__send_request = MethodType(
        _make_guarded_send_request("NoiseTranslator", logger, error_state, error_threshold),
        noise_translator,
    )
    problem_generator._ProblemGenerator__send_request = MethodType(
        _make_guarded_send_request("ProblemGenerator", logger, error_state, error_threshold),
        problem_generator,
    )

    return translator, noise_translator, problem_generator


def _process_single_problem(problem, problem_idx: int, translator, noise_translator, problem_generator, base_seed: int):
    seed_everything(base_seed + problem_idx)

    translated_problem = translator.translate_rules_and_facts(data=[problem])[0]
    noise_translator.data = [translated_problem]
    translated_problem = noise_translator.create_distracting_rules()[0]
    problem_generator.data = [translated_problem]
    final_problem = problem_generator.create_problems()[0]
    return final_problem


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    _disable_inner_progress_bars()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(
                    script_dir,
                    f"logic_skeleton_translator_resume_{time.strftime('%Y%m%d_%H%M%S')}.log",
                )
            ),
        ],
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument("--num", type=int, default=300)
    parser.add_argument("--mode", type=str, default="hard")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=300)
    parser.add_argument("--data_dir", type=str, default="outputs/logic_data")
    parser.add_argument("--output_dir", type=str, default="outputs/translated_data")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct")

    parser.add_argument("--base_url", type=str, default="EMPTY")
    parser.add_argument("--api_key", type=str, default="EMPTY")

    parser.add_argument("--predicate_path", type=str, default="data/wordnet_predicates.json")
    parser.add_argument("--example_path", type=str, default="data/translation_examples.json")
    parser.add_argument("--name_path", type=str, default="data/names")
    parser.add_argument("--seed", type=int, default=727)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--error_threshold",
        type=int,
        default=20,
        help="Restart the translator pipeline after this many cumulative request errors.",
    )

    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help="Directory for per-sample checkpoint files. Defaults to <output_dir>/<mode>-<num>-<start>_<end>_resume.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate samples even if checkpoint files already exist.",
    )

    args = parser.parse_args()

    args.data_dir = _resolve_project_path(script_dir, args.data_dir)
    args.output_dir = _resolve_project_path(script_dir, args.output_dir)
    args.predicate_path = _resolve_project_path(script_dir, args.predicate_path)
    args.example_path = _resolve_project_path(script_dir, args.example_path)
    args.name_path = _resolve_project_path(script_dir, args.name_path)

    if args.resume_dir is None:
        args.resume_dir = os.path.join(
            args.output_dir,
            f"{args.mode}-{args.num}-{args.start}_{args.end}_resume",
        )
    else:
        args.resume_dir = _resolve_project_path(script_dir, args.resume_dir)

    logger.info("Starting resumable logic skeleton translation with args: %s", vars(args))
    seed_everything(args.seed)
    logger.info("Set base random seed to %d", args.seed)

    if args.end <= args.start:
        raise ValueError("--end must be greater than --start.")
    if args.error_threshold <= 0:
        raise ValueError("--error_threshold must be a positive integer.")

    start_time = time.time()
    input_path = os.path.join(args.data_dir, f"{args.mode}-{args.num}.pickle")
    output_path = os.path.join(args.output_dir, f"{args.mode}-{args.num}-{args.start}_{args.end}.json")

    logger.info("Loading dataset from %s", input_path)
    with open(input_path, "rb") as f:
        logic_data = pickle.load(f)

    if args.end > len(logic_data):
        raise ValueError(f"--end={args.end} exceeds dataset size {len(logic_data)}.")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.resume_dir, exist_ok=True)

    stage_args = _build_stage_args(args)
    translator, noise_translator, problem_generator = _build_stage_components(
        stage_args=stage_args,
        logger=logger,
        error_threshold=args.error_threshold,
    )

    target_indices = list(range(args.start, args.end))
    logger.info("Target sample range: [%d, %d), total %d", args.start, args.end, len(target_indices))

    completed_count = 0
    with tqdm(total=len(target_indices), desc="Translating problems", unit="sample") as progress_bar:
        for problem_idx in target_indices:
            shard_path = os.path.join(args.resume_dir, f"{problem_idx}.json")

            if os.path.exists(shard_path) and not args.overwrite:
                completed_count += 1
                logger.info("Skipping sample %d: checkpoint exists.", problem_idx)
                progress_bar.update(1)
                continue

            attempt = 0
            while True:
                attempt += 1
                logger.info(
                    "Processing sample %d/%d (dataset index %d, attempt %d)",
                    completed_count + 1,
                    len(target_indices),
                    problem_idx,
                    attempt,
                )
                try:
                    final_problem = _process_single_problem(
                        problem=logic_data[problem_idx],
                        problem_idx=problem_idx,
                        translator=translator,
                        noise_translator=noise_translator,
                        problem_generator=problem_generator,
                        base_seed=args.seed,
                    )
                    _atomic_write_json(shard_path, final_problem)
                    completed_count += 1
                    logger.info("Saved checkpoint for sample %d to %s", problem_idx, shard_path)
                    progress_bar.update(1)
                    break
                except ErrorThresholdExceeded as exc:
                    logger.warning(
                        "Reached cumulative error threshold (%d) while processing sample %d in %s. Reinitializing pipeline and retrying.",
                        exc.error_count,
                        problem_idx,
                        exc.component_name,
                    )
                    translator, noise_translator, problem_generator = _build_stage_components(
                        stage_args=stage_args,
                        logger=logger,
                        error_threshold=args.error_threshold,
                    )

    logger.info("Merging checkpoint shards into %s", output_path)
    merged_results = []
    missing_indices = []
    for problem_idx in target_indices:
        shard_path = os.path.join(args.resume_dir, f"{problem_idx}.json")
        if not os.path.exists(shard_path):
            missing_indices.append(problem_idx)
            continue
        merged_results.append(_load_json(shard_path))

    if missing_indices:
        raise RuntimeError(f"Missing checkpoint shards for indices: {missing_indices[:10]}")

    _atomic_write_json(output_path, merged_results)

    duration = time.time() - start_time
    logger.info("Saved translated problems to %s", output_path)
    logger.info("Total time: %.2f seconds", duration)
    logger.info("Average time per target problem: %.2f seconds", duration / len(target_indices))
    logger.info("Resumable logic skeleton translation completed successfully.")
