import logging
import multiprocessing as mp
import os
import pickle
import random
import time
from argparse import ArgumentParser, Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from types import SimpleNamespace
from typing import List

import numpy as np

from utils.logic_generator import generator as generator_module
from utils.logic_generator.generator import LogicGenerator


class _NullProgressBar:
    def __init__(self, iterable):
        self.iterable = iterable

    def update(self, _n=1):
        return None


def _disable_worker_tqdm():
    generator_module.tqdm = _NullProgressBar


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _resolve_project_path(script_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(script_dir, path)


def _build_worker_args(args_dict: dict, num: int) -> SimpleNamespace:
    worker_args = dict(args_dict)
    worker_args["num"] = num
    return SimpleNamespace(**worker_args)


def _parse_numeric_list(value: str) -> List[float]:
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        raise ValueError(f"Expected list format like [0.4, 0.3, 0.3], got: {value}")

    items = [item.strip() for item in value[1:-1].split(",") if item.strip()]
    if not items:
        raise ValueError("The list cannot be empty.")

    parsed = []
    for item in items:
        try:
            parsed.append(float(Fraction(item)))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"Invalid numeric item '{item}' in list: {value}") from exc

    return parsed


def _generate_chunk(worker_id: int, args_dict: dict, num_samples: int, seed: int) -> List:
    os.environ["PROVER9"] = args_dict["prover9_path"]
    _disable_worker_tqdm()
    seed_everything(seed)

    worker_args = _build_worker_args(args_dict, num_samples)
    generator = LogicGenerator(worker_args)
    problems = generator.generate_logic_skeletons(verbose=False)

    for local_id, problem in enumerate(problems):
        problem.id = local_id

    return problems


def _split_work(total: int, workers: int) -> List[int]:
    base, remainder = divmod(total, workers)
    return [base + (1 if idx < remainder else 0) for idx in range(workers) if base + (1 if idx < remainder else 0) > 0]


def _parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--num", type=int, default=300)
    parser.add_argument("--mode", type=str, default="hard")
    parser.add_argument("--output_dir", type=str, default="outputs/logic_data")

    parser.add_argument("--seed", type=int, default=730)
    parser.add_argument(
        "--goal_value_probs",
        type=str,
        default="[1/3, 1/3, 1/3]",
        help="The proportion of True, False and Uncertain. Use standard list format, such as [0.4, 0.3, 0.3].",
    )
    parser.add_argument("--rule_candidate_path", type=str, default="data/rules.json")
    parser.add_argument(
        "--rule_as_goal_proportion",
        type=str,
        default="[0.75, 0.25]",
        help="The first number is the proportion of logic skeletons with a fact conclusion and the second is for a rule conclusion.",
    )
    parser.add_argument(
        "--fact_num_threshold",
        type=int,
        default=2,
        help="When the size of the fact pool exceeds the threshold, there is a chance that the fact is given directly.",
    )
    parser.add_argument("--fact_num_prob", type=float, default=0.4)

    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--prover9_path",
        type=str,
        default="LADR-2009-11A/bin",
        help="Path to the Prover9 binaries.",
    )

    args = parser.parse_args()
    args.goal_value_probs = _parse_numeric_list(args.goal_value_probs)
    args.rule_as_goal_proportion = _parse_numeric_list(args.rule_as_goal_proportion)
    return args


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args = _parse_args()

    args.rule_candidate_path = _resolve_project_path(script_dir, args.rule_candidate_path)
    args.output_dir = _resolve_project_path(script_dir, args.output_dir)
    args.prover9_path = _resolve_project_path(script_dir, args.prover9_path)
    os.environ["PROVER9"] = args.prover9_path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(
                    script_dir,
                    f"logic_skeleton_generator_multiprocess_{time.strftime('%Y%m%d_%H%M%S')}.log",
                )
            ),
        ],
    )
    logger = logging.getLogger(__name__)

    if args.num <= 0:
        raise ValueError("--num must be a positive integer.")
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be a positive integer.")

    worker_sizes = _split_work(args.num, min(args.num_workers, args.num))
    start_time = time.time()

    logger.info("Starting multiprocessing logic skeleton generation with args: %s", vars(args))
    logger.info("Using %d worker processes with workload split: %s", len(worker_sizes), worker_sizes)

    mp_context = mp.get_context("spawn")
    args_dict = vars(args).copy()
    futures = {}
    chunk_results = [None] * len(worker_sizes)

    with ProcessPoolExecutor(max_workers=len(worker_sizes), mp_context=mp_context) as executor:
        for worker_id, chunk_size in enumerate(worker_sizes):
            worker_seed = args.seed + worker_id
            future = executor.submit(_generate_chunk, worker_id, args_dict, chunk_size, worker_seed)
            futures[future] = (worker_id, chunk_size, worker_seed)

        for future in as_completed(futures):
            worker_id, chunk_size, worker_seed = futures[future]
            problems = future.result()
            chunk_results[worker_id] = problems
            logger.info(
                "Worker %d finished: generated %d problems with seed %d.",
                worker_id,
                len(problems),
                worker_seed,
            )

    problems = []
    for chunk in chunk_results:
        problems.extend(chunk)

    for global_id, problem in enumerate(problems):
        problem.id = global_id

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.mode}-{args.num}.pickle")

    logger.info("Saving generated problems to %s", output_path)
    with open(output_path, "wb") as f:
        pickle.dump(problems, f)

    duration = time.time() - start_time
    logger.info("Total time: %.2f seconds", duration)
    logger.info("Average time per problem: %.2f seconds", duration / args.num)
    logger.info("Logic skeleton generation completed successfully.")
