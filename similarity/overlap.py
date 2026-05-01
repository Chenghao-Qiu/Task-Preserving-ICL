#!/usr/bin/env python
"""Compute lexical overlap between original and perturbed exemplars.

This script measures pairwise overlap for each row only. It does not compare
against other samples or retrieve nearest neighbors.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute token Jaccard and n-gram overlap for text/adv_text pairs."
    )
    parser.add_argument("--input", required=True, help="CSV containing paired text columns.")
    parser.add_argument("--output", required=True, help="Output CSV with lexical overlap columns.")
    parser.add_argument("--original-col", default="text", help="Column before perturbation.")
    parser.add_argument("--replaced-col", default="adv_text", help="Column after perturbation.")
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase text before tokenization.",
    )
    return parser.parse_args()


def tokenize(text: str, lowercase: bool = False) -> list[str]:
    if lowercase:
        text = text.lower()
    return TOKEN_RE.findall(text)


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(left: Iterable[object], right: Iterable[object]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path, keep_default_na=False)
    required_cols = [args.original_col, args.replaced_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s) in {input_path}: {missing_cols}")

    token_jaccard = []
    unigram_overlap = []
    bigram_overlap = []
    trigram_overlap = []

    for _, row in df.iterrows():
        original_tokens = tokenize(str(row[args.original_col]), lowercase=args.lowercase)
        replaced_tokens = tokenize(str(row[args.replaced_col]), lowercase=args.lowercase)

        token_jaccard.append(jaccard(original_tokens, replaced_tokens))
        unigram_overlap.append(jaccard(ngrams(original_tokens, 1), ngrams(replaced_tokens, 1)))
        bigram_overlap.append(jaccard(ngrams(original_tokens, 2), ngrams(replaced_tokens, 2)))
        trigram_overlap.append(jaccard(ngrams(original_tokens, 3), ngrams(replaced_tokens, 3)))

    result = df.copy()
    result["token_jaccard_overlap"] = token_jaccard
    result["unigram_overlap"] = unigram_overlap
    result["bigram_overlap"] = bigram_overlap
    result["trigram_overlap"] = trigram_overlap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    metric_cols = [
        "token_jaccard_overlap",
        "unigram_overlap",
        "bigram_overlap",
        "trigram_overlap",
    ]
    print(result[metric_cols].agg(["mean", "min", "max"]).to_string())
    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()
