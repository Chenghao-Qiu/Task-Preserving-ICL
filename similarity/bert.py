#!/usr/bin/env python
"""Measure exemplar semantic similarity before/after replacement.

This follows the KATE idea from Liu et al. (2021): encode examples with a
RoBERTa sentence encoder and compare semantic neighbors/similarity in the
embedding space. For the current AICL exemplar files, the paired comparison is
between the original exemplar column (`text`) and the replaced/adversarial
column (`adv_text`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_INPUT = "exemplars/sst2_checklist_matched.csv"
DEFAULT_OUTPUT = "similarity/sst2_checklist_matched_roberta_similarity.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use RoBERTa embeddings, in the spirit of KATE kNN retrieval, to "
            "measure semantic similarity before and after exemplar replacement."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="CSV containing exemplar pairs.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV with similarity scores.")
    parser.add_argument("--original-col", default="text", help="Column before replacement.")
    parser.add_argument("--replaced-col", default="adv_text", help="Column after replacement.")
    parser.add_argument("--model", default="roberta-base", help="HuggingFace RoBERTa model name/path.")
    parser.add_argument(
        "--pooling",
        choices=("mean", "cls"),
        default="mean",
        help="Pooling strategy for token embeddings. KATE used CLS/mean variants.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size.")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max sequence length.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional HuggingFace cache directory for offline/shared model storage.",
    )
    parser.add_argument(
        "--show-worst",
        type=int,
        default=5,
        help="Print this many lowest-similarity pairs for quick inspection.",
    )
    return parser.parse_args()


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@torch.no_grad()
def encode_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    pooling: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    model.eval()
    embeddings = []

    for batch in tqdm(list(batched(texts, batch_size)), desc="Encoding", unit="batch"):
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = model(**encoded)

        if pooling == "cls":
            pooled = output.last_hidden_state[:, 0, :]
        else:
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])

        embeddings.append(pooled.detach().cpu())

    return torch.cat(embeddings, dim=0).numpy()


def describe(scores: pd.Series, name: str = "all") -> str:
    return (
        f"{name}: n={scores.size}, mean={scores.mean():.4f}, std={scores.std(ddof=0):.4f}, "
        f"min={scores.min():.4f}, p25={scores.quantile(0.25):.4f}, "
        f"median={scores.median():.4f}, p75={scores.quantile(0.75):.4f}, max={scores.max():.4f}"
    )


def assert_within_max_length(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    original_col: str,
    replaced_col: str,
    max_length: int,
) -> None:
    too_long = []
    for row_idx, row in df.iterrows():
        for col in (original_col, replaced_col):
            token_count = len(tokenizer(str(row[col]), add_special_tokens=True)["input_ids"])
            if token_count > max_length:
                too_long.append((row_idx, col, token_count))

    if not too_long:
        return

    examples = "\n".join(
        f"  row={row_idx}, column={col}, tokens={token_count}, max_length={max_length}"
        for row_idx, col, token_count in too_long[:10]
    )
    remaining = len(too_long) - 10
    if remaining > 0:
        examples += f"\n  ... and {remaining} more over-length text(s)"

    raise ValueError(
        "Input contains text longer than --max-length. Increase --max-length or shorten the input.\n"
        + examples
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path, keep_default_na=False)
    required_cols = [args.original_col, args.replaced_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s) in {input_path}: {missing_cols}")

    originals = df[args.original_col].astype(str).tolist()
    replacements = df[args.replaced_col].astype(str).tolist()
    all_texts = originals + replacements

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = AutoModel.from_pretrained(args.model, cache_dir=args.cache_dir).to(device)
    assert_within_max_length(
        df,
        tokenizer=tokenizer,
        original_col=args.original_col,
        replaced_col=args.replaced_col,
        max_length=args.max_length,
    )

    embeddings = encode_texts(
        all_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        pooling=args.pooling,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    original_emb = torch.from_numpy(embeddings[: len(df)])
    replaced_emb = torch.from_numpy(embeddings[len(df) :])
    cosine_scores = F.cosine_similarity(original_emb, replaced_emb, dim=1).numpy()
    l2_distances = torch.linalg.vector_norm(original_emb - replaced_emb, ord=2, dim=1).numpy()

    result = df.copy()
    result["roberta_cosine_similarity"] = cosine_scores
    result["roberta_cosine_distance"] = 1.0 - result["roberta_cosine_similarity"]
    result["roberta_l2_distance"] = l2_distances
    result["encoder_model"] = args.model
    result["pooling"] = args.pooling

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(describe(result["roberta_cosine_similarity"]))
    if args.show_worst > 0:
        print("\nLowest-similarity pairs:")
        view_cols = [args.original_col, args.replaced_col, "roberta_cosine_similarity"]
        worst = result.nsmallest(args.show_worst, "roberta_cosine_similarity")
        print(worst[view_cols].to_string(index=False))

    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()
