#!/usr/bin/env python
"""Measure retrieval stability after exemplar perturbation.

For each test query, this script ranks the original exemplar pool and the
perturbed exemplar pool, then reports:

1. Rank shift: rank(perturbed counterpart) - rank(original exemplar).
2. Top-k overlap: fraction of exemplar IDs shared by original and perturbed
   top-k retrieval results.

The comparison is ID-based: row i in `text` and row i in `adv_text` are treated
as the same exemplar before and after perturbation.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = [
    "sst2_checklist",
    "sst2_checklist_matched",
    "proverqa_easy",
    "proverqa_medium",
    "proverqa_hard",
    "problemathic_simple",
    "problemathic_complex",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report rank shift and Top-k overlap for perturbed exemplar retrieval."
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--exemplar-dir", type=Path, default=PROJECT_ROOT / "exemplars")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "similarity" / "retrival")
    parser.add_argument("--query-col", default="text")
    parser.add_argument("--original-col", default="text")
    parser.add_argument("--replaced-col", default="adv_text")
    parser.add_argument("--methods", nargs="+", choices=("tfidf", "bm25"), default=["tfidf", "bm25"])
    parser.add_argument("--top-k", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--lowercase", action="store_true")
    return parser.parse_args()


def tokenize(text: str, lowercase: bool) -> list[str]:
    if lowercase:
        text = text.lower()
    return TOKEN_RE.findall(text)


class BM25Index:
    def __init__(
        self,
        documents: list[str],
        lowercase: bool,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(doc, lowercase) for doc in documents]
        self.doc_lens = np.array([len(doc) for doc in self.docs], dtype=float)
        self.avgdl = float(self.doc_lens.mean()) if len(self.doc_lens) else 0.0
        self.term_freqs = [Counter(doc) for doc in self.docs]
        self.idf = self._build_idf()

    def _build_idf(self) -> dict[str, float]:
        doc_freq = Counter()
        for doc in self.docs:
            doc_freq.update(set(doc))

        n_docs = len(self.docs)
        return {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def score(self, query: str, lowercase: bool) -> np.ndarray:
        query_terms = set(tokenize(query, lowercase))
        scores = np.zeros(len(self.docs), dtype=float)
        if not query_terms or self.avgdl == 0.0:
            return scores

        for doc_idx, term_freq in enumerate(self.term_freqs):
            doc_len = self.doc_lens[doc_idx]
            for term in query_terms:
                tf = term_freq.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl)
                scores[doc_idx] += self.idf.get(term, 0.0) * tf * (self.k1 + 1.0) / denom
        return scores


def rank_from_scores(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((np.arange(len(scores)), -scores))
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks, order


def summarize_query(
    dataset: str,
    method: str,
    query_idx: int,
    orig_scores: np.ndarray,
    pert_scores: np.ndarray,
    top_k_values: list[int],
) -> dict[str, float | int | str]:
    orig_ranks, orig_order = rank_from_scores(orig_scores)
    pert_ranks, pert_order = rank_from_scores(pert_scores)
    rank_shift = pert_ranks - orig_ranks

    row: dict[str, float | int | str] = {
        "dataset": dataset,
        "method": method,
        "query_idx": query_idx,
        "num_exemplars": len(orig_scores),
        "rank_shift_mean": float(rank_shift.mean()),
        "rank_shift_median": float(np.median(rank_shift)),
        "rank_shift_std": float(rank_shift.std(ddof=0)),
        "rank_shift_abs_mean": float(np.abs(rank_shift).mean()),
        "rank_down_rate": float((rank_shift > 0).mean()),
        "rank_up_rate": float((rank_shift < 0).mean()),
        "rank_unchanged_rate": float((rank_shift == 0).mean()),
    }

    for k in top_k_values:
        k_eff = min(k, len(orig_scores))
        orig_top = set(orig_order[:k_eff].tolist())
        pert_top = set(pert_order[:k_eff].tolist())
        row[f"overlap@{k}"] = len(orig_top & pert_top) / k_eff if k_eff else 0.0

    return row


def tfidf_scores(
    queries: list[str],
    originals: list[str],
    perturbed: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    matrix = vectorizer.fit_transform(queries + originals + perturbed)

    n_queries = len(queries)
    n_exemplars = len(originals)
    query_matrix = matrix[:n_queries]
    orig_matrix = matrix[n_queries : n_queries + n_exemplars]
    pert_matrix = matrix[n_queries + n_exemplars :]

    orig_scores = cosine_similarity(query_matrix, orig_matrix)
    pert_scores = cosine_similarity(query_matrix, pert_matrix)
    return orig_scores, pert_scores


def bm25_scores(
    queries: list[str],
    originals: list[str],
    perturbed: list[str],
    lowercase: bool,
) -> tuple[np.ndarray, np.ndarray]:
    orig_index = BM25Index(originals, lowercase=lowercase)
    pert_index = BM25Index(perturbed, lowercase=lowercase)
    orig_scores = np.vstack([orig_index.score(query, lowercase=lowercase) for query in queries])
    pert_scores = np.vstack([pert_index.score(query, lowercase=lowercase) for query in queries])
    return orig_scores, pert_scores


def load_dataset(args: argparse.Namespace, dataset: str) -> tuple[list[str], list[str], list[str]]:
    exemplar_path = args.exemplar_dir / f"{dataset}.csv"
    test_path = args.data_dir / dataset / "validation.csv"

    exemplar_df = pd.read_csv(exemplar_path, keep_default_na=False)
    test_df = pd.read_csv(test_path, keep_default_na=False)
    if args.max_queries is not None:
        test_df = test_df.head(args.max_queries)

    for path, df, cols in [
        (exemplar_path, exemplar_df, [args.original_col, args.replaced_col]),
        (test_path, test_df, [args.query_col]),
    ]:
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f"{path} missing required column(s): {missing}")

    originals = exemplar_df[args.original_col].astype(str).tolist()
    perturbed = exemplar_df[args.replaced_col].astype(str).tolist()
    queries = test_df[args.query_col].astype(str).tolist()
    return queries, originals, perturbed


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "rank_shift_mean",
        "rank_shift_median",
        "rank_shift_abs_mean",
        "rank_down_rate",
        "rank_up_rate",
        "rank_unchanged_rate",
    ] + [col for col in rows.columns if col.startswith("overlap@")]

    return (
        rows.groupby(["dataset", "method"], sort=True)[metric_cols]
        .mean()
        .reset_index()
    )


def to_markdown(summary: pd.DataFrame) -> str:
    headers = summary.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in summary.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if col in {"dataset", "method"}:
                values.append(str(value))
            else:
                values.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in args.datasets:
        queries, originals, perturbed = load_dataset(args, dataset)

        method_scores = {}
        if "tfidf" in args.methods:
            method_scores["tfidf"] = tfidf_scores(queries, originals, perturbed)
        if "bm25" in args.methods:
            method_scores["bm25"] = bm25_scores(queries, originals, perturbed, args.lowercase)

        for method, (orig_scores, pert_scores) in method_scores.items():
            for query_idx in range(len(queries)):
                rows.append(
                    summarize_query(
                        dataset=dataset,
                        method=method,
                        query_idx=query_idx,
                        orig_scores=orig_scores[query_idx],
                        pert_scores=pert_scores[query_idx],
                        top_k_values=args.top_k,
                    )
                )

    per_query = pd.DataFrame(rows)
    summary = aggregate(per_query)

    per_query_path = args.output_dir / "retrieval_stability_per_query.csv"
    summary_csv_path = args.output_dir / "retrieval_stability_summary.csv"
    summary_md_path = args.output_dir / "retrieval_stability_summary.md"

    per_query.to_csv(per_query_path, index=False)
    summary.to_csv(summary_csv_path, index=False)
    summary_md_path.write_text(to_markdown(summary) + "\n", encoding="utf-8")

    print(to_markdown(summary))
    print(f"\nWrote: {per_query_path}")
    print(f"Wrote: {summary_csv_path}")
    print(f"Wrote: {summary_md_path}")


if __name__ == "__main__":
    main()
