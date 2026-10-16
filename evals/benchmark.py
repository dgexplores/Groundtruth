"""Run the retrieval benchmark: index the corpus, retrieve, score Recall@K.

Runs every retriever in ``DEFAULT_RETRIEVERS`` against every query tier
(``tier`` on each query, default "lexical") and reports one cell per
retriever/tier pair. Per-query recall is computed by
``src.evaluator.evaluate_retrieval``; this module only orchestrates.
Metrics are written at the top level of the report so
``evals.regression_gate`` can compare them by name: the headline cell
appears as ``recall_at_<k>``, every cell also as
``<retriever>_<tier>_recall_at_<k>``.
"""

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from src.evaluator import evaluate_retrieval
from src.retriever import DenseRetriever, TfidfRetriever

DEFAULT_CORPUS = Path(__file__).resolve().parent / "corpus.json"
DEFAULT_REPORT = Path(__file__).resolve().parent / "reports" / "latest.json"
K_VALUES = (5, 10)
PRIMARY_METRIC = "recall_at_10"
DEFAULT_RETRIEVERS: dict[str, Callable[[list[dict[str, Any]]], Any]] = {
    "tfidf": TfidfRetriever,
    "dense": DenseRetriever,
}
DEFAULT_HEADLINE = ("dense", "semantic")


def load_corpus(path: Path) -> dict[str, Any]:
    """Load and validate a corpus file.

    Preconditions: the file is JSON with ``documents`` (each with
    ``id`` and ``text``) and ``queries`` (each with ``text`` and
    ``relevant_ids``).
    """
    with open(path, "r", encoding="utf-8") as handle:
        corpus = json.load(handle)
    for key in ("documents", "queries"):
        if key not in corpus or not isinstance(corpus[key], list):
            raise ValueError(f"corpus must have a '{key}' list")
    for doc in corpus["documents"]:
        if "id" not in doc or "text" not in doc:
            raise ValueError("every document needs 'id' and 'text'")
    for query in corpus["queries"]:
        if "text" not in query or "relevant_ids" not in query:
            raise ValueError("every query needs 'text' and 'relevant_ids'")
        if "tier" in query and not (isinstance(query["tier"], str) and query["tier"]):
            raise ValueError("query 'tier' must be a non-empty string")
    return corpus


def run_benchmark(
    corpus: dict[str, Any],
    retriever_factory: Callable[[list[dict[str, Any]]], Any] = TfidfRetriever,
    k_values: Sequence[int] = K_VALUES,
) -> dict[str, Any]:
    """Evaluate a retriever over a corpus.

    Returns a report with one ``recall_at_<k>`` metric per k value,
    averaged over all queries, plus per-query detail. The retriever is
    injectable so the benchmark can be scored without a real one.
    """
    retriever = retriever_factory(corpus["documents"])
    per_query = []
    for query in corpus["queries"]:
        row: dict[str, Any] = {
            "id": query.get("id", ""),
            "query": query["text"],
        }
        for k in k_values:
            retrieved = [hit.id for hit in retriever.search(query["text"], k)]
            row[f"recall_at_{k}"] = evaluate_retrieval(
                retrieved, query["relevant_ids"]
            )
        per_query.append(row)

    metrics = {}
    for k in k_values:
        values = [row[f"recall_at_{k}"] for row in per_query]
        metrics[f"recall_at_{k}"] = sum(values) / len(values) if values else 0.0

    return {
        "retriever": retriever_factory.__name__,
        "metric": PRIMARY_METRIC,
        **metrics,
        "corpus_size": len(corpus["documents"]),
        "total_queries": len(corpus["queries"]),
        "queries": per_query,
    }


def run_matrix(
    corpus: dict[str, Any],
    retrievers: dict[str, Callable[[list[dict[str, Any]]], Any]] | None = None,
    k_values: Sequence[int] = K_VALUES,
    headline: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Run every retriever against every query tier.

    Returns a flat report: per-cell ``<retriever>_<tier>_recall_at_<k>``
    metrics, plus the headline cell's metrics under their bare names
    (``recall_at_<k>``) so the regression gate compares them by name.
    ``headline`` picks that cell; it must exist in the matrix.

    Precondition: ``retrievers`` names all resolve to DEFAULT_RETRIEVERS
    factories (callers validate aliases before passing them in).
    """
    if retrievers is None:
        retrievers = DEFAULT_RETRIEVERS
    if not retrievers:
        raise ValueError("need at least one retriever")

    tiers = sorted({query.get("tier", "lexical") for query in corpus["queries"]})
    if not tiers:
        raise ValueError("need at least one query")

    cells: dict[str, dict[str, Any]] = {}
    for alias, factory in retrievers.items():
        for tier in tiers:
            subset = {
                **corpus,
                "queries": [
                    q for q in corpus["queries"] if q.get("tier", "lexical") == tier
                ],
            }
            cells.setdefault(alias, {})[tier] = run_benchmark(subset, factory, k_values)

    if headline is None:
        if DEFAULT_HEADLINE[0] in cells and DEFAULT_HEADLINE[1] in cells[DEFAULT_HEADLINE[0]]:
            headline = DEFAULT_HEADLINE
        else:
            headline = (next(iter(cells)), next(iter(cells[next(iter(cells))])))
    headline_alias, headline_tier = headline
    if headline_alias not in cells or headline_tier not in cells[headline_alias]:
        available = sorted(
            f"{alias}:{tier}" for alias, tiers_cells in cells.items() for tier in tiers_cells
        )
        raise ValueError(
            f"headline {headline[0]}:{headline[1]} not in matrix; available: {available}"
        )
    headline_cell = cells[headline_alias][headline_tier]

    report: dict[str, Any] = {
        "metric": PRIMARY_METRIC,
        "retrievers": {
            alias: factory.__name__ for alias, factory in retrievers.items()
        },
        "headline_retriever": headline_alias,
        "headline_tier": headline_tier,
        "corpus_size": len(corpus["documents"]),
        "total_queries": len(corpus["queries"]),
        "cells": cells,
        "queries": headline_cell["queries"],
    }
    for k in k_values:
        report[f"recall_at_{k}"] = headline_cell[f"recall_at_{k}"]
    for alias, tiers_cells in cells.items():
        for tier, cell in tiers_cells.items():
            for k in k_values:
                report[f"{alias}_{tier}_recall_at_{k}"] = cell[f"recall_at_{k}"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    if not args.corpus.exists():
        parser.error(f"corpus not found: {args.corpus}")

    report = run_matrix(load_corpus(args.corpus))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"headline:  {report['headline_retriever']}:{report['headline_tier']}")
    print(f"corpus:    {report['corpus_size']} docs, {report['total_queries']} queries")
    for alias in report["retrievers"]:
        for tier in sorted(report["cells"][alias]):
            for k in K_VALUES:
                print(f"{alias}/{tier} recall@{k}: {report[f'{alias}_{tier}_recall_at_{k}']:.4f}")
    print(f"report:    {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
