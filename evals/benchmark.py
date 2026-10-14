"""Run the retrieval benchmark: index the corpus, retrieve, score Recall@K.

Per-query recall is computed by ``src.evaluator.evaluate_retrieval``;
this module only orchestrates. Metrics are written at the top level of
the report so ``evals.regression_gate`` can compare them by name.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

from src.evaluator import evaluate_retrieval
from src.retriever import TfidfRetriever

DEFAULT_CORPUS = Path(__file__).resolve().parent / "corpus.json"
DEFAULT_REPORT = Path(__file__).resolve().parent / "reports" / "latest.json"
K_VALUES = (5, 10)
PRIMARY_METRIC = "recall_at_10"


def load_corpus(path: Path) -> Dict[str, Any]:
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
    return corpus


def run_benchmark(
    corpus: Dict[str, Any],
    retriever_factory: Callable[[List[Dict[str, Any]]], Any] = TfidfRetriever,
    k_values: List[int] = K_VALUES,
) -> Dict[str, Any]:
    """Evaluate a retriever over a corpus.

    Returns a report with one ``recall_at_<k>`` metric per k value,
    averaged over all queries, plus per-query detail. The retriever is
    injectable so the benchmark can be scored without a real one.
    """
    retriever = retriever_factory(corpus["documents"])
    per_query = []
    for query in corpus["queries"]:
        row: Dict[str, Any] = {
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


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    if not args.corpus.exists():
        parser.error(f"corpus not found: {args.corpus}")

    report = run_benchmark(load_corpus(args.corpus))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"retriever: {report['retriever']}")
    print(f"corpus:    {report['corpus_size']} docs, {report['total_queries']} queries")
    for k in K_VALUES:
        print(f"recall@{k}:  {report[f'recall_at_{k}']:.4f}")
    print(f"report:    {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
