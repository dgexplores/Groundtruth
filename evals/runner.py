"""Evaluate against the golden set and emit a machine-readable report.

Two modes:
  retrieval (default) runs the Recall@K benchmark over evals/corpus.json
  qa scores exact-match answers against evals/golden_dataset.json

Reports are the input contract for ``evals.regression_gate``. The qa
mode reports ``exact_match``; the retrieval mode reports
``recall_at_5`` and ``recall_at_10``. The two must not be conflated:
exact_match scores an answer string, recall scores a retrieved set.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

from evals.benchmark import DEFAULT_CORPUS, load_corpus, run_benchmark

DEFAULT_DATASET = Path(__file__).resolve().parent / "golden_dataset.json"
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "reports"

STUB_MODEL_ID = "stub:not-a-real-model"


def load_data(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"dataset must be a JSON list, got {type(data).__name__}")
    return data


def simulate_llm(input_text: str) -> str:
    """Deterministic stand-in for a model call.

    This is a fixed lookup, not a model. Any metric derived from it
    measures the lookup table, so reports always carry
    ``model: stub:not-a-real-model``.
    """
    if "capital of France" in input_text:
        return "Paris"
    if "2 + 2" in input_text:
        return "4"
    if "GIL" in input_text:
        return "Python"
    if "Hermes Agent" in input_text:
        return "Nous Research"
    if "1991" in input_text:
        return "1991"
    if "first version of Python" in input_text:
        return "1991"
    return "Unknown"


def run_evaluation(
    data: List[Dict[str, Any]], predict: Callable[[str], str] = simulate_llm
) -> List[Dict[str, Any]]:
    results = []
    for item in data:
        input_text = item["input"]
        expected = item["expected_output"]
        actual = predict(input_text)
        results.append(
            {
                "input": input_text,
                "expected": expected,
                "actual": actual,
                "match": actual.strip() == expected.strip(),
            }
        )
    return results


def build_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise per-item QA results into the gate's input contract."""
    total = len(results)
    matched = sum(1 for r in results if r["match"])
    return {
        "model": STUB_MODEL_ID,
        "metric": "exact_match",
        "exact_match": matched / total if total else 0.0,
        "matched": matched,
        "total": total,
        "results": results,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("retrieval", "qa"),
        default="retrieval",
        help="retrieval: Recall@K benchmark (default); qa: exact-match scoring",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_DIR / "latest.json")
    args = parser.parse_args(argv)

    if args.mode == "retrieval":
        if not args.corpus.exists():
            parser.error(f"corpus not found: {args.corpus}")
        report = run_benchmark(load_corpus(args.corpus))
    else:
        if not args.dataset.exists():
            parser.error(f"dataset not found: {args.dataset}")
        report = build_report(run_evaluation(load_data(args.dataset)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    if args.mode == "retrieval":
        print(f"retriever: {report['retriever']}")
        for k in (5, 10):
            print(f"recall@{k}:  {report[f'recall_at_{k}']:.4f}")
    else:
        print(f"model:       {report['model']}")
        print(
            f"exact_match: {report['exact_match']:.4f} "
            f"({report['matched']}/{report['total']})"
        )
    print(f"report:      {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
