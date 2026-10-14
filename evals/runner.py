"""Evaluate the golden dataset and emit a machine-readable report.

The report is the input contract for ``evals.regression_gate``. It deliberately
reports ``exact_match`` rather than ``recall``: this harness scores a model's
answer text, not a retrieval result set, so the two metrics must not be
conflated. See ``evaluate_retrieval`` in ``src.evaluator`` for the real
Recall@K, which this runner does not yet exercise.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
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

    This is a fixed lookup, not a model. Any metric derived from it measures
    the lookup table, so reports always carry ``model: stub:not-a-real-model``.
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
    """Summarise per-item results into the gate's input contract."""
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
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_DIR / "latest.json")
    args = parser.parse_args(argv)

    if not args.dataset.exists():
        parser.error(f"dataset not found: {args.dataset}")

    report = build_report(run_evaluation(load_data(args.dataset)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"model:    {report['model']}")
    print(f"dataset:  {args.dataset}")
    print(f"exact_match: {report['exact_match']:.4f} ({report['matched']}/{report['total']})")
    print(f"report:   {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
