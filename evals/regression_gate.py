"""Fail when a metric regresses beyond a tolerated drop.

Usage:
    python -m evals.regression_gate <baseline.json> <current.json> [--metric KEY]

The metric name defaults to ``exact_match``, which is what ``evals.runner``
emits. Pass ``--metric recall`` once a real retrieval benchmark reports it.

A missing metric key is an error, not a zero: treating an absent key as 0.0
would make a malformed report look like a total regression, and a renamed key
would silently disable the gate.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_METRIC = "exact_match"
DEFAULT_THRESHOLD = 0.05


class MissingMetricError(KeyError):
    """Raised when a report does not carry the metric being compared."""


def _read_metric(report: dict[str, Any], metric: str, label: str) -> float:
    if metric not in report:
        available = sorted(k for k, v in report.items() if isinstance(v, (int, float)))
        raise MissingMetricError(
            f"{label} report has no '{metric}' key; numeric keys present: {available}"
        )
    value = report[metric]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} report metric '{metric}' is not numeric: {value!r}")
    return float(value)


def evaluate(
    baseline: dict[str, Any],
    current: dict[str, Any],
    metric: str = DEFAULT_METRIC,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[bool, str]:
    """Return ``(passed, message)`` without touching sys.exit or stdout."""
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")

    baseline_value = _read_metric(baseline, metric, "baseline")
    current_value = _read_metric(current, metric, "current")
    floor = baseline_value - threshold

    if current_value < floor:
        return False, (
            f"Regression detected: {metric} {current_value:.4f} dropped below "
            f"baseline {baseline_value:.4f} - tolerance {threshold:.4f} "
            f"(floor {floor:.4f})"
        )
    return True, f"Regression gate passed: {metric} {current_value:.4f} >= floor {floor:.4f}"


def run_regression_gate(
    baseline_path: Path,
    current_path: Path,
    metric: str = DEFAULT_METRIC,
    threshold: float = DEFAULT_THRESHOLD,
) -> int:
    with open(baseline_path, "r", encoding="utf-8") as handle:
        baseline = json.load(handle)
    with open(current_path, "r", encoding="utf-8") as handle:
        current = json.load(handle)

    passed, message = evaluate(baseline, current, metric, threshold)
    print(message)
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="baseline report JSON")
    parser.add_argument("current", type=Path, help="current report JSON")
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)

    for path in (args.baseline, args.current):
        if not path.exists():
            parser.error(f"report not found: {path}")

    try:
        return run_regression_gate(args.baseline, args.current, args.metric, args.threshold)
    except (MissingMetricError, TypeError) as exc:
        print(f"Gate error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
