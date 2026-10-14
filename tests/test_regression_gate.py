import json

import pytest

from evals.regression_gate import (
    DEFAULT_METRIC,
    MissingMetricError,
    evaluate,
    main,
)


def test_metric_defaults_to_exact_match():
    assert DEFAULT_METRIC == "exact_match"


def test_within_tolerance_passes():
    baseline = {"exact_match": 0.85}
    current = {"exact_match": 0.82}

    passed, message = evaluate(baseline, current)

    assert passed is True
    assert "passed" in message


def test_drop_beyond_tolerance_fails():
    baseline = {"exact_match": 0.85}
    current = {"exact_match": 0.79}

    passed, message = evaluate(baseline, current)

    assert passed is False
    assert "Regression detected" in message


def test_drop_exactly_at_tolerance_passes():
    """The floor is inclusive: only a drop *beyond* the tolerance fails."""
    baseline = {"exact_match": 0.85}
    current = {"exact_match": 0.80}

    passed, _ = evaluate(baseline, current, threshold=0.05)

    assert passed is True


def test_improvement_passes():
    passed, _ = evaluate({"exact_match": 0.10}, {"exact_match": 0.95})

    assert passed is True


def test_missing_metric_raises_rather_than_scoring_zero():
    """A renamed or malformed report must fail loudly, not silently pass or fail."""
    with pytest.raises(MissingMetricError):
        evaluate({"recall": 0.9}, {"exact_match": 0.9})


def test_non_numeric_metric_raises():
    with pytest.raises(TypeError):
        evaluate({"exact_match": "high"}, {"exact_match": 0.9})


def test_boolean_is_not_accepted_as_numeric():
    with pytest.raises(TypeError):
        evaluate({"exact_match": True}, {"exact_match": 0.9})


def test_negative_threshold_rejected():
    with pytest.raises(ValueError):
        evaluate({"exact_match": 0.5}, {"exact_match": 0.5}, threshold=-0.1)


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_main_returns_zero_when_gated_in(tmp_path):
    baseline = _write(tmp_path, "baseline.json", {"exact_match": 0.85})
    current = _write(tmp_path, "current.json", {"exact_match": 0.85})

    assert main([str(baseline), str(current)]) == 0


def test_main_returns_one_on_regression(tmp_path):
    baseline = _write(tmp_path, "baseline.json", {"exact_match": 0.85})
    current = _write(tmp_path, "current.json", {"exact_match": 0.10})

    assert main([str(baseline), str(current)]) == 1


def test_main_returns_two_on_missing_metric(tmp_path):
    """Exit 2 distinguishes a broken harness from a real regression."""
    baseline = _write(tmp_path, "baseline.json", {"exact_match": 0.85})
    current = _write(tmp_path, "current.json", {"recall": 0.85})

    assert main([str(baseline), str(current)]) == 2
