import json
from pathlib import Path

import pytest

import evals.runner as runner_module
from evals.runner import (
    DEFAULT_DATASET,
    STUB_MODEL_ID,
    build_report,
    load_data,
    run_evaluation,
)


def test_dataset_path_resolves_relative_to_the_module():
    """The old hardcoded absolute path only worked on one machine."""
    assert DEFAULT_DATASET.is_absolute()
    assert DEFAULT_DATASET.name == "golden_dataset.json"
    assert DEFAULT_DATASET.exists()


def test_no_hardcoded_home_paths_in_runner_source():
    """Paths must come from __file__, never from someone's machine."""
    source = Path(runner_module.__file__).read_text()

    assert "/Users/" not in source
    assert "dgsmacbook" not in source


def test_load_data_rejects_non_list():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.json"
        bad.write_text(json.dumps({"input": "not a list"}))
        with pytest.raises(ValueError):
            load_data(bad)


def test_bundled_dataset_is_well_formed():
    data = load_data(DEFAULT_DATASET)
    assert len(data) > 0
    for item in data:
        assert set(item) == {"input", "expected_output"}


def test_run_evaluation_uses_injected_predictor():
    """predict is injectable, so the stub is not welded into the harness."""
    data = [{"input": "q", "expected_output": "a"}]

    results = run_evaluation(data, predict=lambda _: "a")
    assert results[0]["match"] is True

    results = run_evaluation(data, predict=lambda _: "b")
    assert results[0]["match"] is False


def test_build_report_computes_exact_match():
    results = [
        {"input": "a", "expected_output": "x", "actual": "x", "match": True},
        {"input": "b", "expected_output": "y", "actual": "z", "match": False},
    ]

    report = build_report(results)

    assert report["exact_match"] == 0.5
    assert report["matched"] == 1
    assert report["total"] == 2


def test_report_does_not_claim_to_be_recall():
    """Guards against reintroducing a 'recall' key that means exact match."""
    report = build_report([])

    assert "recall" not in report
    assert report["metric"] == "exact_match"


def test_report_declares_the_stub_model():
    """A report must never be mistakable for a real model evaluation."""
    report = build_report([])

    assert report["model"] == STUB_MODEL_ID
    assert "stub" in report["model"]


def test_empty_results_do_not_divide_by_zero():
    report = build_report([])

    assert report["exact_match"] == 0.0
    assert report["total"] == 0


def test_end_to_end_report_satisfies_the_gate_contract():
    from evals.regression_gate import evaluate

    data = load_data(DEFAULT_DATASET)
    report = build_report(run_evaluation(data))
    passed, _ = evaluate(report, report)

    assert passed is True


def test_main_retrieval_mode_is_the_default(tmp_path):
    out = tmp_path / "r.json"

    assert runner_module.main(["--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert "recall_at_10" in report
    assert "recall_at_5" in report


def test_main_retrieval_mode_report_feeds_the_gate(tmp_path):
    from evals.regression_gate import evaluate

    out = tmp_path / "r.json"
    runner_module.main(["--out", str(out)])

    report = json.loads(out.read_text())
    passed, _ = evaluate(report, report, metric="recall_at_10")
    assert passed is True


def test_main_qa_mode_still_writes_exact_match(tmp_path):
    out = tmp_path / "r.json"

    assert runner_module.main(["--mode", "qa", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["metric"] == "exact_match"
    assert 0.0 <= report["exact_match"] <= 1.0


def test_main_rejects_unknown_mode(tmp_path):
    with pytest.raises(SystemExit):
        runner_module.main(["--mode", "neither", "--out", str(tmp_path / "x.json")])
