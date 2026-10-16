import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import evals.runner as runner_module
from evals.runner import (
    DEFAULT_DATASET,
    STUB_MODEL_ID,
    build_report,
    gemini_llm_predict,
    load_data,
    normalize_answer,
    real_llm_predict,
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


def test_main_retrieval_report_has_matrix_cells(tmp_path):
    out = tmp_path / "r.json"

    runner_module.main(["--out", str(out)])
    report = json.loads(out.read_text())

    assert report["headline_retriever"] == "dense"
    assert report["headline_tier"] == "semantic"
    for alias in ("tfidf", "dense"):
        for tier in ("lexical", "semantic"):
            assert f"{alias}_{tier}_recall_at_10" in report


def test_main_retrievers_flag_selects_cells(tmp_path):
    out = tmp_path / "r.json"

    assert (
        runner_module.main(
            ["--out", str(out), "--retrievers", "tfidf", "--headline", "tfidf:lexical"]
        )
        == 0
    )
    report = json.loads(out.read_text())

    assert set(report["retrievers"]) == {"tfidf"}
    assert "tfidf_semantic_recall_at_10" in report
    assert "dense_lexical_recall_at_10" not in report


def test_main_rejects_unknown_retriever(tmp_path):
    with pytest.raises(SystemExit):
        runner_module.main(
            ["--out", str(tmp_path / "x.json"), "--retrievers", "tfidf,bm25"]
        )


def test_main_rejects_malformed_headline(tmp_path):
    with pytest.raises(SystemExit):
        runner_module.main(["--out", str(tmp_path / "x.json"), "--headline", "dense"])


def test_main_rejects_headline_missing_from_matrix(tmp_path):
    out = tmp_path / "r.json"

    with pytest.raises(ValueError, match="not in matrix"):
        runner_module.main(
            ["--out", str(out), "--retrievers", "tfidf", "--headline", "dense:semantic"]
        )


def test_normalize_answer_ignores_surface_differences():
    assert normalize_answer("  Paris. ") == "paris"
    assert normalize_answer("THE Capital of France") == "capital of france"
    assert normalize_answer("an  apple") == "apple"


def test_run_evaluation_matches_after_normalization():
    data = [{"input": "q", "expected_output": "Paris"}]

    results = run_evaluation(data, predict=lambda _: "paris.")

    assert results[0]["match"] is True


def test_build_report_records_the_model():
    report = build_report([], model="gpt-4o-mini")

    assert report["model"] == "gpt-4o-mini"


class _FakeCompletions:
    def __init__(self, client):
        self.client = client

    def create(self, **kwargs):
        self.client.kwargs = kwargs
        message = SimpleNamespace(content=self.client.answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeOpenAI:
    instances = []

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.answer = "Paris."
        self.chat = SimpleNamespace(completions=_FakeCompletions(self))
        _FakeOpenAI.instances.append(self)


@pytest.fixture
def fake_openai(monkeypatch):
    _FakeOpenAI.instances.clear()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return _FakeOpenAI


def test_real_predictor_calls_chat_completions(fake_openai):
    answer = real_llm_predict("What is the capital of France?", model="gpt-4o-mini")

    assert answer == "Paris."
    client = fake_openai.instances[0]
    assert client.api_key == "test-key"
    assert client.kwargs["model"] == "gpt-4o-mini"
    assert client.kwargs["temperature"] == 0
    roles = [message["role"] for message in client.kwargs["messages"]]
    assert roles == ["system", "user"]


def test_real_predictor_needs_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        real_llm_predict("anything")


def test_main_qa_real_predictor_labels_the_model(tmp_path, fake_openai, monkeypatch):
    out = tmp_path / "r.json"
    monkeypatch.setenv("QA_MODEL", "gpt-4o-mini")

    assert (
        runner_module.main(
            ["--mode", "qa", "--predictor", "real", "--out", str(out)]
        )
        == 0
    )
    report = json.loads(out.read_text())

    assert report["model"] == "gpt-4o-mini"
    assert report["metric"] == "exact_match"
    assert report["total"] == len(load_data(DEFAULT_DATASET))


def test_main_qa_real_predictor_needs_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SystemExit):
        runner_module.main(
            ["--mode", "qa", "--predictor", "real", "--out", str(tmp_path / "x.json")]
        )


class _FakeGenaiModels:
    failures_before_success = 0

    def __init__(self, client):
        self.client = client

    def generate_content(self, **kwargs):
        if _FakeGenaiModels.failures_before_success > 0:
            _FakeGenaiModels.failures_before_success -= 1
            raise _FakeServerError("transient")
        self.client.kwargs = kwargs
        return SimpleNamespace(text=self.client.answer)


class _FakeServerError(Exception):
    pass


class _FakeClientError(Exception):
    def __init__(self, code):
        self.code = code


class _FakeGenaiClient:
    instances = []

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.answer = "Paris."
        self.models = _FakeGenaiModels(self)
        _FakeGenaiClient.instances.append(self)


@pytest.fixture
def fake_genai(monkeypatch):
    _FakeGenaiClient.instances.clear()
    _FakeGenaiModels.failures_before_success = 0
    fake_genai_module = SimpleNamespace(
        Client=_FakeGenaiClient,
        types=SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs),
        errors=SimpleNamespace(
            ServerError=_FakeServerError, ClientError=_FakeClientError
        ),
    )
    monkeypatch.setitem(
        sys.modules, "google", SimpleNamespace(genai=fake_genai_module)
    )
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai_module)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return _FakeGenaiClient


def test_gemini_predictor_calls_generate_content(fake_genai):
    answer = gemini_llm_predict("What is the capital of France?")

    assert answer == "Paris."
    client = fake_genai.instances[0]
    assert client.api_key == "test-key"
    assert client.kwargs["model"] == "gemini-2.5-flash"
    assert client.kwargs["contents"] == "What is the capital of France?"
    assert client.kwargs["config"]["temperature"] == 0


def test_gemini_predictor_needs_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        gemini_llm_predict("anything")


def test_gemini_predictor_retries_transient_server_errors(fake_genai, monkeypatch):
    import time as time_module

    monkeypatch.setattr(time_module, "sleep", lambda seconds: None)
    _FakeGenaiModels.failures_before_success = 2

    assert gemini_llm_predict("What is the capital of France?") == "Paris."


def test_main_qa_gemini_predictor_labels_the_model(tmp_path, fake_genai):
    out = tmp_path / "r.json"

    assert (
        runner_module.main(
            ["--mode", "qa", "--predictor", "gemini", "--out", str(out)]
        )
        == 0
    )
    report = json.loads(out.read_text())

    assert report["model"] == "gemini-2.5-flash"
    assert report["metric"] == "exact_match"
    assert report["total"] == len(load_data(DEFAULT_DATASET))


def test_main_qa_gemini_predictor_needs_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(SystemExit):
        runner_module.main(
            ["--mode", "qa", "--predictor", "gemini", "--out", str(tmp_path / "x.json")]
        )
