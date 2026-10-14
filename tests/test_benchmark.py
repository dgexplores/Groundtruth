import json
from pathlib import Path

import pytest

from evals.benchmark import (
    DEFAULT_CORPUS,
    K_VALUES,
    load_corpus,
    run_benchmark,
)
from evals.regression_gate import evaluate
from src.retriever import Hit

CANDIDATE_CORPUS = {
    "documents": [{"id": f"d{i}", "text": f"document {i}"} for i in range(4)],
    "queries": [
        {"id": "q0", "text": "query a", "relevant_ids": ["d0"]},
        {"id": "q1", "text": "query b", "relevant_ids": ["d1", "d2"]},
    ],
}


class AlwaysRelevantRetriever:
    """Returns every document, so every relevant id is present."""

    def __init__(self, documents):
        self.documents = documents

    def search(self, query, k):
        return [Hit(doc["id"], 1.0) for doc in self.documents][:k]


class AlwaysIrrelevantRetriever:
    """Returns a doc id that is never relevant."""

    def __init__(self, documents):
        self.documents = documents

    def search(self, query, k):
        return [Hit("nope", 1.0)]


def test_perfect_retrieval_scores_one():
    report = run_benchmark(
        CANDIDATE_CORPUS, retriever_factory=AlwaysRelevantRetriever
    )

    assert report["recall_at_5"] == 1.0
    assert report["recall_at_10"] == 1.0


def test_missed_retrieval_scores_zero():
    report = run_benchmark(
        CANDIDATE_CORPUS, retriever_factory=AlwaysIrrelevantRetriever
    )

    assert report["recall_at_5"] == 0.0


def test_multi_relevant_docs_partial_credit():
    """k=1 misses one of two relevant docs, so recall is 0.5 not 0."""
    report = run_benchmark(
        CANDIDATE_CORPUS, retriever_factory=AlwaysRelevantRetriever, k_values=[1]
    )

    assert report["recall_at_1"] == 0.5


def test_report_has_top_level_metric_keys_for_the_gate():
    report = run_benchmark(CANDIDATE_CORPUS, retriever_factory=AlwaysRelevantRetriever)

    for k in K_VALUES:
        assert isinstance(report[f"recall_at_{k}"], float)
    assert report["metric"] == "recall_at_10"
    passed, _ = evaluate(report, report, metric="recall_at_10")
    assert passed is True


def test_report_counts_and_detail():
    report = run_benchmark(CANDIDATE_CORPUS, retriever_factory=AlwaysRelevantRetriever)

    assert report["corpus_size"] == 4
    assert report["total_queries"] == 2
    assert len(report["queries"]) == 2
    assert {"id", "query", "recall_at_5", "recall_at_10"} <= set(report["queries"][0])


def test_empty_queries_score_zero():
    report = run_benchmark(
        {"documents": [{"id": "d", "text": "t"}], "queries": []},
        retriever_factory=AlwaysRelevantRetriever,
    )

    assert report["recall_at_10"] == 0.0
    assert report["total_queries"] == 0


def test_bundled_corpus_is_well_formed():
    corpus = load_corpus(DEFAULT_CORPUS)

    doc_ids = {doc["id"] for doc in corpus["documents"]}
    assert len(corpus["documents"]) == 50
    assert len(corpus["queries"]) == 50
    for query in corpus["queries"]:
        assert query["relevant_ids"], f"{query['id']} has no relevant docs"
        for doc_id in query["relevant_ids"]:
            assert doc_id in doc_ids, f"{query['id']} points at missing {doc_id}"


def test_load_corpus_rejects_missing_keys(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"queries": []}))

    with pytest.raises(ValueError, match="documents"):
        load_corpus(bad)


def test_load_corpus_rejects_document_without_text(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"documents": [{"id": "d"}], "queries": []}))

    with pytest.raises(ValueError, match="id.*text"):
        load_corpus(bad)


def test_load_corpus_rejects_query_without_relevance(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"documents": [{"id": "d", "text": "t"}], "queries": [{"text": "q"}]}))

    with pytest.raises(ValueError, match="relevant_ids"):
        load_corpus(bad)
