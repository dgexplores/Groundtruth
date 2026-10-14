import pytest

from src.retriever import Hit, TfidfRetriever

DOCS = [
    {"id": "d1", "text": "Paris is the capital city of France."},
    {"id": "d2", "text": "Python is a programming language with a GIL."},
    {"id": "d3", "text": "Paris is also known for the Louvre museum."},
]


def test_ranks_relevant_doc_first():
    hits = TfidfRetriever(DOCS).search("What is the capital of France?", k=3)

    assert hits[0].id == "d1"


def test_returns_hits_with_scores():
    hits = TfidfRetriever(DOCS).search("capital France", k=3)

    assert isinstance(hits[0], Hit)
    assert 0 < hits[0].score <= 1


def test_respects_k_bound():
    retriever = TfidfRetriever(DOCS)

    assert len(retriever.search("Paris", k=1)) == 1
    assert len(retriever.search("Paris", k=2)) == 2


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="k must be >= 1"):
        TfidfRetriever(DOCS).search("Paris", k=0)

    with pytest.raises(ValueError, match="k must be >= 1"):
        TfidfRetriever(DOCS).search("Paris", k=-3)


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="unique"):
        TfidfRetriever([
            {"id": "d", "text": "first"},
            {"id": "d", "text": "second"},
        ])


def test_empty_corpus_returns_no_hits():
    assert TfidfRetriever([]).search("anything", k=5) == []


def test_zero_score_docs_excluded():
    hits = TfidfRetriever(DOCS).search("zebra quantum", k=5)

    assert hits == []


def test_deterministic_ordering():
    retriever = TfidfRetriever(DOCS)

    first = retriever.search("Paris France capital", k=3)
    second = retriever.search("Paris France capital", k=3)

    assert first == second


def test_ties_break_by_id():
    docs = [
        {"id": "b", "text": "identical text repeated here"},
        {"id": "a", "text": "identical text repeated here"},
    ]

    hits = TfidfRetriever(docs).search("identical text", k=2)

    assert [hit.id for hit in hits] == ["a", "b"]


def test_single_digit_tokens_are_indexed():
    """sklearn's default token_pattern drops 1-char tokens, leaving
    numeric queries like "2 + 2" with an empty vector and zero hits."""
    docs = [
        {"id": "math", "text": "The arithmetic expression 2 + 2 evaluates to 4."},
        {"id": "other", "text": "Unrelated sentence about clouds."},
    ]

    hits = TfidfRetriever(docs).search("What is 2 + 2?", k=1)

    assert hits[0].id == "math"
