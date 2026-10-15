import numpy as np
import pytest

from src.retriever import DenseRetriever, Hit, TfidfRetriever

DOCS = [
    {"id": "d1", "text": "Paris is the capital city of France."},
    {"id": "d2", "text": "Python is a programming language with a GIL."},
    {"id": "d3", "text": "Paris is also known for the Louvre museum."},
]


def marker_encoder(texts):
    """Deterministic fake encoder: dim 0 counts 'apple', dim 1 counts 'banana'.

    Gives a 2-D vector space with known geometry and no model download.
    """
    rows = []
    for text in texts:
        words = text.lower().split()
        rows.append([words.count("apple"), words.count("banana")])
    return np.asarray(rows, dtype=np.float64)


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


FRUIT_DOCS = [
    {"id": "a", "text": "apple apple apple"},
    {"id": "b", "text": "banana banana"},
    {"id": "c", "text": "apple banana"},
]


def test_dense_ranks_by_embedding_similarity():
    hits = DenseRetriever(FRUIT_DOCS, encoder=marker_encoder).search(
        "apple", k=3
    )

    assert hits[0].id == "a"
    assert hits[0].score > hits[-1].score


def test_dense_returns_all_docs_even_at_zero_score():
    """Dense scores are never exactly absent, so nothing is filtered out."""
    hits = DenseRetriever(FRUIT_DOCS, encoder=marker_encoder).search(
        "cherry", k=3
    )

    assert len(hits) == 3


def test_dense_respects_k_bound():
    retriever = DenseRetriever(FRUIT_DOCS, encoder=marker_encoder)

    assert len(retriever.search("apple", k=1)) == 1
    assert len(retriever.search("apple", k=2)) == 2


def test_dense_k_must_be_positive():
    with pytest.raises(ValueError, match="k must be >= 1"):
        DenseRetriever(FRUIT_DOCS, encoder=marker_encoder).search("apple", k=0)


def test_dense_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="unique"):
        DenseRetriever(
            [
                {"id": "d", "text": "apple"},
                {"id": "d", "text": "banana"},
            ],
            encoder=marker_encoder,
        )


def test_dense_empty_corpus_returns_no_hits():
    assert DenseRetriever([], encoder=marker_encoder).search("apple", k=5) == []


def test_dense_ties_break_by_id():
    docs = [
        {"id": "b", "text": "apple"},
        {"id": "a", "text": "apple"},
    ]

    hits = DenseRetriever(docs, encoder=marker_encoder).search("apple", k=2)

    assert [hit.id for hit in hits] == ["a", "b"]


def test_dense_deterministic_ordering():
    retriever = DenseRetriever(FRUIT_DOCS, encoder=marker_encoder)

    first = retriever.search("apple banana", k=3)
    second = retriever.search("apple banana", k=3)

    assert first == second


def test_dense_normalizes_unnormalized_encoder_output():
    """Scores must be cosine (<= 1), even if the encoder returns raw vectors."""
    def scaled_encoder(texts):
        return marker_encoder(texts) * 100.0

    hits = DenseRetriever(FRUIT_DOCS, encoder=scaled_encoder).search("apple", k=1)

    assert hits[0].id == "a"
    assert hits[0].score <= 1.0


def test_dense_rejects_encoder_with_wrong_row_count():
    def bad_encoder(texts):
        return np.asarray([[1.0, 0.0]])

    with pytest.raises(ValueError, match="one row per document"):
        DenseRetriever(FRUIT_DOCS, encoder=bad_encoder)
