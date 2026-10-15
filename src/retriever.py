"""Retrievers: lexical (TF-IDF) and semantic (dense sentence embeddings).

Both rank by cosine similarity with deterministic (-score, id) tie-breaks
so equal scores always rank in a stable order regardless of corpus order.

TfidfRetriever is offline and dependency-light. DenseRetriever uses a
sentence-transformers model; the model import is lazy so importing this
module never loads torch, and the encoder is injectable so tests can run
without downloading a model.
"""

from typing import Any, Callable, Dict, List, NamedTuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Hit(NamedTuple):
    id: str
    score: float


class TfidfRetriever:
    """Rank corpus documents against a query by TF-IDF cosine similarity."""

    # sklearn's default token_pattern requires 2+ chars, which silently
    # drops single-digit tokens ("2 + 2" -> empty query vector).
    TOKEN_PATTERN = r"(?u)\b\w+\b"

    def __init__(self, documents: List[Dict[str, Any]]):
        ids = [doc["id"] for doc in documents]
        if len(set(ids)) != len(ids):
            raise ValueError("document ids must be unique")
        self._ids = ids
        self._texts = [doc["text"] for doc in documents]
        self._vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", token_pattern=self.TOKEN_PATTERN
        )
        self._matrix = (
            self._vectorizer.fit_transform(self._texts) if self._texts else None
        )

    def search(self, query: str, k: int = 5) -> List[Hit]:
        """Return up to ``k`` hits, best first, excluding zero-score docs."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if self._matrix is None:
            return []

        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix)[0]
        scored = [
            (float(scores[i]), self._ids[i])
            for i in range(len(self._ids))
            if scores[i] > 0
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [Hit(doc_id, score) for score, doc_id in scored[:k]]


class DenseRetriever:
    """Rank corpus documents against a query by embedding cosine similarity.

    ``encoder`` maps a list of texts to a 2-D array of equal-length vectors
    (one row per text); rows are L2-normalised here, so an encoder may
    return raw vectors. Defaults to the cached sentence-transformers model
    ``MODEL_NAME``. Unlike the sparse TF-IDF retriever, every document gets
    a score, so no score filtering is applied.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(
        self,
        documents: List[Dict[str, Any]],
        encoder: Callable[[List[str]], np.ndarray] | None = None,
    ):
        ids = [doc["id"] for doc in documents]
        if len(set(ids)) != len(ids):
            raise ValueError("document ids must be unique")
        self._ids = ids
        self._encoder = encoder if encoder is not None else _default_encoder
        if not documents:
            self._matrix = None
            return
        vectors = np.asarray(
            self._encoder([doc["text"] for doc in documents]), dtype=np.float64
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(documents):
            raise ValueError(
                "encoder must return a 2-D array with one row per document, "
                f"got shape {vectors.shape} for {len(documents)} documents"
            )
        self._matrix = _row_normalize(vectors)

    def search(self, query: str, k: int = 5) -> List[Hit]:
        """Return the top ``k`` hits by cosine similarity, best first."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if self._matrix is None:
            return []

        query_vector = _row_normalize(np.asarray(self._encoder([query]), dtype=np.float64))
        scores = self._matrix @ query_vector[0]
        scored = [
            (float(scores[i]), self._ids[i]) for i in range(len(self._ids))
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [Hit(doc_id, score) for score, doc_id in scored[:k]]


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows, leaving zero rows as zeros."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


_MODEL: Any = None


def _default_encoder(texts: List[str]) -> np.ndarray:
    """Encode with the sentence-transformers model, loaded once per process."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(DenseRetriever.MODEL_NAME)
    return _MODEL.encode(list(texts), convert_to_numpy=True)
