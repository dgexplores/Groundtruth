"""Lexical retriever: TF-IDF over document text, cosine-similarity ranking.

Deterministic and offline. Scores are sorted by (-score, id) so equal
scores always rank in a stable order regardless of corpus order.
"""

from typing import Any, Dict, List, NamedTuple

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
