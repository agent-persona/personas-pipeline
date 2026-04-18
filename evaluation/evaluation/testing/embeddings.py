"""Thin wrapper around sentence-transformers for embeddings.

All embedding calls go through this so the model can be swapped.
Uses local models — zero cost, zero API dependency, runs in CI.
"""

from __future__ import annotations

import numpy as np


class Embedder:
    """Wrapper for sentence-transformers embedding model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        vec = self._model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        vecs = self._model.encode(texts, convert_to_numpy=True)
        return vecs.tolist()

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        vecs = self._model.encode([text_a, text_b], convert_to_numpy=True)
        cos = np.dot(vecs[0], vecs[1]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]) + 1e-10)
        return float(cos)

    @staticmethod
    def vector_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two pre-computed embedding vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-10)

    def retrieval_score(self, query: str, passages: list[str], top_k: int = 3) -> list[tuple[int, float]]:
        """Retrieve top-k passages by similarity to query.

        Returns list of (passage_index, similarity_score).
        """
        q_vec = self._model.encode(query, convert_to_numpy=True)
        p_vecs = self._model.encode(passages, convert_to_numpy=True)
        sims = np.dot(p_vecs, q_vec) / (
            np.linalg.norm(p_vecs, axis=1) * np.linalg.norm(q_vec) + 1e-10
        )
        top_indices = np.argsort(sims)[-top_k:][::-1]
        return [(int(i), float(sims[i])) for i in top_indices]
