"""D5 Behavioral Consistency — repeated query consistency via embedding clusters.

Trustworthiness: MEDIUM (measures surface consistency).
Method: Embed responses, compute centroid + radius of cluster and pairwise cosine similarity.
"""

from __future__ import annotations

import numpy as np

from evaluation.testing.embeddings import Embedder
from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import Persona, EvalResult
from evaluation.testing.source_context import SourceContext


class BehavioralConsistencyScorer(BaseScorer):
    """Evaluates whether a persona's responses cluster tightly (consistent behavior)."""

    dimension_id = "D5"
    dimension_name = "Behavioral Consistency"
    tier = 2

    def __init__(self) -> None:
        self._embedder: Embedder | None = None

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        """Measure consistency of responses via embedding cluster tightness.

        Reads responses from source_context.extra_data["responses"].
        In production, responses would be generated via LLM; for testing,
        pre-computed responses are passed through extra_data.
        """
        responses: list[str] = source_context.extra_data.get("responses", [])

        if len(responses) < 3:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "Need >= 3 responses"},
            )

        embedder = self._get_embedder()
        vecs = np.array(embedder.embed_batch(responses))

        # Compute centroid and distances
        centroid = vecs.mean(axis=0)
        distances = np.array([np.linalg.norm(v - centroid) for v in vecs])

        mean_dist = float(distances.mean())
        max_dist = float(distances.max())

        # Compute pairwise cosine similarities
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        normalized = vecs / (norms + 1e-10)
        sim_matrix = np.dot(normalized, normalized.T)
        triu_indices = np.triu_indices(len(vecs), k=1)
        pairwise_sims = sim_matrix[triu_indices]
        mean_sim = float(pairwise_sims.mean())
        min_sim = float(pairwise_sims.min())

        score = max(0.0, min(1.0, mean_sim))
        passed = mean_sim >= 0.5

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details={
                "centroid_radius": round(mean_dist, 4),
                "max_distance": round(max_dist, 4),
                "mean_pairwise_similarity": round(mean_sim, 4),
                "min_pairwise_similarity": round(min_sim, 4),
                "n_responses": len(responses),
            },
        )
