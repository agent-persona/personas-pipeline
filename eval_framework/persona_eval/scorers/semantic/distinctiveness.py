"""D6 Distinctiveness — pairwise embedding distance + near-duplicate detection.

Trustworthiness: MEDIUM (captures surface differences).
Method: Embed persona text representations, compute pairwise cosine distances.
"""

from __future__ import annotations

import numpy as np

from persona_eval.embeddings import Embedder
from persona_eval.scorer import BaseScorer
from persona_eval.schemas import Persona, EvalResult
from persona_eval.source_context import SourceContext

DUPLICATE_THRESHOLD = 0.05  # Cosine distance < 0.05 = near-duplicate


def _persona_to_text(persona: Persona) -> str:
    """Convert a Persona to a text representation for embedding."""
    parts = []
    if persona.occupation:
        parts.append(f"occupation: {persona.occupation}")
    if persona.industry:
        parts.append(f"industry: {persona.industry}")
    if persona.education:
        parts.append(f"education: {persona.education}")
    if persona.location:
        parts.append(f"location: {persona.location}")
    if persona.personality_traits:
        parts.append(f"traits: {', '.join(persona.personality_traits)}")
    if persona.communication_style.tone:
        parts.append(f"tone: {persona.communication_style.tone}")
    if persona.lifestyle:
        parts.append(f"lifestyle: {persona.lifestyle}")
    if persona.bio:
        parts.append(f"bio: {persona.bio}")
    return " | ".join(parts)


class DistinctivenessScorer(BaseScorer):
    """Evaluates whether personas in a set are sufficiently distinct from each other."""

    dimension_id = "D6"
    dimension_name = "Distinctiveness"
    tier = 2
    requires_set = True

    def __init__(self) -> None:
        self._embedder: Embedder | None = None

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        return self._result(
            persona, passed=True, score=1.0,
            details={"skipped": True, "reason": "D6 is a set-level dimension"},
        )

    def score_set(
        self, personas: list[Persona], source_contexts: list[SourceContext]
    ) -> list[EvalResult]:
        if len(personas) < 2:
            return [EvalResult(
                dimension_id=self.dimension_id,
                dimension_name=self.dimension_name,
                persona_id="__set__",
                passed=True, score=1.0,
                details={"skipped": True, "reason": "Need >= 2 personas"},
            )]

        embedder = self._get_embedder()
        texts = [_persona_to_text(p) for p in personas]
        vecs = np.array(embedder.embed_batch(texts))

        # Compute pairwise cosine distances
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        normalized = vecs / (norms + 1e-10)
        sim_matrix = np.dot(normalized, normalized.T)
        triu_indices = np.triu_indices(len(vecs), k=1)
        pairwise_sims = sim_matrix[triu_indices]
        pairwise_dists = 1.0 - pairwise_sims

        mean_dist = float(pairwise_dists.mean())
        min_dist = float(pairwise_dists.min())

        # Flag near-duplicate pairs
        near_duplicates: list[dict] = []
        for idx, (i, j) in enumerate(zip(*triu_indices)):
            if pairwise_dists[idx] < DUPLICATE_THRESHOLD:
                near_duplicates.append({
                    "persona_a": personas[i].id,
                    "persona_b": personas[j].id,
                    "distance": round(float(pairwise_dists[idx]), 4),
                })

        score = min(1.0, mean_dist / 0.5)  # Normalize: 0.5 distance = perfect diversity
        passed = len(near_duplicates) == 0 and mean_dist >= 0.1

        return [EvalResult(
            dimension_id=self.dimension_id,
            dimension_name=self.dimension_name,
            persona_id="__set__",
            passed=passed,
            score=round(score, 4),
            details={
                "mean_pairwise_distance": round(mean_dist, 4),
                "min_pairwise_distance": round(min_dist, 4),
                "near_duplicates": near_duplicates,
                "persona_count": len(personas),
            },
        )]
