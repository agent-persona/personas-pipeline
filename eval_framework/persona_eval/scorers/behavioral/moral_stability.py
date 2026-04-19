"""D27 Moral Stability — consistency of moral positions across repeated questions.

Trustworthiness: MEDIUM-HIGH (embedding similarity is objective).
Method: Group responses by moral foundation, compute within-group embedding similarity.
High similarity within a foundation = stable moral position.
Expects source_context.extra_data["moral_responses"]:
    list of {"question": str, "response": str, "foundation": str}.
"""

from __future__ import annotations

from collections import defaultdict

from persona_eval.embeddings import Embedder
from persona_eval.scorer import BaseScorer
from persona_eval.schemas import EvalResult, Persona
from persona_eval.source_context import SourceContext

# Pass if mean within-foundation consistency >= this
CONSISTENCY_THRESHOLD = 0.5


class MoralStabilityScorer(BaseScorer):
    """Evaluates consistency of moral positions across repeated questions."""

    dimension_id = "D27"
    dimension_name = "Moral Stability"
    tier = 5
    requires_set = False

    def __init__(self) -> None:
        self._embedder: Embedder | None = None

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        responses: list[dict] = source_context.extra_data.get("moral_responses", [])
        if not responses:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No moral_responses in extra_data"},
            )

        # Group responses by foundation
        by_foundation: dict[str, list[str]] = defaultdict(list)
        for item in responses:
            foundation = item.get("foundation", "general")
            by_foundation[foundation].append(item["response"])

        embedder = self._get_embedder()

        # Compute within-group pairwise similarity
        foundation_scores: dict[str, float] = {}
        for foundation, texts in by_foundation.items():
            if len(texts) < 2:
                continue
            vecs = embedder.embed_batch(texts)
            sims: list[float] = []
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    sims.append(Embedder.vector_similarity(vecs[i], vecs[j]))
            foundation_scores[foundation] = round(sum(sims) / len(sims), 4) if sims else 1.0

        if not foundation_scores:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "Not enough responses per foundation"},
            )

        mean_consistency = sum(foundation_scores.values()) / len(foundation_scores)
        passed = mean_consistency >= CONSISTENCY_THRESHOLD

        clamped = max(0.0, min(1.0, round(mean_consistency, 4)))
        return self._result(
            persona, passed=passed, score=clamped,
            details={
                "mean_consistency": round(mean_consistency, 4),
                "foundation_scores": foundation_scores,
                "response_count": len(responses),
            },
        )
