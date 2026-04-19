"""D31 Recovery Behavior — post-perturbation consistency measurement.

Trustworthiness: MEDIUM-HIGH (embedding centroid comparison is objective).
Method: Compare baseline response embeddings to recovery response embeddings.
High similarity = good recovery to pre-perturbation behavior.
Expects source_context.extra_data["recovery_phases"]:
    {"baseline": [str], "perturbation": [str], "recovery": [str]}.
"""

from __future__ import annotations

from persona_eval.embeddings import Embedder
from persona_eval.scorer import BaseScorer
from persona_eval.schemas import EvalResult, Persona
from persona_eval.source_context import SourceContext

# Pass if recovery similarity >= this
RECOVERY_THRESHOLD = 0.5


class RecoveryBehaviorScorer(BaseScorer):
    """Evaluates whether a persona recovers to baseline behavior after perturbation."""

    dimension_id = "D31"
    dimension_name = "Recovery Behavior"
    tier = 5
    requires_set = False

    def __init__(self) -> None:
        self._embedder: Embedder | None = None

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        phases: dict = source_context.extra_data.get("recovery_phases", {})
        baseline = phases.get("baseline", [])
        recovery = phases.get("recovery", [])

        if not baseline or not recovery:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "Missing baseline or recovery phases"},
            )

        embedder = self._get_embedder()

        # Compute centroids for baseline and recovery
        base_vecs = embedder.embed_batch(baseline)
        recv_vecs = embedder.embed_batch(recovery)

        base_centroid = [sum(v[i] for v in base_vecs) / len(base_vecs) for i in range(len(base_vecs[0]))]
        recv_centroid = [sum(v[i] for v in recv_vecs) / len(recv_vecs) for i in range(len(recv_vecs[0]))]

        similarity = Embedder.vector_similarity(base_centroid, recv_centroid)
        passed = similarity >= RECOVERY_THRESHOLD

        clamped = max(0.0, min(1.0, round(similarity, 4)))
        return self._result(
            persona, passed=passed, score=clamped,
            details={
                "recovery_similarity": round(similarity, 4),
                "baseline_count": len(baseline),
                "recovery_count": len(recovery),
            },
        )
