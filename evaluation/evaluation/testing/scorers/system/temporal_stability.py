"""D37 Temporal Stability — does the pipeline produce consistent personas over time?

Trustworthiness: HIGH (temporal comparison is straightforward).
Method: Compare baseline text embedding to current text embedding via cosine similarity.
Optional PSI (Population Stability Index) on attribute distributions.
Expects source_context.extra_data:
    "baseline_text": str (previous persona text)
    "current_text": str (current persona text)
    Optional "baseline_distribution": list[float]
    Optional "current_distribution": list[float]
"""

from __future__ import annotations

import math

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import EvalResult, Persona
from evaluation.testing.source_context import SourceContext

# Pass if semantic similarity >= this
STABILITY_THRESHOLD = 0.7
# PSI > this triggers alert
PSI_ALERT_THRESHOLD = 0.2


def _compute_psi(baseline: list[float], current: list[float]) -> float:
    """Population Stability Index between two distributions.

    PSI < 0.1 = no significant shift
    PSI 0.1-0.2 = moderate shift
    PSI > 0.2 = significant shift
    """
    eps = 1e-6
    psi = 0.0
    for b, c in zip(baseline, current):
        b = max(b, eps)
        c = max(c, eps)
        psi += (c - b) * math.log(c / b)
    return psi


class TemporalStabilityScorer(BaseScorer):
    """Evaluates whether persona outputs remain stable over time."""

    dimension_id = "D37"
    dimension_name = "Temporal Stability"
    tier = 6
    requires_set = False

    def __init__(self) -> None:
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from evaluation.testing.embeddings import Embedder
            self._embedder = Embedder()
        return self._embedder

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        baseline_text = source_context.extra_data.get("baseline_text")
        current_text = source_context.extra_data.get("current_text")

        if not baseline_text or not current_text:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "Missing baseline_text or current_text"},
            )

        embedder = self._get_embedder()
        similarity = embedder.similarity(baseline_text, current_text)
        clamped = max(0.0, min(1.0, similarity))

        details: dict = {
            "semantic_similarity": round(similarity, 4),
        }

        # Optional PSI computation
        baseline_dist = source_context.extra_data.get("baseline_distribution")
        current_dist = source_context.extra_data.get("current_distribution")
        if baseline_dist and current_dist and len(baseline_dist) == len(current_dist):
            psi = _compute_psi(baseline_dist, current_dist)
            details["psi"] = round(psi, 4)
            details["psi_alert"] = psi > PSI_ALERT_THRESHOLD

        passed = clamped >= STABILITY_THRESHOLD

        return self._result(
            persona, passed=passed, score=round(clamped, 4),
            details=details,
        )
