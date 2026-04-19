"""D41 Degradation Detection — statistical process control for quality drift.

Trustworthiness: HIGH (standard monitoring practice).
Method: Compare current score to historical baseline using z-score.
Detect drift > 1 sigma below mean. Handles zero-variance baselines.
Expects source_context.extra_data:
    "historical_scores": list[float] (past quality scores)
    "current_score": float (latest quality score)
"""

from __future__ import annotations

import math

from persona_eval.scorer import BaseScorer
from persona_eval.schemas import EvalResult, Persona
from persona_eval.source_context import SourceContext

# Minimum historical samples for SPC
MIN_HISTORY = 5
# Z-score threshold: drift below -1 sigma = alert
DRIFT_THRESHOLD = -1.0
# Minimum std dev to avoid division by zero
MIN_STD = 0.01


class DegradationDetectionScorer(BaseScorer):
    """Detects quality degradation via statistical process control."""

    dimension_id = "D41"
    dimension_name = "Degradation Detection"
    tier = 6
    requires_set = False

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        historical: list[float] = source_context.extra_data.get("historical_scores", [])
        current = source_context.extra_data.get("current_score")

        if current is None or len(historical) < MIN_HISTORY:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": f"Need current_score and >= {MIN_HISTORY} historical scores"},
            )

        mean = sum(historical) / len(historical)
        variance = sum((x - mean) ** 2 for x in historical) / len(historical)
        std = math.sqrt(variance)

        # Use MIN_STD for zero-variance baselines
        effective_std = max(std, MIN_STD)
        z_score = (current - mean) / effective_std

        drift_detected = z_score < DRIFT_THRESHOLD

        # Score: map z-score to [0, 1]. z=0 → 1.0, z=-2 → 0.0
        score = max(0.0, min(1.0, 1.0 + z_score / 2))

        passed = not drift_detected

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details={
                "z_score": round(z_score, 4),
                "historical_mean": round(mean, 4),
                "historical_std": round(std, 4),
                "current_score": round(current, 4),
                "drift_detected": drift_detected,
                "n_historical": len(historical),
            },
        )
