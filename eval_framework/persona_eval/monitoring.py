"""Production monitoring — drift detection on eval results."""

from __future__ import annotations

import math

from persona_eval.alerting import SlackAlerter
from persona_eval.schemas import EvalResult

# Minimum historical samples for drift detection
MIN_HISTORY = 3
# Z-score below this triggers drift alert
DRIFT_Z_THRESHOLD = -1.0
# Minimum std dev to avoid division by zero
MIN_STD = 0.01


class ProductionMonitor:
    """Checks for quality drift by comparing current scores to historical baselines."""

    def __init__(self, alerter: SlackAlerter | None = None, drift_threshold: float = DRIFT_Z_THRESHOLD):
        self.alerter = alerter
        self.drift_threshold = drift_threshold

    def check_drift(
        self,
        dimension_id: str,
        history: list[EvalResult],
        current: EvalResult,
    ) -> dict:
        """Check if current score drifts from historical baseline.

        Args:
            dimension_id: Which dimension to check
            history: Historical EvalResults for this dimension
            current: Latest EvalResult to compare
        """
        if len(history) < MIN_HISTORY:
            return {"status": "insufficient_data", "history_length": len(history)}

        scores = [r.score for r in history]
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        std = math.sqrt(variance)
        effective_std = max(std, MIN_STD)

        z_score = (current.score - mean) / effective_std
        drift_detected = z_score < self.drift_threshold

        if drift_detected and self.alerter:
            self.alerter.alert_regression(
                current.persona_id,
                {dimension_id: round(abs(z_score), 2)},
            )

        return {
            "status": "drift_detected" if drift_detected else "stable",
            "z_score": round(z_score, 4),
            "historical_mean": round(mean, 4),
            "historical_std": round(std, 4),
            "effective_std": round(effective_std, 4),
            "current_score": current.score,
        }
