"""D17 Calibration — Expected Calibration Error (ECE).

Trustworthiness: HIGH (well-established statistical methodology).
Method: Bin responses by confidence, measure accuracy per bin, compute ECE.
"""

from __future__ import annotations

from typing import Any

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import Persona, EvalResult
from evaluation.testing.source_context import SourceContext


def _compute_ece(
    confidences: list[float], accuracies: list[bool], n_bins: int = 10
) -> tuple[float, list[dict]]:
    """Compute Expected Calibration Error.

    Returns (ece_value, bin_details).
    """
    if not confidences:
        return 0.0, []

    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    bin_details: list[dict] = []
    ece = 0.0
    total = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        # First bin is inclusive on both sides, rest are (lo, hi]
        if i == 0:
            indices = [j for j, c in enumerate(confidences) if lo <= c <= hi]
        else:
            indices = [j for j, c in enumerate(confidences) if lo < c <= hi]

        bin_size = len(indices)
        if bin_size == 0:
            continue

        avg_conf = sum(confidences[j] for j in indices) / bin_size
        avg_acc = sum(1.0 for j in indices if accuracies[j]) / bin_size
        gap = abs(avg_acc - avg_conf)
        ece += (bin_size / total) * gap

        bin_details.append({
            "bin": f"({lo:.1f}, {hi:.1f}]",
            "count": bin_size,
            "avg_confidence": round(avg_conf, 4),
            "avg_accuracy": round(avg_acc, 4),
            "gap": round(gap, 4),
        })

    return ece, bin_details


class CalibrationScorer(BaseScorer):
    """Evaluates calibration of confidence-labeled predictions via ECE."""

    dimension_id = "D17"
    dimension_name = "Calibration"
    tier = 3
    requires_set = True

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        return self._result(
            persona, passed=True, score=1.0,
            details={"skipped": True, "reason": "D17 is a set-level dimension"},
        )

    def score_set(
        self,
        personas: list[Persona],
        source_contexts: list[SourceContext],
        **kwargs: Any,
    ) -> list[EvalResult]:
        """Evaluate calibration from confidence-labeled predictions.

        Args:
            kwargs: predictions — list of {"confidence": float, "correct": bool}
        """
        predictions: list[dict] = kwargs.get("predictions", [])
        # Fallback: collect calibration predictions from source_contexts extra_data
        if not predictions:
            for ctx in source_contexts:
                predictions.extend(ctx.extra_data.get("calibration_predictions", []))

        if len(predictions) < 4:
            return [EvalResult(
                dimension_id=self.dimension_id,
                dimension_name=self.dimension_name,
                persona_id="__set__",
                passed=True, score=1.0,
                details={"skipped": True, "reason": "Need >= 4 predictions"},
            )]

        confidences = [p["confidence"] for p in predictions]
        accuracies = [p["correct"] for p in predictions]

        ece, bin_details = _compute_ece(confidences, accuracies)
        # ECE of 0 = perfect calibration, 1 = worst
        score = max(0.0, 1.0 - ece)
        passed = ece < 0.15

        return [EvalResult(
            dimension_id=self.dimension_id,
            dimension_name=self.dimension_name,
            persona_id="__set__",
            passed=passed,
            score=round(score, 4),
            details={
                "ece": round(ece, 4),
                "bins": bin_details,
                "n_predictions": len(predictions),
            },
        )]
