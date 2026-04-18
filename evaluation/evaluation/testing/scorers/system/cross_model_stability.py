"""D38 Cross-Model Stability — quality consistency across different LLM models.

Trustworthiness: HIGH (objective score comparison).
Method: Accept per-model dimension scores, compute cross-model variance.
Detect regressions where a model drops significantly on any dimension.
Expects source_context.extra_data["model_scores"]:
    {"model_name": {"dimension_id": score, ...}, ...}
"""

from __future__ import annotations

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import EvalResult, Persona
from evaluation.testing.source_context import SourceContext

# Max acceptable regression (score drop) between models
MAX_REGRESSION = 0.15


class CrossModelStabilityScorer(BaseScorer):
    """Evaluates whether persona quality is stable across different LLM models."""

    dimension_id = "D38"
    dimension_name = "Cross-Model Stability"
    tier = 6
    requires_set = False

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        model_scores: dict[str, dict[str, float]] = source_context.extra_data.get("model_scores", {})

        if len(model_scores) < 2:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "Need scores from >= 2 models"},
            )

        # Gather all dimensions across models
        all_dims: set[str] = set()
        for scores in model_scores.values():
            all_dims.update(scores.keys())

        # Compute per-dimension range (max - min across models)
        dimension_ranges: dict[str, float] = {}
        for dim in sorted(all_dims):
            values = [scores[dim] for scores in model_scores.values() if dim in scores]
            if len(values) >= 2:
                dimension_ranges[dim] = round(max(values) - min(values), 4)

        if not dimension_ranges:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No overlapping dimensions across models"},
            )

        max_range_val = max(dimension_ranges.values())
        mean_range_val = sum(dimension_ranges.values()) / len(dimension_ranges)

        # Score: 1.0 - mean_range (lower range = more stable)
        score = max(0.0, min(1.0, 1.0 - mean_range_val))
        passed = max_range_val <= MAX_REGRESSION

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details={
                "max_regression": round(max_range_val, 4),
                "mean_range": round(mean_range_val, 4),
                "dimension_ranges": dimension_ranges,
                "n_models": len(model_scores),
                "n_dimensions": len(dimension_ranges),
            },
        )
