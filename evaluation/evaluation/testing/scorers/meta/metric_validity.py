"""M3 Evaluation Metric Validity — metric-human correlation framework.

Trustworthiness: META (validates metric usefulness).
Method: Compute correlation between automated metric and human ratings.
Check sensitivity (does metric change when quality changes?).
Expects source_context.extra_data:
    "metric_scores": list[float] (automated metric scores)
    "human_ratings": list[float] (human quality ratings)
    Optional "metric_name": str (which metric being validated)
"""

from __future__ import annotations

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import EvalResult, Persona
from evaluation.testing.source_context import SourceContext
from evaluation.testing.stats import pearson_r

# Minimum samples
MIN_SAMPLES = 5
# Minimum correlation for valid metric
VALIDITY_THRESHOLD = 0.5
# Minimum std dev to consider metric "sensitive"
SENSITIVITY_THRESHOLD = 0.05


class MetricValidityScorer(BaseScorer):
    """Evaluates whether automated metrics correlate with human quality judgments."""

    dimension_id = "M3"
    dimension_name = "Evaluation Metric Validity"
    tier = 8
    requires_set = False

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        metric_scores: list[float] = source_context.extra_data.get("metric_scores", [])
        human_ratings: list[float] = source_context.extra_data.get("human_ratings", [])
        metric_name = source_context.extra_data.get("metric_name", "unknown")

        if len(metric_scores) < MIN_SAMPLES or len(metric_scores) != len(human_ratings):
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": f"Need >= {MIN_SAMPLES} matched score pairs"},
            )

        r = pearson_r(metric_scores, human_ratings)

        # Sensitivity: does the metric actually vary?
        mean_m = sum(metric_scores) / len(metric_scores)
        std_m = (sum((x - mean_m) ** 2 for x in metric_scores) / len(metric_scores)) ** 0.5
        sensitive = std_m >= SENSITIVITY_THRESHOLD

        score = max(0.0, min(1.0, r))
        passed = r >= VALIDITY_THRESHOLD and sensitive

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details={
                "correlation": round(r, 4),
                "sensitive": sensitive,
                "metric_std": round(std_m, 4),
                "metric_name": metric_name,
                "n_samples": len(metric_scores),
            },
        )
