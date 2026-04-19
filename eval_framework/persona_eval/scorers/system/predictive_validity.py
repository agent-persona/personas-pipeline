"""D36 Predictive Validity — do persona predictions match real human behavior?

Trustworthiness: HIGH (but requires real behavioral data).
Method: Accept predicted vs actual value pairs, compute Pearson correlation.
Detect sign flips (negative correlation = persona predicts opposite of reality).
Expects source_context.extra_data["predictions"]:
    list of {"predicted": float, "actual": float}.
"""

from __future__ import annotations

from persona_eval.scorer import BaseScorer
from persona_eval.schemas import EvalResult, Persona
from persona_eval.source_context import SourceContext
from persona_eval.stats import pearson_r

# Minimum samples for meaningful correlation
MIN_SAMPLES = 5
# Pass if correlation >= this
CORRELATION_THRESHOLD = 0.5


class PredictiveValidityScorer(BaseScorer):
    """Evaluates whether persona predictions correlate with real outcomes."""

    dimension_id = "D36"
    dimension_name = "Predictive Validity"
    tier = 6
    requires_set = False

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        predictions: list[dict] = source_context.extra_data.get("predictions", [])

        if len(predictions) < MIN_SAMPLES:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": f"Need >= {MIN_SAMPLES} predictions"},
            )

        predicted = [p["predicted"] for p in predictions]
        actual = [p["actual"] for p in predictions]

        r = pearson_r(predicted, actual)
        sign_flip = r < 0

        score = max(0.0, min(1.0, r))
        passed = r >= CORRELATION_THRESHOLD and not sign_flip

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details={
                "correlation": round(r, 4),
                "sign_flip": sign_flip,
                "n_predictions": len(predictions),
            },
        )
