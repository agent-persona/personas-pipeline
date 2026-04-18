"""M2 Judge Gaming Prevention — adversarial judge testing framework.

Trustworthiness: META (validates judge robustness).
Method: Submit known-bad personas, verify the judge catches them.
Track detection rate and false positive rate.
Optional cross-family agreement score.
Expects source_context.extra_data:
    "adversarial_tests": list[{"is_bad": bool, "judge_caught": bool}]
    Optional "cross_family_agreement": float (agreement between different model families)
"""

from __future__ import annotations

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import EvalResult, Persona
from evaluation.testing.source_context import SourceContext

# Pass if detection rate >= this
DETECTION_THRESHOLD = 0.7
# Penalty weight for false positives in score calculation
FP_PENALTY_WEIGHT = 0.3


class JudgeGamingPreventionScorer(BaseScorer):
    """Evaluates whether judges can detect known-bad personas and resist gaming."""

    dimension_id = "M2"
    dimension_name = "Judge Gaming Prevention"
    tier = 8
    requires_set = False

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        tests: list[dict] = source_context.extra_data.get("adversarial_tests", [])

        if not tests:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No adversarial_tests in extra_data"},
            )

        # Compute detection rate (true positives / total bad)
        bad_tests = [t for t in tests if t["is_bad"]]
        good_tests = [t for t in tests if not t["is_bad"]]

        if bad_tests:
            detected = sum(1 for t in bad_tests if t["judge_caught"])
            detection_rate = detected / len(bad_tests)
        else:
            detection_rate = 1.0

        # Compute false positive rate (good marked as bad)
        if good_tests:
            false_positives = sum(1 for t in good_tests if t["judge_caught"])
            false_positive_rate = false_positives / len(good_tests)
        else:
            false_positive_rate = 0.0

        # Score: weight detection rate heavily, penalize false positives
        score = max(0.0, min(1.0, detection_rate - false_positive_rate * FP_PENALTY_WEIGHT))
        passed = detection_rate >= DETECTION_THRESHOLD

        details: dict = {
            "detection_rate": round(detection_rate, 4),
            "false_positive_rate": round(false_positive_rate, 4),
            "n_bad": len(bad_tests),
            "n_good": len(good_tests),
        }

        # Include cross-family agreement if provided
        cross_family = source_context.extra_data.get("cross_family_agreement")
        if cross_family is not None:
            details["cross_family_agreement"] = cross_family

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details=details,
        )
