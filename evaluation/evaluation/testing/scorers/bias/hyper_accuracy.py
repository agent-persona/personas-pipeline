"""D22 Hyper-Accuracy Distortion — detects unrealistically perfect factual knowledge.

Trustworthiness: HIGH (objective comparison against human baselines).
Method: Compare persona accuracy on factual questions against known human accuracy rates.
Flag if persona is significantly MORE accurate than humans on hard questions.
Expects source_context.extra_data["factual_answers"]: list of
    {"question": str, "persona_answer": str, "correct_answer": str, "human_accuracy": float}.
"""

from __future__ import annotations

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import EvalResult, Persona
from evaluation.testing.source_context import SourceContext

# If persona exceeds expected accuracy by more than this, it's distorted
ACCURACY_GAP_THRESHOLD = 0.25


class HyperAccuracyScorer(BaseScorer):
    """Evaluates whether a persona exhibits unrealistically high factual accuracy."""

    dimension_id = "D22"
    dimension_name = "Hyper-Accuracy Distortion"
    tier = 4
    requires_set = False

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        answers: list[dict] = source_context.extra_data.get("factual_answers", [])
        if not answers:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No factual_answers in extra_data"},
            )

        correct_count = 0
        weighted_expected = 0.0
        valid_count = 0

        for item in answers:
            try:
                pa = str(item["persona_answer"]).strip().lower()
                ca = str(item["correct_answer"]).strip().lower()
                ha = float(item["human_accuracy"])
            except (KeyError, TypeError, ValueError):
                continue
            valid_count += 1
            if pa == ca:
                correct_count += 1
            weighted_expected += ha

        if valid_count == 0:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No valid factual_answers entries"},
            )

        persona_accuracy = correct_count / valid_count
        expected_accuracy = weighted_expected / valid_count
        accuracy_gap = persona_accuracy - expected_accuracy

        # Score: penalize being too accurate (gap > threshold)
        if accuracy_gap > ACCURACY_GAP_THRESHOLD:
            score = max(0.0, 1.0 - (accuracy_gap - ACCURACY_GAP_THRESHOLD) * 4)
        else:
            score = 1.0

        passed = accuracy_gap <= ACCURACY_GAP_THRESHOLD

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details={
                "persona_accuracy": round(persona_accuracy, 4),
                "expected_accuracy": round(expected_accuracy, 4),
                "accuracy_gap": round(accuracy_gap, 4),
                "question_count": valid_count,
                "correct_count": correct_count,
            },
        )
