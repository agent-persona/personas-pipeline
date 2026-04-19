"""M1 LLM-as-Judge Reliability — human annotation baseline framework.

Trustworthiness: META (validates other scorers).
Method: Compare LLM judge scores to human annotation baselines.
Compute Pearson correlation, classify trust level.
Expects source_context.extra_data:
    "judge_scores": list[float] (LLM judge scores)
    "human_scores": list[float] (human annotation scores)
    Optional "dimension_name": str (which dimension being validated)
"""

from __future__ import annotations

from persona_eval.scorer import BaseScorer
from persona_eval.schemas import EvalResult, Persona
from persona_eval.source_context import SourceContext
from persona_eval.stats import pearson_r

# Minimum samples for meaningful correlation
MIN_SAMPLES = 5


class JudgeReliabilityScorer(BaseScorer):
    """Evaluates how well LLM judge scores correlate with human annotations."""

    dimension_id = "M1"
    dimension_name = "LLM-as-Judge Reliability"
    tier = 8
    requires_set = False

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        judge_scores: list[float] = source_context.extra_data.get("judge_scores", [])
        human_scores: list[float] = source_context.extra_data.get("human_scores", [])
        dim_name = source_context.extra_data.get("dimension_name", "unknown")

        if len(judge_scores) < MIN_SAMPLES or len(judge_scores) != len(human_scores):
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": f"Need >= {MIN_SAMPLES} matched score pairs"},
            )

        r = pearson_r(judge_scores, human_scores)

        # Trust classification
        if r >= 0.8:
            trust = "high"
        elif r >= 0.6:
            trust = "medium"
        elif r >= 0.4:
            trust = "low"
        else:
            trust = "unreliable"

        passed = trust in ("high", "medium")
        score = max(0.0, min(1.0, r))

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details={
                "correlation": round(r, 4),
                "trust_level": trust,
                "dimension": dim_name,
                "n_samples": len(judge_scores),
            },
        )
