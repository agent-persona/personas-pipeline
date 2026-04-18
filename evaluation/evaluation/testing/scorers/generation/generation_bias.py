"""D42 Generation Bias Amplification — quality degradation with LLM involvement.

Trustworthiness: HIGH (well-established ablation methodology).
Method: Accept quality scores at each LLM involvement level, measure degradation.
Levels: meta → objective_tabular → subjective_tabular → descriptive.
More LLM involvement should monotonically decrease quality.
Expects source_context.extra_data["ablation_scores"]:
    {"level_name": quality_score, ...} ordered from least to most LLM involvement.
"""

from __future__ import annotations

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import EvalResult, Persona
from evaluation.testing.source_context import SourceContext

# Expected order of increasing LLM involvement
INVOLVEMENT_ORDER = ["meta", "objective_tabular", "subjective_tabular", "descriptive"]
# Max acceptable total degradation (highest - lowest score)
MAX_DEGRADATION = 0.50


class GenerationBiasAmplificationScorer(BaseScorer):
    """Evaluates quality degradation curve across LLM involvement levels."""

    dimension_id = "D42"
    dimension_name = "Generation Bias Amplification"
    tier = 7
    requires_set = False

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        ablation: dict[str, float] = source_context.extra_data.get("ablation_scores", {})

        if len(ablation) < 2:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "Need >= 2 ablation levels"},
            )

        # Order scores by involvement level
        ordered_scores: list[tuple[str, float]] = []
        for level in INVOLVEMENT_ORDER:
            if level in ablation:
                ordered_scores.append((level, ablation[level]))
        # Include any custom levels not in the standard order
        for level, score_val in ablation.items():
            if level not in INVOLVEMENT_ORDER:
                ordered_scores.append((level, score_val))

        if len(ordered_scores) < 2:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "Need >= 2 ablation levels"},
            )

        scores = [s for _, s in ordered_scores]
        total_degradation = max(0.0, scores[0] - scores[-1])

        # Check monotonicity: each step should decrease or stay flat
        monotonic = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

        # Score: average quality across levels, penalized by degradation
        avg_quality = sum(scores) / len(scores)
        degradation_penalty = max(0.0, total_degradation - MAX_DEGRADATION) * 0.5
        score = max(0.0, min(1.0, avg_quality - degradation_penalty))

        passed = total_degradation <= MAX_DEGRADATION and monotonic

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details={
                "total_degradation": round(total_degradation, 4),
                "monotonic_degradation": monotonic,
                "level_scores": dict(ordered_scores),
                "avg_quality": round(avg_quality, 4),
            },
        )
