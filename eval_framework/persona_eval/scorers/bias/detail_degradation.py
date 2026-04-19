"""D24b Persona Detail Degradation — detects accuracy loss from over-specification.

Trustworthiness: HIGH (direct accuracy measurement).
Method: Compare accuracy at increasing persona detail levels.
Evidence: Li et al. 2025 ("Promise with a Catch") — accuracy monotonically
decreases with more LLM-generated detail across ~1M personas.
"""

from __future__ import annotations

from persona_eval.scorer import BaseScorer
from persona_eval.schemas import Persona, EvalResult
from persona_eval.source_context import SourceContext

# If accuracy drops more than this from min to max detail, flag it
DEGRADATION_THRESHOLD = 0.10  # 10% absolute accuracy drop


class DetailDegradationScorer(BaseScorer):
    """Detects whether adding persona detail degrades accuracy (Li/Promise 2025)."""

    dimension_id = "D24b"
    dimension_name = "Persona Detail Degradation"
    tier = 4
    requires_set = True

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        return self._result(
            persona, passed=True, score=1.0,
            details={"skipped": True, "reason": "D24b is a set-level dimension"},
        )

    def score_set(
        self, personas: list[Persona], source_contexts: list[SourceContext]
    ) -> list[EvalResult]:
        # Collect detail_level_results from all contexts (use first non-empty)
        all_results = []
        for ctx in source_contexts:
            dlr = ctx.extra_data.get("detail_level_results", [])
            if dlr:
                all_results = dlr
                break

        if not all_results:
            return [EvalResult(
                dimension_id=self.dimension_id,
                dimension_name=self.dimension_name,
                persona_id="__set__",
                passed=True, score=1.0,
                details={"skipped": True, "reason": "No detail_level_results provided"},
            )]

        # Sort by detail level
        sorted_results = sorted(all_results, key=lambda x: x["detail_level"])
        accuracies = [r["accuracy"] for r in sorted_results]
        labels = [r.get("label", f"level_{r['detail_level']}") for r in sorted_results]

        # Check if monotonically decreasing
        is_monotonic = all(
            accuracies[i] >= accuracies[i + 1]
            for i in range(len(accuracies) - 1)
        )

        # Compute degradation rate (accuracy drop from first to last level)
        degradation = accuracies[0] - accuracies[-1] if len(accuracies) >= 2 else 0.0

        # Score: 1.0 if no degradation, lower if degradation detected
        # Clamp to [0.0, 1.0] — negative degradation (improvement) still caps at 1.0
        score = min(1.0, max(0.0, 1.0 - (degradation * 2)))  # 50% drop → score 0.0
        passed = degradation < DEGRADATION_THRESHOLD

        return [EvalResult(
            dimension_id=self.dimension_id,
            dimension_name=self.dimension_name,
            persona_id="__set__",
            passed=passed,
            score=round(score, 4),
            details={
                "is_monotonically_decreasing": is_monotonic,
                "degradation_rate": round(degradation, 4),
                "accuracies_by_level": dict(zip(labels, [round(a, 4) for a in accuracies])),
                "degradation_threshold": DEGRADATION_THRESHOLD,
            },
        )]
