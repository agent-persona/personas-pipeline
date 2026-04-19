"""D35 Role Identifiability — can evaluators identify which persona is speaking?

Trustworthiness: LOW-MEDIUM (LLM judges achieve 68.8% vs human 90.8%).
Method: Accept pre-computed identification results (true_id vs predicted_id),
compute accuracy. Set-level scorer since identification requires a lineup.
Expects source_context.extra_data["identification_result"]:
    {"true_id": str, "predicted_id": str} per persona.
"""

from __future__ import annotations

from persona_eval.scorer import BaseScorer
from persona_eval.schemas import EvalResult, Persona
from persona_eval.source_context import SourceContext

# Pass if identification accuracy >= this
ACCURACY_THRESHOLD = 0.6


class RoleIdentifiabilityScorer(BaseScorer):
    """Evaluates whether personas are distinct enough to be identified from transcripts."""

    dimension_id = "D35"
    dimension_name = "Role Identifiability"
    tier = 6
    requires_set = True

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        return self._result(
            persona, passed=True, score=1.0,
            details={"skipped": True, "reason": "Set-level scorer — use score_set()"},
        )

    def score_set(
        self, personas: list[Persona], source_contexts: list[SourceContext]
    ) -> list[EvalResult]:
        results: list[dict] = []
        for ctx in source_contexts:
            id_result = ctx.extra_data.get("identification_result")
            if id_result and "true_id" in id_result and "predicted_id" in id_result:
                results.append(id_result)

        sentinel = Persona(id="__set__", name="__set__")

        if not results:
            return [self._result(
                sentinel, passed=True, score=1.0,
                details={"skipped": True, "reason": "No identification_result in extra_data"},
            )]

        correct = sum(1 for r in results if r["true_id"] == r["predicted_id"])
        total = len(results)
        accuracy = correct / total

        passed = accuracy >= ACCURACY_THRESHOLD

        return [self._result(
            sentinel, passed=passed, score=round(accuracy, 4),
            details={
                "accuracy": round(accuracy, 4),
                "correct": correct,
                "total": total,
            },
        )]
