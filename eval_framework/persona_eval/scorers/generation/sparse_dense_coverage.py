"""D44 Sparse vs Dense Dimension Coverage — eval coverage analysis.

Trustworthiness: MEDIUM-HIGH (methodologically sound).
Method: Build coverage matrix (attribute × conversation), compute coverage rate.
Identify sparse dimensions (covered in < 50% of conversations).
Expects source_context.extra_data["coverage_matrix"]:
    {"attribute_name": bool, ...} per conversation indicating which attributes were tested.
"""

from __future__ import annotations

from collections import defaultdict

from persona_eval.scorer import BaseScorer
from persona_eval.schemas import EvalResult, Persona
from persona_eval.source_context import SourceContext

# Fraction of conversations an attribute must appear in to not be "sparse"
SPARSE_THRESHOLD = 0.5
# Pass if overall coverage rate >= this
COVERAGE_PASS_THRESHOLD = 0.6


class SparseDenseCoverageScorer(BaseScorer):
    """Evaluates whether eval conversations cover both dense and sparse dimensions."""

    dimension_id = "D44"
    dimension_name = "Sparse vs Dense Coverage"
    tier = 7
    requires_set = True

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        return self._result(
            persona, passed=True, score=1.0,
            details={"skipped": True, "reason": "Set-level scorer — use score_set()"},
        )

    def score_set(
        self, personas: list[Persona], source_contexts: list[SourceContext]
    ) -> list[EvalResult]:
        sentinel = Persona(id="__set__", name="__set__")

        if not personas:
            return [self._result(
                sentinel, passed=True, score=1.0,
                details={"skipped": True, "reason": "No personas provided"},
            )]

        # Aggregate coverage across all conversations
        attribute_coverage: dict[str, int] = defaultdict(int)
        n_conversations = 0

        for ctx in source_contexts:
            matrix = ctx.extra_data.get("coverage_matrix")
            if not matrix:
                continue
            n_conversations += 1
            for attr, covered in matrix.items():
                if covered:
                    attribute_coverage[attr] += 1

        if n_conversations == 0 or not attribute_coverage:
            return [self._result(
                sentinel, passed=True, score=1.0,
                details={"skipped": True, "reason": "No coverage_matrix data found"},
            )]

        # Compute per-attribute coverage rate
        attribute_rates: dict[str, float] = {}
        sparse_dims: list[str] = []
        dense_dims: list[str] = []

        for attr, count in sorted(attribute_coverage.items()):
            rate = count / n_conversations
            attribute_rates[attr] = round(rate, 4)
            if rate < SPARSE_THRESHOLD:
                sparse_dims.append(attr)
            else:
                dense_dims.append(attr)

        # Overall coverage rate = mean of per-attribute rates
        overall_rate = sum(attribute_rates.values()) / len(attribute_rates)
        score = max(0.0, min(1.0, overall_rate))
        passed = overall_rate >= COVERAGE_PASS_THRESHOLD

        return [self._result(
            sentinel, passed=passed, score=round(score, 4),
            details={
                "coverage_rate": round(overall_rate, 4),
                "n_conversations": n_conversations,
                "n_attributes": len(attribute_rates),
                "sparse_dimensions": sparse_dims,
                "dense_dimensions": dense_dims,
                "attribute_rates": attribute_rates,
            },
        )]
