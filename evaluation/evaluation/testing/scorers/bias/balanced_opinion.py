"""D47 Balanced-Opinion Inflation — false-balance bias on opinionated topics.

Trustworthiness: MEDIUM-HIGH (structural pattern matching + explicit persona_opinion signal).
Method: Detect contrastive connectors in responses when persona has a stated strong opinion.

Evidence: LLMs default to measured "both sides" framing even when the persona has
strong stated values or opinions. A persona who lists "meritocracy" as a core value
should not hedge when asked about merit-based promotion — but LLMs produce balanced
responses by default across all persona types.
"""

from __future__ import annotations

import re

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import EvalResult, Persona
from evaluation.testing.source_context import SourceContext

# ---------------------------------------------------------------------------
# Balance detection patterns
# ---------------------------------------------------------------------------

# DOTALL is intentionally omitted from multi-span patterns so `.` does not cross
# newlines. Balance connectors must appear within the same sentence; cross-paragraph
# matches are false positives (e.g. "However, X.\n\nWe also considered Y." is not
# a genuine both-sides framing of a single position).
BALANCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bon the other hand\b", re.IGNORECASE),
    re.compile(r"\bhowever.{0,60}also\b", re.IGNORECASE),
    re.compile(r"\bwhile.{0,60}(also|yet)\b", re.IGNORECASE),
    re.compile(r"\bboth sides\b", re.IGNORECASE),
    re.compile(r"\bpros and cons\b", re.IGNORECASE),
    re.compile(r"\badvantages and disadvantages\b", re.IGNORECASE),
    re.compile(r"\bit depends\b", re.IGNORECASE),
    re.compile(r"\bon one hand.{0,100}on the other\b", re.IGNORECASE),
    re.compile(r"\bthat said.{0,60}also\b", re.IGNORECASE),
    re.compile(r"\balthough.{0,60}however\b", re.IGNORECASE),
]

# Minimum number of opinionated items to run the scorer (skip if fewer)
MIN_OPINIONATED_ITEMS = 2


def _is_balanced(response: str) -> bool:
    """Return True if the response contains a contrastive balance pattern."""
    for pattern in BALANCE_PATTERNS:
        if pattern.search(response):
            return True
    return False


class BalancedOpinionScorer(BaseScorer):
    """Evaluates whether persona over-applies diplomatic both-sides framing on opinionated topics."""

    dimension_id = "D47"
    dimension_name = "Balanced-Opinion Inflation"
    tier = 4
    requires_set = False

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        opinion_responses: list[dict] = source_context.extra_data.get("opinion_responses") or []

        if not opinion_responses:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No opinion_responses provided"},
            )

        # Only count items where persona has a stated strong opinion
        opinionated_items = [
            r for r in opinion_responses
            if isinstance(r, dict) and r.get("persona_opinion")
        ]

        if len(opinionated_items) < MIN_OPINIONATED_ITEMS:
            return self._result(
                persona, passed=True, score=1.0,
                details={
                    "skipped": True,
                    "reason": f"Fewer than {MIN_OPINIONATED_ITEMS} opinionated items",
                    "opinionated_items": len(opinionated_items),
                },
            )

        inflated_count = sum(
            1 for r in opinionated_items
            if _is_balanced(r.get("response", ""))
        )
        inflation_rate = inflated_count / len(opinionated_items)
        score = max(0.0, 1.0 - inflation_rate)
        passed = inflation_rate < 0.5

        return self._result(
            persona,
            passed=passed,
            score=round(score, 4),
            details={
                "inflation_rate": round(inflation_rate, 4),
                "inflated_count": inflated_count,
                "opinionated_items": len(opinionated_items),
                "total_items": len(opinion_responses),
            },
        )
