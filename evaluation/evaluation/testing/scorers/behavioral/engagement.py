"""D32-D33 Engagement — response diversity and lexical richness.

Trustworthiness: HIGH (type-token ratio is objective).
Method: Measure lexical diversity (unique words / total words) across conversation turns.
Low diversity = repetitive, disengaging responses.
Expects source_context.extra_data["conversation_turns"]: list of response strings.
"""

from __future__ import annotations

import re

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import EvalResult, Persona
from evaluation.testing.source_context import SourceContext

# Pass if lexical diversity >= this
DIVERSITY_THRESHOLD = 0.4

_WORD_PATTERN = re.compile(r"\b\w+\b")


class EngagementScorer(BaseScorer):
    """Evaluates response diversity and engagement quality."""

    dimension_id = "D32-D33"
    dimension_name = "Engagement & Diversity"
    tier = 5
    requires_set = False

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        turns: list[str] = source_context.extra_data.get("conversation_turns", [])
        if not turns:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No conversation_turns in extra_data"},
            )

        # Compute type-token ratio (lexical diversity)
        all_words: list[str] = []
        for turn in turns:
            all_words.extend(w.lower() for w in _WORD_PATTERN.findall(turn))

        if not all_words:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No words found in turns"},
            )

        unique_words = len(set(all_words))
        total_words = len(all_words)
        lexical_diversity = unique_words / total_words

        passed = lexical_diversity >= DIVERSITY_THRESHOLD

        return self._result(
            persona, passed=passed, score=round(lexical_diversity, 4),
            details={
                "lexical_diversity": round(lexical_diversity, 4),
                "unique_words": unique_words,
                "total_words": total_words,
                "turn_count": len(turns),
            },
        )
