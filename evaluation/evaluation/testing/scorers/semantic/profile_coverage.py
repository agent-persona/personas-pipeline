"""D11 Profile Coverage — attribute expression tracking.

Trustworthiness: MEDIUM (depends on how well mentions are detected).
Method: Check which persona attributes appear in conversation turns.
Coverage ratio = attributes_mentioned / total_checkable_attributes.
"""

from __future__ import annotations

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import Persona, EvalResult
from evaluation.testing.source_context import SourceContext

# Persona fields to check for coverage in conversation text
COVERAGE_FIELDS = [
    ("occupation", lambda p: p.occupation),
    ("industry", lambda p: p.industry),
    ("education", lambda p: p.education),
    ("location", lambda p: p.location),
    ("lifestyle", lambda p: p.lifestyle),
    ("comm.tone", lambda p: p.communication_style.tone),
]


class ProfileCoverageScorer(BaseScorer):
    """Evaluates what fraction of persona attributes are expressed in conversation."""

    dimension_id = "D11"
    dimension_name = "Profile Coverage"
    tier = 2

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        """Check which persona attributes appear in conversation turns.

        Reads from source_context.extra_data["conversation"]: a single text string
        representing all conversation turns concatenated, or the conversation_transcript.
        """
        conversation = source_context.extra_data.get("conversation", "")
        if not conversation and source_context.conversation_transcript:
            conversation = " ".join(
                turn.get("content", "") for turn in source_context.conversation_transcript
            )

        if not conversation.strip():
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No conversation data provided"},
            )

        conversation_lower = conversation.lower()

        checkable = 0
        mentioned = 0
        coverage_detail: dict[str, bool] = {}

        for field_name, extractor in COVERAGE_FIELDS:
            value = extractor(persona)
            if not value or not value.strip():
                continue
            checkable += 1
            is_mentioned = value.lower() in conversation_lower
            coverage_detail[field_name] = is_mentioned
            if is_mentioned:
                mentioned += 1

        if checkable == 0:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No checkable attributes"},
            )

        score = mentioned / checkable
        passed = score >= 0.4

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details={
                "coverage": coverage_detail,
                "mentioned": mentioned,
                "checkable": checkable,
            },
        )
