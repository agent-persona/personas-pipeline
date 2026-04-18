from __future__ import annotations
import re
from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import Persona, EvalResult
from evaluation.testing.source_context import SourceContext

# Patterns that look filled but are semantically empty
_FILLER_RE = re.compile(
    r"^\s*(n/?a|not specified|unknown|tbd|to be determined|none|placeholder|lorem|example)\s*$",
    re.IGNORECASE,
)

_REQUIRED_STR_FIELDS = [
    "name", "occupation", "industry", "location", "education",
    "lifestyle", "bio",
]
_REQUIRED_LIST_FIELDS = [
    "goals", "pain_points", "values", "knowledge_domains",
    "personality_traits", "behaviors",
]
_MIN_BIO_LENGTH = 50


def _is_filler(value: str) -> bool:
    return bool(_FILLER_RE.match(value))


class CompletenessScorer(BaseScorer):
    dimension_id = "D2"
    dimension_name = "Completeness"
    tier = 1

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        errors: list[str] = []
        total_checks = 0

        for field in _REQUIRED_STR_FIELDS:
            total_checks += 1
            val = getattr(persona, field, "")
            if not val or not val.strip():
                errors.append(f"{field} is empty")
            elif _is_filler(val):
                errors.append(f"{field} contains placeholder value: '{val}'")
            elif field == "bio" and len(val) < _MIN_BIO_LENGTH:
                errors.append(f"bio is too short ({len(val)} chars, min {_MIN_BIO_LENGTH})")

        for field in _REQUIRED_LIST_FIELDS:
            total_checks += 1
            val = getattr(persona, field, [])
            if not val:
                errors.append(f"{field} list is empty")
            else:
                filler_items = [v for v in val if isinstance(v, str) and _is_filler(v)]
                if filler_items:
                    errors.append(f"{field} contains filler items: {filler_items}")

        passed = len(errors) == 0
        score = max(0.0, 1.0 - len(errors) / total_checks)
        return self._result(persona, passed=passed, score=score, errors=errors)
