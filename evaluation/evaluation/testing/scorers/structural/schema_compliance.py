from __future__ import annotations
from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import Persona, EvalResult
from evaluation.testing.source_context import SourceContext


class SchemaComplianceScorer(BaseScorer):
    dimension_id = "D1"
    dimension_name = "Schema Compliance"
    tier = 1

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        errors: list[str] = []

        # Validate required string fields are actually strings
        if not isinstance(persona.id, str) or not persona.id:
            errors.append("id must be a non-empty string")
        if not isinstance(persona.name, str) or not persona.name:
            errors.append("name must be a non-empty string")

        # Validate age range if provided
        if persona.age is not None and not (0 < persona.age < 130):
            errors.append(f"age={persona.age} is outside plausible range 1-129")

        # Validate experience_years is non-negative if provided
        if persona.experience_years is not None and persona.experience_years < 0:
            errors.append("experience_years must be >= 0")

        # Validate list fields are actually lists
        for field in ("goals", "pain_points", "values", "knowledge_domains", "behaviors"):
            val = getattr(persona, field)
            if not isinstance(val, list):
                errors.append(f"{field} must be a list, got {type(val).__name__}")

        passed = len(errors) == 0
        score = 1.0 if passed else max(0.0, 1.0 - len(errors) * 0.2)
        return self._result(persona, passed=passed, score=score, errors=errors)
