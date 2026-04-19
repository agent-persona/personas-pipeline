from __future__ import annotations
from abc import ABC, abstractmethod
from persona_eval.schemas import Persona, EvalResult
from persona_eval.source_context import SourceContext


class BaseScorer(ABC):
    dimension_id: str
    dimension_name: str
    tier: int
    requires_set: bool = False

    @abstractmethod
    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        ...

    def score_set(
        self, personas: list[Persona], source_contexts: list[SourceContext]
    ) -> list[EvalResult]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement score_set(). "
            "Override this method for set-level dimensions."
        )

    def _result(
        self,
        persona: Persona,
        passed: bool,
        score: float,
        details: dict | None = None,
        errors: list[str] | None = None,
    ) -> EvalResult:
        return EvalResult(
            dimension_id=self.dimension_id,
            dimension_name=self.dimension_name,
            persona_id=persona.id,
            passed=passed,
            score=score,
            details=details or {},
            errors=errors or [],
        )
