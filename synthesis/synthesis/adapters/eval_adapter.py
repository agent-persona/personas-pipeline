"""Convert PersonaV1 (synthesis output) → persona_eval.Persona (eval input).

Pure field-mapping adapter — no LLM calls. Assumes PersonaV1 already
carries psychological depth (communication_style, emotional_profile,
moral_framework).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from persona_eval.schemas import Persona as EvalPersona
    from synthesis.models.persona import PersonaV1


def persona_v1_to_eval(persona: "PersonaV1", persona_id: str) -> "EvalPersona":
    raise NotImplementedError
