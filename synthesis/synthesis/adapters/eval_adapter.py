"""Convert PersonaV1 (synthesis output) → persona_eval.Persona (eval input).

Pure field-mapping adapter — no LLM calls. Assumes PersonaV1 already
carries psychological depth (communication_style, emotional_profile,
moral_framework).
"""
from __future__ import annotations

import re

from persona_eval.schemas import (
    CommunicationStyle as EvalCommunicationStyle,
    EmotionalProfile as EvalEmotionalProfile,
    MoralFramework as EvalMoralFramework,
    Persona as EvalPersona,
)

from synthesis.models.persona import PersonaV1

_AGE_RANGE_RE = re.compile(r"^\s*(\d{1,3})\s*[-–]\s*(\d{1,3})\s*$")
_GENDER_PREFIX_RE = re.compile(r"^(predominantly|mostly|mainly|primarily)\s+", re.IGNORECASE)
_GENDER_MIXED = {"mixed", "balanced", "diverse", "varied"}


def _age_midpoint(age_range: str) -> int | None:
    m = _AGE_RANGE_RE.match(age_range)
    if not m:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    return round((lo + hi) / 2)


def _normalize_gender(gender_distribution: str) -> str:
    stripped = _GENDER_PREFIX_RE.sub("", gender_distribution).strip().lower()
    if stripped in _GENDER_MIXED or stripped == "":
        return ""
    return stripped


def persona_v1_to_eval(persona: PersonaV1, persona_id: str) -> EvalPersona:
    """Convert a PersonaV1 to a persona_eval.Persona."""
    demo = persona.demographics
    firm = persona.firmographics

    extra: dict = {
        "age_range": demo.age_range,
        "gender_distribution": demo.gender_distribution,
        "location_signals": list(demo.location_signals),
    }
    if firm.company_size is not None:
        extra["company_size"] = firm.company_size

    return EvalPersona(
        id=persona_id,
        name=persona.name,
        bio=persona.summary,
        age=_age_midpoint(demo.age_range),
        gender=_normalize_gender(demo.gender_distribution),
        location=demo.location_signals[0] if demo.location_signals else "",
        education=demo.education_level or "",
        income_bracket=demo.income_bracket or "",
        occupation=firm.role_titles[0] if firm.role_titles else "",
        industry=firm.industry or "",
        knowledge_domains=list(firm.tech_stack_signals),
        extra=extra,
    )
