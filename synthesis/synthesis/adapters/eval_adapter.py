"""Convert PersonaV1 (synthesis output) → persona_eval.Persona (eval input).

Pure field-mapping adapter — no LLM calls. Assumes PersonaV1 already
carries psychological depth (communication_style, emotional_profile,
moral_framework).

Directional contract: PersonaV1 enforces stricter bounds than
persona_eval.Persona (e.g. `preferred_channels` min_length=1 here, unconstrained
there). That's intentional — strict producer, lenient consumer — so the adapter
never needs to pad or default; it only passes data through.

Derivations for eval-only fields that PersonaV1 doesn't carry directly:
- `expertise_level` from `communication_style.vocabulary_level`
- `behaviors` from flattened `journey_stages[*].key_actions` (dedup, order-preserved)

Remaining un-derivable fields (`habits`, `personality_traits`, `interests`,
`lifestyle`, `ethnicity`, `marital_status`, `experience_years`) stay empty/None —
inferring them from current PersonaV1 signal would be guessing, and the Tier 4
bias scorers exist specifically to catch that.
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
_AGE_OPEN_UPPER_RE = re.compile(r"^\s*(\d{1,3})\s*\+\s*$")
_GENDER_PREFIX_RE = re.compile(r"^(predominantly|mostly|mainly|primarily)\s+", re.IGNORECASE)
_GENDER_MIXED = {"mixed", "balanced", "diverse", "varied"}

_VOCAB_TO_EXPERTISE = {
    "basic": "novice",
    "intermediate": "intermediate",
    "advanced": "expert",
}


def _age_midpoint(age_range: str) -> int | None:
    m = _AGE_RANGE_RE.match(age_range)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return round((lo + hi) / 2)
    # Open-upper notation like "65+" — no upper bound, so take lower + 5
    # to preserve a demographic signal instead of dropping to None.
    m = _AGE_OPEN_UPPER_RE.match(age_range)
    if m:
        return int(m.group(1)) + 5
    return None


def _normalize_gender(gender_distribution: str) -> str:
    stripped = _GENDER_PREFIX_RE.sub("", gender_distribution).strip().lower()
    if stripped in _GENDER_MIXED or stripped == "":
        return ""
    return stripped


def persona_v1_to_eval(persona: PersonaV1, persona_id: str) -> EvalPersona:
    """Convert a PersonaV1 to a persona_eval.Persona."""
    demo = persona.demographics
    firm = persona.firmographics

    comm = persona.communication_style
    emo = persona.emotional_profile
    moral = persona.moral_framework

    # Deduplicate source ids while preserving order
    seen_ids: set[str] = set()
    source_ids: list[str] = []
    for ev in persona.source_evidence:
        for rid in ev.record_ids:
            if rid not in seen_ids:
                seen_ids.add(rid)
                source_ids.append(rid)

    # Behaviors: flatten journey_stages key_actions, dedup preserving order
    seen_actions: set[str] = set()
    behaviors: list[str] = []
    for stage in persona.journey_stages:
        for action in stage.key_actions:
            if action not in seen_actions:
                seen_actions.add(action)
                behaviors.append(action)

    expertise_level = _VOCAB_TO_EXPERTISE.get(comm.vocabulary_level.lower(), "")

    extra: dict = {
        "age_range": demo.age_range,
        "gender_distribution": demo.gender_distribution,
        "location_signals": list(demo.location_signals),
        "vocabulary": list(persona.vocabulary),
        "channels": list(persona.channels),
        "decision_triggers": list(persona.decision_triggers),
        "sample_quotes": list(persona.sample_quotes),
        "journey_stages": [js.model_dump() for js in persona.journey_stages],
        "source_evidence": [ev.model_dump() for ev in persona.source_evidence],
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
        expertise_level=expertise_level,
        behaviors=behaviors,
        goals=list(persona.goals),
        motivations=list(persona.motivations),
        pain_points=list(persona.pains),
        frustrations=list(persona.objections),
        # Mirrored from core_values — some scorers (weird_bias) read both
        # `values` and `moral_framework.core_values`; others (schema_compliance)
        # read only `values`. Populating both is required, not redundant.
        values=list(moral.core_values),
        communication_style=EvalCommunicationStyle(
            tone=comm.tone,
            formality=comm.formality,
            vocabulary_level=comm.vocabulary_level,
            preferred_channels=list(comm.preferred_channels),
        ),
        emotional_profile=EvalEmotionalProfile(
            baseline_mood=emo.baseline_mood,
            stress_triggers=list(emo.stress_triggers),
            coping_mechanisms=list(emo.coping_mechanisms),
        ),
        moral_framework=EvalMoralFramework(
            core_values=list(moral.core_values),
            ethical_stance=moral.ethical_stance,
            moral_foundations=dict(moral.moral_foundations),
        ),
        source_ids=source_ids,
        extra=extra,
    )
