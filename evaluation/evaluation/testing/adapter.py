"""Adapter: synthesis.models.persona.PersonaV1 -> testing.schemas.Persona.

The pipeline produces ``PersonaV1`` (see ``synthesis/synthesis/models/persona.py``).
The scorers want a flatter shape with fields like ``occupation``, ``age``,
``pain_points``. This module bridges the two without duplicating schemas in the
pipeline. Keep the mapping conservative: only fill a scorer field when the
pipeline has direct evidence for it.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from evaluation.testing.schemas import (
    CommunicationStyle as TestingCommunicationStyle,
    EmotionalProfile as TestingEmotionalProfile,
    MoralFramework as TestingMoralFramework,
    Persona as TestingPersona,
)
from evaluation.testing.source_context import SourceContext


_AGE_RANGE_RE = re.compile(r"(\d{1,3})\s*(?:[-–to]+)\s*(\d{1,3})")


def _midpoint_age(age_range: str | None) -> int | None:
    if not age_range:
        return None
    m = _AGE_RANGE_RE.search(age_range)
    if not m:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo <= 0 or hi <= 0 or hi < lo:
        return None
    return (lo + hi) // 2


def _stable_id(name: str, summary: str) -> str:
    h = hashlib.sha1(f"{name}|{summary}".encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "persona"
    return f"{slug}-{h}"


def persona_v1_to_testing(persona_v1: Any) -> TestingPersona:
    """Translate a PersonaV1 (or dict with the same shape) to testing.Persona.

    Accepts either the pydantic model or a plain dict (from ``persona_*.json``).
    """
    d = persona_v1.model_dump() if hasattr(persona_v1, "model_dump") else dict(persona_v1)

    demographics = d.get("demographics") or {}
    firmographics = d.get("firmographics") or {}
    comm = d.get("communication_style") or {}
    emo = d.get("emotional_profile") or {}
    moral = d.get("moral_framework") or {}
    evidence = d.get("source_evidence") or []

    role_titles = firmographics.get("role_titles") or []
    location_signals = demographics.get("location_signals") or []

    source_ids: list[str] = []
    for ev in evidence:
        for rid in ev.get("record_ids") or []:
            if rid not in source_ids:
                source_ids.append(rid)

    return TestingPersona(
        id=_stable_id(d.get("name", ""), d.get("summary", "")),
        name=d.get("name", ""),
        age=_midpoint_age(demographics.get("age_range")),
        gender=demographics.get("gender_distribution") or "",
        location=location_signals[0] if location_signals else "",
        education=demographics.get("education_level") or "",
        occupation=role_titles[0] if role_titles else "",
        industry=firmographics.get("industry") or "",
        income_bracket=demographics.get("income_bracket") or "",
        goals=list(d.get("goals") or []),
        pain_points=list(d.get("pains") or []),
        frustrations=list(d.get("objections") or []),
        motivations=list(d.get("motivations") or []),
        values=list(moral.get("core_values") or []),
        knowledge_domains=list(firmographics.get("tech_stack_signals") or []),
        interests=list(d.get("vocabulary") or []),
        communication_style=TestingCommunicationStyle(
            tone=comm.get("tone", ""),
            formality=comm.get("formality", ""),
            vocabulary_level=comm.get("vocabulary_level", ""),
            preferred_channels=list(comm.get("preferred_channels") or d.get("channels") or []),
        ),
        emotional_profile=TestingEmotionalProfile(
            baseline_mood=emo.get("baseline_mood", ""),
            stress_triggers=list(emo.get("stress_triggers") or []),
            coping_mechanisms=list(emo.get("coping_mechanisms") or []),
        ),
        moral_framework=TestingMoralFramework(
            core_values=list(moral.get("core_values") or []),
            ethical_stance=moral.get("ethical_stance", ""),
            moral_foundations=dict(moral.get("moral_foundations") or {}),
        ),
        bio=d.get("summary", ""),
        source_ids=source_ids,
        extra={
            "sample_quotes": list(d.get("sample_quotes") or []),
            "decision_triggers": list(d.get("decision_triggers") or []),
            "not_this": list(d.get("not_this") or []),
            "journey_stages": list(d.get("journey_stages") or []),
            "vocabulary": list(d.get("vocabulary") or []),
            "channels": list(d.get("channels") or []),
        },
    )


def source_context_from_records(
    context_id: str,
    records: list[dict[str, Any]] | list[str],
    *,
    metadata: dict[str, str] | None = None,
) -> SourceContext:
    """Build a SourceContext from the crawler records backing a persona.

    ``records`` accepts either raw text strings or dicts with a ``text`` / ``body``
    field. Pass whatever the pipeline has on hand — the scorers only need joined
    text and optional chunks.
    """
    chunks: list[str] = []
    for r in records:
        if isinstance(r, str):
            chunks.append(r)
        elif isinstance(r, dict):
            text = r.get("text") or r.get("body") or r.get("content") or ""
            if text:
                chunks.append(text)
    return SourceContext(
        id=context_id,
        text="\n\n".join(chunks),
        chunks=chunks,
        metadata=metadata or {},
    )
