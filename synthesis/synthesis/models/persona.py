from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .evidence import SourceEvidence


class Demographics(BaseModel):
    age_range: str = Field(description="e.g. '25-34'")
    gender_distribution: str = Field(description="e.g. 'predominantly female'")
    location_signals: list[str] = Field(
        description="Geographic indicators from data",
    )
    education_level: str | None = None
    income_bracket: str | None = None


class Firmographics(BaseModel):
    company_size: str | None = Field(
        default=None,
        description="e.g. 'SMB (10-50 employees)'",
    )
    industry: str | None = None
    role_titles: list[str] = Field(default_factory=list)
    tech_stack_signals: list[str] = Field(default_factory=list)


class JourneyStage(BaseModel):
    stage: str = Field(description="e.g. 'awareness', 'consideration', 'decision'")
    mindset: str
    key_actions: list[str]
    content_preferences: list[str]


class EdgeCaseBehavior(BaseModel):
    """exp-1.15: how the persona reacts under provocation or stress.

    Explicit schema slot so synthesis commits to adversarial-robustness
    behaviors rather than leaving them as emergent twin-runtime behavior.
    Not groundedness-required — downstream inference from the persona's
    declared personality traits.
    """

    trigger: str = Field(
        description=(
            "The kind of user input that provokes this reaction, e.g. "
            "'rude tone', 'false premise', 'unsolicited advice', 'moralizing'"
        ),
    )
    reaction: str = Field(
        description=(
            "How the persona responds — a concrete behavior, not a feeling. "
            "E.g. 'pushes back factually without escalating', "
            "'asks a clarifying question instead of accepting the frame'"
        ),
    )
    tone_shift: str = Field(
        description=(
            "How the persona's speaking tone changes when triggered, e.g. "
            "'becomes more clipped and factual', 'shifts to deflecting humor', "
            "'stays warm but adds boundary-setting phrases'"
        ),
    )


class PersonaV1(BaseModel):
    """Core persona schema v1 — the structured output the LLM is forced to produce."""

    schema_version: Literal["1.0"] = "1.0"
    name: str = Field(description="A memorable, descriptive name for this persona")
    summary: str = Field(
        description="2-3 sentence overview of who this persona is",
    )
    demographics: Demographics
    firmographics: Firmographics
    goals: list[str] = Field(min_length=2, max_length=8)
    pains: list[str] = Field(min_length=2, max_length=8)
    motivations: list[str] = Field(min_length=2, max_length=6)
    objections: list[str] = Field(min_length=1, max_length=6)
    channels: list[str] = Field(
        min_length=1,
        max_length=8,
        description="Where they spend time online/offline",
    )
    vocabulary: list[str] = Field(
        min_length=3,
        max_length=15,
        description="Words and phrases this persona uses",
    )
    decision_triggers: list[str] = Field(min_length=1, max_length=6)
    sample_quotes: list[str] = Field(
        min_length=2,
        max_length=5,
        description="Things this persona might say, in their own voice",
    )
    journey_stages: list[JourneyStage] = Field(min_length=2, max_length=5)
    source_evidence: list[SourceEvidence] = Field(min_length=3)


class PersonaV1WithEdgeCases(PersonaV1):
    """exp-1.15 treatment schema: PersonaV1 + explicit edge-case behaviors.

    edge_case_behaviors is NOT a groundedness-required field — the existing
    groundedness checker only enforces evidence for goals/pains/motivations/
    objections, so no checker changes are needed.
    """

    edge_case_behaviors: list[EdgeCaseBehavior] = Field(
        min_length=3,
        max_length=6,
        description=(
            "How this persona reacts under provocation, stress, or adversarial "
            "conversational turns. At least 3 distinct trigger/reaction pairs."
        ),
    )
