from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

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


class CommunicationStyle(BaseModel):
    """How this persona expresses themselves — matches persona_eval.CommunicationStyle."""

    tone: str = Field(
        description="Dominant emotional register, e.g. 'direct', 'warm', 'enthusiastic', 'analytical', 'skeptical'",
    )
    formality: str = Field(
        description="e.g. 'casual', 'professional', 'formal'",
    )
    vocabulary_level: str = Field(
        description="'basic', 'intermediate', or 'advanced' — based on technical sophistication of their language",
    )
    preferred_channels: list[str] = Field(
        min_length=1,
        description="Channels where this persona actually communicates (Slack, email, Intercom, forums, etc.)",
    )


class EmotionalProfile(BaseModel):
    """Emotional baseline + triggers + coping — matches persona_eval.EmotionalProfile."""

    baseline_mood: str = Field(
        description="Dominant emotional baseline — 'calm', 'anxious', 'optimistic', 'frustrated', 'enthusiastic', etc.",
    )
    stress_triggers: list[str] = Field(
        min_length=1,
        max_length=6,
        description="What reliably makes this persona stressed or frustrated, grounded in source records",
    )
    coping_mechanisms: list[str] = Field(
        min_length=1,
        max_length=6,
        description="How this persona handles frustration — 'files support ticket', 'writes automation', 'vents on Twitter', etc.",
    )


class MoralFramework(BaseModel):
    """Values and ethical stance — matches persona_eval.MoralFramework."""

    core_values: list[str] = Field(
        min_length=2,
        max_length=6,
        description="The values this persona treats as non-negotiable — 'fairness', 'autonomy', 'efficiency', etc.",
    )
    ethical_stance: str = Field(
        description=(
            "Best-fit label, e.g. 'utilitarian', 'virtue ethics', 'deontological', "
            "'principlist', 'care ethics'. Open-ended — prefer one of these if the "
            "language fits, otherwise use a short descriptive label."
        ),
    )
    moral_foundations: dict[str, float] = Field(
        description=(
            "Moral Foundations Theory weights in [0.0, 1.0]. "
            "Keys: care, fairness, loyalty, authority, sanctity, liberty. "
            "Not all keys required — include only those with clear evidence."
        ),
    )

    @field_validator("moral_foundations")
    @classmethod
    def _weights_in_range(cls, v: dict[str, float]) -> dict[str, float]:
        for k, weight in v.items():
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"moral_foundations[{k}] = {weight} must be in [0.0, 1.0]")
        return v


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
    communication_style: CommunicationStyle = Field(
        description="How this persona speaks and writes. Derived from their verbatim messages in source data.",
    )
    emotional_profile: EmotionalProfile = Field(
        description="Emotional baseline, stress triggers, and coping mechanisms. Grounded in support tickets, complaints, and behavioral signals.",
    )
    moral_framework: MoralFramework = Field(
        description="Core values and ethical stance. Inferred from what this persona cares about — language of fairness, autonomy, efficiency, etc.",
    )
    source_evidence: list[SourceEvidence] = Field(min_length=3)
