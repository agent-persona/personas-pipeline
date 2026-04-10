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
    not_this: list[str] = Field(
        min_length=2,
        max_length=6,
        description=(
            "Identity-level negatives — things this persona would NOT do, say, "
            "or believe. Distinct from objections (which are sales pushback). "
            "Use these as the scaffolding for authentic out-of-character refusals."
        ),
    )
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
