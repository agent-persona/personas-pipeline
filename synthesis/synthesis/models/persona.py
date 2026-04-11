from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .evidence import SourceEvidence

# ── Experiment 1.01: schema width ────────────────────────────────────
SchemaWidth = Literal["minimal", "current", "maximal"]


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


# ── Experiment 1.01: schema width variants ───────────────────────────

class PersonaMinimal(BaseModel):
    """Minimal persona (~5 fields). Tests whether sparse schemas still
    produce usable twins, or if they lack enough detail for consistency."""

    schema_version: Literal["1.0"] = "1.0"
    name: str = Field(description="A memorable, descriptive name for this persona")
    summary: str = Field(
        description="2-3 sentence overview of who this persona is",
    )
    goals: list[str] = Field(min_length=2, max_length=8)
    pains: list[str] = Field(min_length=2, max_length=8)
    source_evidence: list[SourceEvidence] = Field(min_length=3)


class PersonaMaximal(PersonaV1):
    """Maximal persona (~25 fields). Tests whether extra surface area
    helps twin consistency or introduces contradictions."""

    backstory: str = Field(
        description="2-3 paragraph first-person narrative life history",
    )
    daily_routine: str = Field(
        description="Typical workday from morning to evening",
    )
    communication_style: str = Field(
        description="How they communicate: formal/informal, verbose/terse, "
        "preferred medium (Slack, email, meetings)",
    )
    decision_making_process: str = Field(
        description="How they evaluate and decide on new tools or vendors",
    )
    brand_affinities: list[str] = Field(
        min_length=1,
        max_length=8,
        description="Brands, tools, or products they trust and use regularly",
    )
    frustration_triggers: list[str] = Field(
        min_length=1,
        max_length=6,
        description="Specific situations that trigger frustration or disengagement",
    )
    success_metrics: list[str] = Field(
        min_length=1,
        max_length=5,
        description="How they measure their own professional success",
    )
    learning_style: str = Field(
        description="How they prefer to learn: docs, video, peer advice, "
        "hands-on experimentation",
    )
    team_dynamics: str = Field(
        description="Role in team: leader, contributor, mentor, lone wolf. "
        "How they collaborate.",
    )
    budget_authority: str = Field(
        description="Purchasing power: individual contributor, influencer, "
        "decision maker, budget owner",
    )
    content_consumption: list[str] = Field(
        min_length=1,
        max_length=6,
        description="Newsletters, podcasts, blogs, communities they follow",
    )
    career_trajectory: str = Field(
        description="Where they have been professionally and where they "
        "aspire to go next",
    )
    pet_peeves: list[str] = Field(
        min_length=1,
        max_length=5,
        description="Specific things that annoy them in products or processes",
    )


SCHEMA_WIDTH_MAP: dict[SchemaWidth, type[BaseModel]] = {
    "minimal": PersonaMinimal,
    "current": PersonaV1,
    "maximal": PersonaMaximal,
}
