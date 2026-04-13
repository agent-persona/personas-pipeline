"""PersonaV1 variant with value-level provenance (experiment 1.10).

Each item in goals, pains, motivations, and objections carries inline
(source_record_ids, confidence, model_version) rather than relying on
a separate source_evidence array.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .evidence import SourceEvidence


class ProvenancedValue(BaseModel):
    """A string value with inline provenance metadata."""
    text: str = Field(description="The actual goal, pain, motivation, or objection")
    source_record_ids: list[str] = Field(
        min_length=1,
        description="Record IDs that support this claim",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How strongly the data supports this claim",
    )
    model_version: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Which model generated this value",
    )


class Demographics(BaseModel):
    age_range: str
    gender_distribution: str
    location_signals: list[str]
    education_level: str | None = None
    income_bracket: str | None = None


class Firmographics(BaseModel):
    company_size: str | None = None
    industry: str | None = None
    role_titles: list[str] = Field(default_factory=list)
    tech_stack_signals: list[str] = Field(default_factory=list)


class JourneyStage(BaseModel):
    stage: str
    mindset: str
    key_actions: list[str]
    content_preferences: list[str]


class PersonaV1Provenance(BaseModel):
    """Persona schema with inline provenance on every evidence-required field."""

    schema_version: Literal["1.0-provenance"] = "1.0-provenance"
    name: str
    summary: str
    demographics: Demographics
    firmographics: Firmographics
    goals: list[ProvenancedValue] = Field(min_length=2, max_length=8)
    pains: list[ProvenancedValue] = Field(min_length=2, max_length=8)
    motivations: list[ProvenancedValue] = Field(min_length=2, max_length=6)
    objections: list[ProvenancedValue] = Field(min_length=1, max_length=6)
    channels: list[str] = Field(min_length=1, max_length=8)
    vocabulary: list[str] = Field(min_length=3, max_length=15)
    decision_triggers: list[str] = Field(min_length=1, max_length=6)
    sample_quotes: list[str] = Field(min_length=2, max_length=5)
    journey_stages: list[JourneyStage] = Field(min_length=2, max_length=5)
    # No separate source_evidence — it's inline on each value
