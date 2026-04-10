from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .capability import DomainProfile
from .cohort import CohortModel, TechFamiliarity
from .evidence import SourceEvidence
from .persona import Demographics, Firmographics, JourneyStage
from .relational import RelationalSelf


class Contradiction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    axis: str = Field(description="The tension axis, e.g. autonomy_vs_belonging")
    description: str
    behavioral_manifestation: str
    confidence: float = Field(ge=0.0, le=1.0)


class PersonaV2Layer1(BaseModel):
    """Grounded layer: fields that should map directly to source data."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    name: str
    summary: str
    demographics: Demographics
    firmographics: Firmographics
    goals: list[str] = Field(min_length=2, max_length=8)
    pains: list[str] = Field(min_length=2, max_length=8)
    motivations: list[str] = Field(min_length=2, max_length=6)
    objections: list[str] = Field(min_length=1, max_length=6)
    channels: list[str] = Field(min_length=1, max_length=8)
    vocabulary: list[str] = Field(min_length=3, max_length=15)
    decision_triggers: list[str] = Field(min_length=1, max_length=6)
    sample_quotes: list[str] = Field(min_length=2, max_length=5)
    journey_stages: list[JourneyStage] = Field(min_length=2, max_length=5)
    source_evidence: list[SourceEvidence] = Field(min_length=3)


class PersonaV2Layer2(BaseModel):
    """Psychological / tension layer to be merged with Layer 1."""

    model_config = ConfigDict(extra="forbid")

    contradictions: list[Contradiction] = Field(default_factory=list, max_length=3)
    coping_style: str | None = None
    narrative_identity: str | None = None
    attachment_style: str | None = None
    conflict_style: str | None = None


class PersonaV2(PersonaV2Layer1):
    """Extended schema for age/cohort/capability/relational work."""

    model_config = ConfigDict(extra="forbid")

    birth_year: int = Field(ge=1920, le=2100)
    eval_year: int = Field(ge=1920, le=2100)
    age: int = Field(ge=0, le=120)
    cohort_label: str
    tech_familiarity_snapshot: TechFamiliarity
    cohort: CohortModel | None = None
    layer1: PersonaV2Layer1 | None = None
    layer2: PersonaV2Layer2 | None = None
    contradictions: list[Contradiction] = Field(default_factory=list, max_length=3)
    capability_matrix: dict[str, DomainProfile] = Field(default_factory=dict)
    relational_self: RelationalSelf | None = None


__all__ = [
    "Contradiction",
    "PersonaV2",
    "PersonaV2Layer1",
    "PersonaV2Layer2",
]
