from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImitationMode(str, Enum):
    imitator = "imitator"
    originator = "originator"
    both = "both"


class ObservationEffect(str, Enum):
    improves = "improves"
    degrades = "degrades"
    neutral = "neutral"


class IdentitySalience(str, Enum):
    core = "core"
    competent = "competent"
    chore = "chore"
    avoidance = "avoidance"


class ConfidenceLevel(str, Enum):
    grounded = "grounded"
    inferred = "inferred"
    guessed = "guessed"


class Recency(str, Enum):
    active = "active"
    rusty = "rusty"
    dormant = "dormant"
    atrophied = "atrophied"


class MentorshipLevel(str, Enum):
    none = "none"
    informal = "informal"
    formal = "formal"
    intensive = "intensive"


class ExperienceProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    years_exposed: int = Field(ge=0)
    years_practiced: int = Field(ge=0)
    deliberate_practice_intensity: float = Field(ge=0.0, le=5.0)
    recency: Recency = Recency.active
    decay_rate: float = Field(ge=0.0, le=1.0)
    success_rate: float = Field(ge=0.0, le=1.0)
    mentorship_received: MentorshipLevel = MentorshipLevel.none
    unconscious_competence_stage: int = Field(ge=1, le=4)

    @model_validator(mode="after")
    def _validate_practice_exposure(self) -> "ExperienceProfile":
        if self.years_practiced > self.years_exposed:
            raise ValueError("years_practiced cannot exceed years_exposed")
        return self


class ConditionModifiers(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stress_modifier: float = Field(default=0.0, ge=-2.0, le=0.0)
    fatigue_modifier: float = Field(default=0.0, ge=-2.0, le=0.0)
    context_switch_cost: Literal["low", "medium", "high"] = "medium"
    peak_performance_context: str | None = None
    ceiling: float = Field(default=5.0, ge=0.0, le=5.0)
    floor: float = Field(default=0.0, ge=0.0, le=5.0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ConditionModifiers":
        if self.floor > self.ceiling:
            raise ValueError("floor cannot exceed ceiling")
        return self


class DomainProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factual_knowledge: float = Field(ge=0.0, le=5.0)
    procedural_skill: float = Field(ge=0.0, le=5.0)
    taste_judgment: float = Field(ge=0.0, le=5.0)
    creativity: float = Field(ge=0.0, le=5.0)
    speed: float = Field(ge=0.0, le=5.0)
    consistency: float = Field(ge=0.0, le=5.0)
    error_recovery: float = Field(ge=0.0, le=5.0)
    teaching_ability: float = Field(ge=0.0, le=5.0)
    tool_fluency: float = Field(ge=0.0, le=5.0)
    confidence_calibration: float = Field(ge=0.0, le=5.0)

    imitation_vs_origination: ImitationMode = ImitationMode.both
    collaboration_multiplier: float = Field(default=0.0, ge=-1.0, le=3.0)
    failure_mode: str
    observation_effect: ObservationEffect = ObservationEffect.neutral
    meta_learning_ability: float = Field(ge=0.0, le=5.0)
    social_proof_dependency: float = Field(ge=0.0, le=5.0)

    identity_salience: IdentitySalience = IdentitySalience.competent
    motivation: float = Field(ge=0.0, le=5.0)
    moral_boundary: str | None = None

    confidence_level: ConfidenceLevel = ConfidenceLevel.inferred
    evidence: str | None = None

    experience: ExperienceProfile | None = None
    conditions: ConditionModifiers | None = None


CapabilityMatrix = dict[str, DomainProfile]


__all__ = [
    "CapabilityMatrix",
    "ConditionModifiers",
    "ConfidenceLevel",
    "DomainProfile",
    "ExperienceProfile",
    "IdentitySalience",
    "ImitationMode",
    "MentorshipLevel",
    "ObservationEffect",
    "Recency",
]
