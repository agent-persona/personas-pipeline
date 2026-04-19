from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field
from enum import Enum


class CommunicationStyle(BaseModel):
    tone: str = ""
    formality: str = ""
    vocabulary_level: str = ""
    preferred_channels: list[str] = Field(default_factory=list)


class EmotionalProfile(BaseModel):
    baseline_mood: str = ""
    stress_triggers: list[str] = Field(default_factory=list)
    coping_mechanisms: list[str] = Field(default_factory=list)


class MoralFramework(BaseModel):
    core_values: list[str] = Field(default_factory=list)
    ethical_stance: str = ""
    moral_foundations: dict[str, float] = Field(default_factory=dict)


class Persona(BaseModel):
    id: str
    name: str
    age: int | None = None
    gender: str = ""
    location: str = ""
    education: str = ""
    occupation: str = ""
    industry: str = ""
    experience_years: int | None = None
    income_bracket: str = ""
    ethnicity: str = ""
    marital_status: str = ""
    behaviors: list[str] = Field(default_factory=list)
    habits: list[str] = Field(default_factory=list)
    personality_traits: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    lifestyle: str = ""
    communication_style: CommunicationStyle = Field(default_factory=CommunicationStyle)
    goals: list[str] = Field(default_factory=list)
    motivations: list[str] = Field(default_factory=list)
    pain_points: list[str] = Field(default_factory=list)
    frustrations: list[str] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)
    knowledge_domains: list[str] = Field(default_factory=list)
    expertise_level: str = ""
    emotional_profile: EmotionalProfile = Field(default_factory=EmotionalProfile)
    moral_framework: MoralFramework = Field(default_factory=MoralFramework)
    bio: str = ""
    source_ids: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    dimension_id: str
    dimension_name: str
    persona_id: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    suite: str = "persona"
    model: str = ""
    run_id: str = ""
