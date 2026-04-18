from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .evidence import SourceEvidence


class Demographics(BaseModel):
    age_range: str = Field(description="e.g. '25-34'")
    gender_distribution: str = Field(description="e.g. 'predominantly female'")
    location_signals: list[str] = Field(
        description="Geographic indicators from data",
    )
    education_level: str | None = None
    income_bracket: str | None = None

    @field_validator("age_range", "gender_distribution", mode="before")
    @classmethod
    def _coerce_scalar_strings(cls, value: object) -> str:
        if isinstance(value, str):
            return value
        if value is None:
            return "unknown"
        return str(value)


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
    demographics: Demographics = Field(
        default_factory=lambda: Demographics(age_range="unknown", gender_distribution="unknown", location_signals=[]),
    )
    firmographics: Firmographics = Field(default_factory=Firmographics)
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


class PublicPersonPersonaV1(BaseModel):
    """Public profile persona extension used by lead magnet generation.

    This mirrors the SaaS TypeScript public persona contract instead of the
    customer-segment schema's psychological fields.
    """

    schema_version: Literal["1.0"] = "1.0"
    name: str = Field(description="The public person's name or public handle")
    summary: str = Field(description="2-3 sentence evidence-bound overview")
    system_role: str = Field(
        description="First-person role/purpose inferred from public evidence, not a document label",
    )
    demographics: Demographics
    firmographics: Firmographics
    capabilities: list[str] = Field(
        min_length=1,
        max_length=8,
        description="First-person capabilities grounded in public work, writing, projects, or profile text",
    )
    proof_points: list[str] = Field(
        min_length=1,
        max_length=8,
        description="Concrete public receipts that support who this person is and what they do",
    )
    decision_heuristics: list[str] = Field(
        min_length=1,
        max_length=8,
        description="How this person appears to make technical, product, or work decisions",
    )
    target_outcomes: list[str] = Field(
        min_length=1,
        max_length=6,
        description="Likely outcomes this person is trying to create, stated in first person when possible",
    )
    voice_markers: list[str] = Field(
        default_factory=lambda: ["First person; stay grounded in public evidence and say when memory is missing."],
        min_length=1,
        max_length=8,
        description="Observed voice/register/cadence markers from public writing or profile text",
    )
    not_this: list[str] = Field(
        default_factory=lambda: ["Do not invent private facts, metrics, roles, or identity links."],
        min_length=1,
        max_length=8,
    )
    conversation_contract: list[str] = Field(
        default_factory=lambda: ["Answer in first person and ground claims in public evidence."],
        min_length=1,
        max_length=8,
        description="Rules for chatting as this person without inventing private facts",
    )
    goals: list[str] = Field(min_length=1, max_length=8)
    pains: list[str] = Field(min_length=1, max_length=8)
    motivations: list[str] = Field(min_length=1, max_length=6)
    objections: list[str] = Field(min_length=1, max_length=6)
    channels: list[str] = Field(min_length=1, max_length=8)
    vocabulary: list[str] = Field(min_length=1, max_length=15)
    decision_triggers: list[str] = Field(min_length=1, max_length=6)
    sample_quotes: list[str] = Field(min_length=1, max_length=5)
    journey_stages: list[JourneyStage] = Field(min_length=1, max_length=5)
    source_evidence: list[SourceEvidence] = Field(min_length=3)

    @model_validator(mode="before")
    @classmethod
    def _fill_public_person_required_lists(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        defaults = {
            "capabilities": ["Public capabilities should be inferred only from used public records."],
            "proof_points": ["Used public crawl records are the available proof points."],
            "decision_heuristics": ["Prefer evidence grounded in public profile and activity records."],
            "target_outcomes": ["Keep agent responses grounded in public work and writing."],
            "voice_markers": ["Use first person and admit missing memory when public records are thin."],
            "not_this": ["Do not invent private facts, metrics, roles, or identity links."],
            "conversation_contract": ["Answer in first person and ground claims in public evidence."],
            "goals": ["Represent the public profile accurately from used records."],
            "pains": ["Thin or blocked public records can leave useful memory incomplete."],
            "motivations": ["Keep the public persona useful without adding private claims."],
            "objections": ["Do not trust claims that lack used public record support."],
            "channels": ["linkedin"],
            "vocabulary": ["public evidence"],
            "decision_triggers": ["Used public source records."],
            "sample_quotes": ["I can only speak from the public records available here."],
            "journey_stages": [{
                "stage": "awareness",
                "mindset": "evidence-bound",
                "key_actions": ["review used public records"],
                "content_preferences": ["public profile evidence"],
            }],
        }
        for key, fallback in defaults.items():
            value = data.get(key)
            if value is None or value == []:
                data[key] = fallback
        return data

    @field_validator(
        "capabilities",
        "proof_points",
        "decision_heuristics",
        "target_outcomes",
        "voice_markers",
        "not_this",
        "conversation_contract",
        "goals",
        "pains",
        "motivations",
        "objections",
        "channels",
        "vocabulary",
        "decision_triggers",
        "sample_quotes",
        mode="before",
    )
    @classmethod
    def _coerce_public_person_string_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            items: list[str] = []
            for item in value:
                items.extend(_coerce_public_person_string_values(item))
            return items
        return _coerce_public_person_string_values(value)


def _coerce_public_person_string_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        for key in ("text", "value", "claim", "goal", "pain", "motivation", "objection", "quote", "name"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return [nested]
        if value and all(str(key).isdigit() for key in value.keys()):
            return [_coerce_public_person_string(item) for _key, item in sorted(value.items(), key=lambda pair: int(str(pair[0])))]
    return [_coerce_public_person_string(value)]


def _coerce_public_person_string(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "value", "claim", "goal", "pain", "motivation", "objection", "quote", "name"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


class PersonaV1VoiceFirst(BaseModel):
    """exp-2.07 variant: vocabulary and sample_quotes declared FIRST.

    Pydantic v2 preserves declaration order in model_json_schema(); the
    Anthropic tool-use API fills fields in that order during structured
    output. So reordering the class declaration is the cleanest possible
    test of whether anchoring voice before demographics reduces stereotyping.

    All field constraints, types, and descriptions are identical to PersonaV1 —
    ONLY the order changes. No prompt changes, no model changes, no temperature
    changes. If distinctiveness differs, it is causally attributable to order.
    """

    schema_version: Literal["1.0"] = "1.0"
    name: str = Field(description="A memorable, descriptive name for this persona")
    summary: str = Field(
        description="2-3 sentence overview of who this persona is",
    )
    # --- VOICE FIRST ---
    vocabulary: list[str] = Field(
        min_length=3,
        max_length=15,
        description="Words and phrases this persona uses",
    )
    sample_quotes: list[str] = Field(
        min_length=2,
        max_length=5,
        description="Things this persona might say, in their own voice",
    )
    # --- then everything else ---
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
    decision_triggers: list[str] = Field(min_length=1, max_length=6)
    journey_stages: list[JourneyStage] = Field(min_length=2, max_length=5)
    source_evidence: list[SourceEvidence] = Field(min_length=3)
