from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RelationshipType(str, Enum):
    close_friend = "close_friend"
    acquaintance = "acquaintance"
    boss = "boss"
    subordinate = "subordinate"
    parent = "parent"
    child = "child"
    sibling = "sibling"
    romantic_partner = "romantic_partner"
    stranger = "stranger"
    rival = "rival"
    admired_peer = "admired_peer"
    mentor = "mentor"
    mentee = "mentee"


class TieStrength(str, Enum):
    weak = "weak"
    moderate = "moderate"
    strong = "strong"
    bonded = "bonded"


class PowerDynamic(str, Enum):
    equal = "equal"
    dependent = "dependent"
    authority = "authority"
    mutual_respect = "mutual_respect"


class InteractionGoal(str, Enum):
    bond = "bond"
    impress = "impress"
    teach = "teach"
    learn = "learn"
    defend = "defend"
    sell = "sell"
    avoid = "avoid"
    repair = "repair"
    collaborate = "collaborate"
    compete = "compete"


class Channel(str, Enum):
    text_async = "text_async"
    text_live = "text_live"
    voice = "voice"
    video = "video"
    public_thread = "public_thread"
    presentation = "presentation"


class ConflictStyle(str, Enum):
    avoid = "avoid"
    accommodate = "accommodate"
    compete = "compete"
    compromise = "compromise"
    collaborate = "collaborate"


class AttachmentStyle(str, Enum):
    secure = "secure"
    anxious = "anxious"
    avoidant = "avoidant"
    disorganized = "disorganized"


class TraitDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean: float = Field(ge=0.0, le=5.0)
    variance: float = Field(ge=0.0)
    skew: float = Field(default=0.0, ge=-3.0, le=3.0)
    floor: float = Field(ge=0.0, le=5.0)
    ceiling: float = Field(ge=0.0, le=5.0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "TraitDistribution":
        if self.floor > self.ceiling:
            raise ValueError("floor cannot exceed ceiling")
        return self


class RelationalProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relationship_type: RelationshipType = RelationshipType.acquaintance
    tie_strength: TieStrength = TieStrength.moderate
    power_dynamic: PowerDynamic = PowerDynamic.equal
    interaction_goal: InteractionGoal = InteractionGoal.collaborate
    channel: Channel = Channel.text_async

    warmth: float = Field(ge=0.0, le=1.0)
    dominance: float = Field(ge=0.0, le=1.0)
    disclosure_level: float = Field(ge=0.0, le=1.0)
    self_monitoring: float = Field(ge=0.0, le=1.0)
    spontaneity: float = Field(ge=0.0, le=1.0)
    slang_level: float = Field(ge=0.0, le=1.0)
    humor_frequency: float = Field(ge=0.0, le=1.0)
    conflict_style: ConflictStyle = ConflictStyle.compromise
    trust_level: float = Field(ge=0.0, le=1.0)
    vulnerability: float = Field(ge=0.0, le=1.0)
    reputation_concern: float = Field(ge=0.0, le=1.0)
    teaching_quality_modifier: float = Field(default=0.0, ge=-1.0, le=1.0)
    cognitive_load_modifier: float = Field(default=0.0, ge=-1.0, le=1.0)


class GroupProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conformity_increase: float = Field(ge=0.0, le=1.0)
    signaling_increase: float = Field(ge=0.0, le=1.0)
    nuance_decrease: float = Field(ge=0.0, le=1.0)
    risk_taking_shift: float = Field(ge=-1.0, le=1.0)
    in_group_warmth_boost: float = Field(ge=0.0, le=1.0)
    out_group_guardedness: float = Field(ge=0.0, le=1.0)
    audience_size_sensitivity: float = Field(ge=0.0, le=1.0)
    leadership_emergence: float = Field(ge=0.0, le=1.0)
    deference_to_consensus: float = Field(ge=0.0, le=1.0)


class IfThenSignature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition: dict[str, str] = Field(default_factory=dict)
    behavior_modifiers: dict[str, float] = Field(default_factory=dict)
    strength: float = Field(ge=0.0, le=1.0)
    evidence: str


class ConflictHistory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    safe: bool
    last_conflict_type: str | None = None
    repair_status: Literal["unresolved", "partial", "repaired", "forgiven"] = "unresolved"
    trust_recovery_rate: float = Field(ge=0.0, le=1.0)


class RelationalSelf(BaseModel):
    model_config = ConfigDict(extra="forbid")

    self_monitoring_level: float = Field(ge=0.0, le=5.0)
    attachment_style: AttachmentStyle = AttachmentStyle.secure
    baseline_warmth: float = Field(ge=0.0, le=1.0)
    baseline_dominance: float = Field(ge=0.0, le=1.0)
    trait_distributions: dict[str, TraitDistribution] = Field(default_factory=dict)
    relationship_profiles: dict[str, RelationalProfile] = Field(default_factory=dict)
    group_profile: GroupProfile | None = None
    if_then_signatures: list[IfThenSignature] = Field(default_factory=list)
    conflict_history: dict[str, ConflictHistory] = Field(default_factory=dict)


__all__ = [
    "AttachmentStyle",
    "Channel",
    "ConflictHistory",
    "ConflictStyle",
    "GroupProfile",
    "IfThenSignature",
    "InteractionGoal",
    "PowerDynamic",
    "RelationshipType",
    "RelationalProfile",
    "RelationalSelf",
    "TieStrength",
    "TraitDistribution",
]
