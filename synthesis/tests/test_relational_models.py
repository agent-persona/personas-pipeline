from __future__ import annotations

import pytest
from pydantic import ValidationError

from synthesis.models import (
    AttachmentStyle,
    Channel,
    ConflictHistory,
    ConflictStyle,
    GroupProfile,
    IfThenSignature,
    InteractionGoal,
    PowerDynamic,
    RelationshipType,
    RelationalProfile,
    RelationalSelf,
    TieStrength,
    TraitDistribution,
)


def test_relational_self_round_trips_core_context() -> None:
    relational_self = RelationalSelf(
        self_monitoring_level=3.4,
        attachment_style=AttachmentStyle.secure,
        baseline_warmth=0.72,
        baseline_dominance=0.41,
        trait_distributions={
            "extraversion": TraitDistribution(
                mean=2.8,
                variance=0.7,
                skew=-0.2,
                floor=1.0,
                ceiling=4.3,
            ),
        },
        relationship_profiles={
            "boss": RelationalProfile(
                relationship_type=RelationshipType.boss,
                tie_strength=TieStrength.moderate,
                power_dynamic=PowerDynamic.dependent,
                interaction_goal=InteractionGoal.impress,
                channel=Channel.text_async,
                warmth=0.45,
                dominance=0.2,
                disclosure_level=0.25,
                self_monitoring=0.8,
                spontaneity=0.2,
                slang_level=0.1,
                humor_frequency=0.15,
                conflict_style=ConflictStyle.compromise,
                trust_level=0.65,
                vulnerability=0.2,
                reputation_concern=0.7,
                teaching_quality_modifier=-0.1,
                cognitive_load_modifier=-0.2,
            ),
        },
        group_profile=GroupProfile(
            conformity_increase=0.7,
            signaling_increase=0.5,
            nuance_decrease=0.4,
            risk_taking_shift=-0.2,
            in_group_warmth_boost=0.6,
            out_group_guardedness=0.8,
            audience_size_sensitivity=0.7,
            leadership_emergence=0.3,
            deference_to_consensus=0.6,
        ),
        if_then_signatures=[
            IfThenSignature(
                condition={"relationship": "boss", "stakes": "high"},
                behavior_modifiers={"warmth": -0.2, "self_monitoring": 0.3},
                strength=0.9,
                evidence="Observed in review meetings",
            ),
        ],
        conflict_history={
            "boss": ConflictHistory(
                safe=True,
                last_conflict_type="disagreement",
                repair_status="partial",
                trust_recovery_rate=0.5,
            ),
        },
    )

    dumped = relational_self.model_dump()
    assert dumped["attachment_style"] == "secure"
    assert dumped["relationship_profiles"]["boss"]["interaction_goal"] == "impress"
    assert dumped["group_profile"]["conformity_increase"] == 0.7
    assert dumped["if_then_signatures"][0]["condition"]["relationship"] == "boss"


def test_trait_distribution_requires_plausible_bounds() -> None:
    with pytest.raises(ValidationError, match="floor cannot exceed ceiling"):
        TraitDistribution(mean=2.0, variance=0.4, floor=4.0, ceiling=3.0)


def test_relational_profile_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RelationalProfile(
            relationship_type=RelationshipType.acquaintance,
            tie_strength=TieStrength.weak,
            power_dynamic=PowerDynamic.equal,
            interaction_goal=InteractionGoal.learn,
            channel=Channel.voice,
            warmth=0.4,
            dominance=0.4,
            disclosure_level=0.4,
            self_monitoring=0.4,
            spontaneity=0.4,
            slang_level=0.4,
            humor_frequency=0.4,
            conflict_style=ConflictStyle.collaborate,
            trust_level=0.4,
            vulnerability=0.4,
            reputation_concern=0.4,
            teaching_quality_modifier=0.0,
            cognitive_load_modifier=0.0,
            surprise="nope",
        )
