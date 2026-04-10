from __future__ import annotations

import pytest
from pydantic import ValidationError

from synthesis.models import (
    CapabilityMatrix,
    ConditionModifiers,
    ConfidenceLevel,
    DomainProfile,
    ExperienceProfile,
    IdentitySalience,
    ImitationMode,
    MentorshipLevel,
    ObservationEffect,
    Recency,
)


def test_domain_profile_validates_and_serializes_cleanly() -> None:
    profile = DomainProfile(
        factual_knowledge=4.5,
        procedural_skill=4.0,
        taste_judgment=3.5,
        creativity=3.0,
        speed=4.2,
        consistency=4.1,
        error_recovery=3.8,
        teaching_ability=2.5,
        tool_fluency=4.4,
        confidence_calibration=3.9,
        imitation_vs_origination=ImitationMode.originator,
        collaboration_multiplier=1.2,
        failure_mode="freeze",
        observation_effect=ObservationEffect.degrades,
        meta_learning_ability=4.0,
        social_proof_dependency=1.0,
        identity_salience=IdentitySalience.core,
        motivation=4.8,
        moral_boundary="No deception in client work",
        confidence_level=ConfidenceLevel.grounded,
        evidence="Source text mentions repeated production debugging and code reviews",
        experience=ExperienceProfile(
            years_exposed=8,
            years_practiced=5,
            deliberate_practice_intensity=3.5,
            recency=Recency.active,
            decay_rate=0.1,
            success_rate=0.82,
            mentorship_received=MentorshipLevel.formal,
            unconscious_competence_stage=3,
        ),
        conditions=ConditionModifiers(
            stress_modifier=-0.6,
            fatigue_modifier=-1.0,
            context_switch_cost="high",
            peak_performance_context="quiet solo work",
            ceiling=4.8,
            floor=1.2,
        ),
    )

    matrix: CapabilityMatrix = {"coding": profile}

    assert matrix["coding"].experience is not None
    assert matrix["coding"].conditions is not None
    dumped = matrix["coding"].model_dump()
    assert dumped["failure_mode"] == "freeze"
    assert dumped["experience"]["years_practiced"] == 5
    assert dumped["conditions"]["context_switch_cost"] == "high"


def test_domain_profile_rejects_invalid_ranges() -> None:
    with pytest.raises(ValidationError, match="Input should be less than or equal to 5"):
        DomainProfile(
            factual_knowledge=5.1,
            procedural_skill=4.0,
            taste_judgment=3.5,
            creativity=3.0,
            speed=4.2,
            consistency=4.1,
            error_recovery=3.8,
            teaching_ability=2.5,
            tool_fluency=4.4,
            confidence_calibration=3.9,
            failure_mode="freeze",
            meta_learning_ability=4.0,
            social_proof_dependency=1.0,
            motivation=4.8,
        )

    with pytest.raises(ValidationError, match="years_practiced cannot exceed years_exposed"):
        ExperienceProfile(
            years_exposed=2,
            years_practiced=3,
            deliberate_practice_intensity=2.0,
            decay_rate=0.2,
            success_rate=0.5,
            unconscious_competence_stage=2,
        )
