from __future__ import annotations

import pytest
from pydantic import ValidationError

from synthesis.models.persona import (
    CommunicationStyle,
    Demographics,
    EmotionalProfile,
    Firmographics,
    JourneyStage,
    MoralFramework,
    PersonaV1,
)
from synthesis.models.evidence import SourceEvidence


def _minimal_persona_kwargs() -> dict:
    return dict(
        name="Test Persona",
        summary="A test persona for schema validation.",
        demographics=Demographics(
            age_range="25-34",
            gender_distribution="mixed",
            location_signals=["US"],
        ),
        firmographics=Firmographics(),
        goals=["goal one", "goal two"],
        pains=["pain one", "pain two"],
        motivations=["motivation one", "motivation two"],
        objections=["objection one"],
        not_this=["wouldn't use jargon sarcastically", "wouldn't skip post-mortems"],
        channels=["Slack"],
        vocabulary=["alpha", "beta", "gamma"],
        decision_triggers=["trigger one"],
        sample_quotes=["quote one", "quote two"],
        journey_stages=[
            JourneyStage(
                stage="evaluation",
                mindset="skeptical",
                key_actions=["reads docs"],
                content_preferences=["API reference"],
            ),
            JourneyStage(
                stage="activation",
                mindset="building",
                key_actions=["first integration"],
                content_preferences=["code samples"],
            ),
        ],
        communication_style=CommunicationStyle(
            tone="direct",
            formality="professional",
            vocabulary_level="advanced",
            preferred_channels=["Slack"],
        ),
        emotional_profile=EmotionalProfile(
            baseline_mood="calm",
            stress_triggers=["outages"],
            coping_mechanisms=["automation"],
        ),
        moral_framework=MoralFramework(
            core_values=["efficiency", "fairness"],
            ethical_stance="utilitarian",
            moral_foundations={"care": 0.5, "fairness": 0.8},
        ),
        source_evidence=[
            SourceEvidence(claim="c1", record_ids=["r1"], field_path="goals.0", confidence=0.9),
            SourceEvidence(claim="c2", record_ids=["r1"], field_path="pains.0", confidence=0.9),
            SourceEvidence(claim="c3", record_ids=["r1"], field_path="motivations.0", confidence=0.9),
        ],
    )


class TestPersonaV1Psychological:
    def test_valid_with_psychological_fields(self) -> None:
        p = PersonaV1(**_minimal_persona_kwargs())
        assert p.communication_style.tone == "direct"
        assert p.emotional_profile.baseline_mood == "calm"
        assert p.moral_framework.ethical_stance == "utilitarian"

    def test_communication_style_is_optional(self) -> None:
        kwargs = _minimal_persona_kwargs()
        del kwargs["communication_style"]
        p = PersonaV1(**kwargs)
        assert p.communication_style is None

    def test_emotional_profile_is_optional(self) -> None:
        kwargs = _minimal_persona_kwargs()
        del kwargs["emotional_profile"]
        p = PersonaV1(**kwargs)
        assert p.emotional_profile is None

    def test_moral_framework_is_optional(self) -> None:
        kwargs = _minimal_persona_kwargs()
        del kwargs["moral_framework"]
        p = PersonaV1(**kwargs)
        assert p.moral_framework is None
