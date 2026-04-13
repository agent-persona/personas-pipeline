from __future__ import annotations

import pytest
from pydantic import ValidationError

from synthesis.models.persona import (
    CommunicationStyle,
    EmotionalProfile,
    MoralFramework,
)


class TestCommunicationStyle:
    def test_valid(self) -> None:
        cs = CommunicationStyle(
            tone="direct",
            formality="professional",
            vocabulary_level="advanced",
            preferred_channels=["Slack", "email"],
        )
        assert cs.tone == "direct"
        assert cs.vocabulary_level == "advanced"

    def test_requires_at_least_one_channel(self) -> None:
        with pytest.raises(ValidationError):
            CommunicationStyle(
                tone="direct",
                formality="professional",
                vocabulary_level="advanced",
                preferred_channels=[],
            )


class TestEmotionalProfile:
    def test_valid(self) -> None:
        ep = EmotionalProfile(
            baseline_mood="calm",
            stress_triggers=["production outages"],
            coping_mechanisms=["deep work blocks"],
        )
        assert ep.baseline_mood == "calm"

    def test_requires_at_least_one_stress_trigger(self) -> None:
        with pytest.raises(ValidationError):
            EmotionalProfile(
                baseline_mood="calm",
                stress_triggers=[],
                coping_mechanisms=["deep work blocks"],
            )

    def test_requires_at_least_one_coping_mechanism(self) -> None:
        with pytest.raises(ValidationError):
            EmotionalProfile(
                baseline_mood="calm",
                stress_triggers=["production outages"],
                coping_mechanisms=[],
            )


class TestMoralFramework:
    def test_valid(self) -> None:
        mf = MoralFramework(
            core_values=["fairness", "honesty"],
            ethical_stance="utilitarian",
            moral_foundations={"care": 0.7, "fairness": 0.9},
        )
        assert mf.ethical_stance == "utilitarian"

    def test_requires_at_least_two_core_values(self) -> None:
        with pytest.raises(ValidationError):
            MoralFramework(
                core_values=["fairness"],
                ethical_stance="utilitarian",
                moral_foundations={"care": 0.7},
            )

    def test_moral_foundations_weights_must_be_in_range(self) -> None:
        with pytest.raises(ValidationError):
            MoralFramework(
                core_values=["fairness", "honesty"],
                ethical_stance="utilitarian",
                moral_foundations={"care": 1.5},  # > 1.0
            )

    def test_moral_foundations_negative_weights_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MoralFramework(
                core_values=["fairness", "honesty"],
                ethical_stance="utilitarian",
                moral_foundations={"care": -0.1},
            )

    def test_moral_foundations_empty_dict_accepted(self) -> None:
        """Per prompt rule 'omit rather than guess' — empty dict is a valid 'no evidence' state."""
        mf = MoralFramework(
            core_values=["fairness", "honesty"],
            ethical_stance="utilitarian",
            moral_foundations={},
        )
        assert mf.moral_foundations == {}

    def test_moral_foundations_boundary_values_accepted(self) -> None:
        mf = MoralFramework(
            core_values=["fairness", "honesty"],
            ethical_stance="utilitarian",
            moral_foundations={"care": 0.0, "liberty": 1.0},
        )
        assert mf.moral_foundations["care"] == 0.0
        assert mf.moral_foundations["liberty"] == 1.0

    def test_core_values_max_length_enforced(self) -> None:
        with pytest.raises(ValidationError):
            MoralFramework(
                core_values=["a", "b", "c", "d", "e", "f", "g"],  # 7 > 6
                ethical_stance="utilitarian",
                moral_foundations={"care": 0.5},
            )
