from __future__ import annotations

import json
from pathlib import Path

import pytest

from persona_eval.schemas import Persona as EvalPersona
from synthesis.adapters.eval_adapter import persona_v1_to_eval
from synthesis.models.persona import PersonaV1

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

FIXTURES = [
    "persona_00.json",
    "persona_01.json",
]


@pytest.mark.parametrize("filename", FIXTURES)
def test_converted_persona_has_all_eval_fields(filename: str) -> None:
    """Every field the eval scorers read must be populated after conversion."""
    src = FIXTURES_DIR / filename
    data = json.loads(src.read_text())
    persona_v1 = PersonaV1.model_validate(data["persona"])

    eval_persona = persona_v1_to_eval(persona_v1, persona_id=data["cluster_id"])

    # Round-trip through JSON (simulates persona_eval CLI file loading)
    rehydrated = EvalPersona.model_validate(
        json.loads(json.dumps(eval_persona.model_dump()))
    )

    assert rehydrated.id
    assert rehydrated.name
    assert rehydrated.bio

    # communication_style — read by distributional scorers (formality, tone,
    # vocabulary_level) and channel-aware scorers (preferred_channels).
    assert rehydrated.communication_style.tone
    assert rehydrated.communication_style.formality
    assert rehydrated.communication_style.vocabulary_level
    assert rehydrated.communication_style.preferred_channels

    # emotional_profile — baseline + triggers + coping all feed behavioral scorers.
    assert rehydrated.emotional_profile.baseline_mood
    assert rehydrated.emotional_profile.stress_triggers
    assert rehydrated.emotional_profile.coping_mechanisms

    # moral_framework — moral_foundations may legitimately be empty per schema
    # (only include foundations with clear evidence), so pin the shape, not length.
    assert rehydrated.moral_framework.core_values
    assert rehydrated.moral_framework.ethical_stance
    assert isinstance(rehydrated.moral_framework.moral_foundations, dict)

    assert rehydrated.goals
    assert rehydrated.pain_points
    assert rehydrated.source_ids
