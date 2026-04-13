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
def test_fixture_roundtrip(filename: str) -> None:
    path = FIXTURES_DIR / filename
    data = json.loads(path.read_text())

    # 1. PersonaV1 validates (proves fixture was regenerated with psych fields)
    persona = PersonaV1.model_validate(data["persona"])

    # 2. Adapter produces a valid eval.Persona
    cluster_id = data["cluster_id"]
    eval_persona = persona_v1_to_eval(persona, persona_id=cluster_id)
    assert isinstance(eval_persona, EvalPersona)

    # 3. Round-trips through JSON (catches Pydantic shape drift)
    rt = EvalPersona.model_validate(eval_persona.model_dump())
    assert rt.id == cluster_id
    assert rt.name == persona.name
    assert rt.communication_style.tone == persona.communication_style.tone
    assert rt.emotional_profile.baseline_mood == persona.emotional_profile.baseline_mood
    assert rt.moral_framework.ethical_stance == persona.moral_framework.ethical_stance
