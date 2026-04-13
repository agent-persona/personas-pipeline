from __future__ import annotations

import json
from pathlib import Path

from persona_eval.schemas import Persona as EvalPersona
from synthesis.adapters.eval_adapter import persona_v1_to_eval
from synthesis.models.persona import PersonaV1

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"


def test_converted_persona_has_all_eval_fields() -> None:
    """Every field the eval scorers read must be populated after conversion."""
    src = OUTPUT_DIR / "persona_00.json"
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
    assert rehydrated.communication_style.tone
    assert rehydrated.communication_style.vocabulary_level
    assert rehydrated.emotional_profile.baseline_mood
    assert rehydrated.emotional_profile.stress_triggers
    assert rehydrated.moral_framework.core_values
    assert rehydrated.moral_framework.ethical_stance
    assert rehydrated.goals
    assert rehydrated.pain_points
    assert rehydrated.source_ids
