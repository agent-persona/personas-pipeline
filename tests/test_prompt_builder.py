from __future__ import annotations

from synthesis.engine.prompt_builder import SYSTEM_PROMPT


class TestSystemPrompt:
    def test_mentions_communication_style(self) -> None:
        assert "communication_style" in SYSTEM_PROMPT

    def test_mentions_emotional_profile(self) -> None:
        assert "emotional_profile" in SYSTEM_PROMPT

    def test_mentions_moral_framework(self) -> None:
        assert "moral_framework" in SYSTEM_PROMPT

    def test_requires_evidence_for_psychological_fields(self) -> None:
        assert (
            "moral_framework.core_values" in SYSTEM_PROMPT
            or "moral_framework.ethical_stance" in SYSTEM_PROMPT
        )
        assert (
            "emotional_profile.stress_triggers" in SYSTEM_PROMPT
            or "emotional_profile.baseline_mood" in SYSTEM_PROMPT
        )

    def test_warns_against_fabrication(self) -> None:
        lower = SYSTEM_PROMPT.lower()
        assert (
            "do not fabricate" in lower
            or "do not invent" in lower
            or "must be grounded" in lower
        )
