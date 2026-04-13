from __future__ import annotations

from synthesis.engine.prompt_builder import SYSTEM_PROMPT


class TestSystemPrompt:
    def test_mentions_communication_style(self) -> None:
        assert "communication_style" in SYSTEM_PROMPT

    def test_mentions_emotional_profile(self) -> None:
        assert "emotional_profile" in SYSTEM_PROMPT

    def test_mentions_moral_framework(self) -> None:
        assert "moral_framework" in SYSTEM_PROMPT

    def test_requires_evidence_for_psychological_fields_when_filled(self) -> None:
        # Psych sub-objects are optional; when filled they should be grounded.
        lower = SYSTEM_PROMPT.lower()
        assert "source_evidence entry rooted in that sub-object".lower() in lower

    def test_discourages_fabrication(self) -> None:
        lower = SYSTEM_PROMPT.lower()
        assert (
            "rather than guessing" in lower
            or "do not fabricate" in lower
            or "omit" in lower
        )
