"""Tests for Experiment 2.14: Constitutional persona.

Verifies:
1. Constitution loads from disk
2. Constitutional system prompt includes constitution
3. Hedging rate metric
4. Grounded-quote rate metric
5. Edge cases
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from synthesis.engine.prompt_builder import (
    SYSTEM_PROMPT,
    build_constitutional_system_prompt,
    load_constitution,
)
from experiment_2_14 import compute_hedging_rate, compute_grounded_quote_rate


class TestConstitution:
    def test_loads_from_disk(self):
        text = load_constitution()
        assert len(text) > 100
        assert "must not hedge" in text.lower() or "hedge" in text.lower()

    def test_includes_grounding_principles(self):
        text = load_constitution()
        assert "evidence" in text.lower()
        assert "confidence" in text.lower()

    def test_includes_voice_principles(self):
        text = load_constitution()
        assert "first-person" in text.lower()
        assert "vocabulary" in text.lower()

    def test_constitutional_prompt_extends_base(self):
        prompt = build_constitutional_system_prompt()
        assert SYSTEM_PROMPT in prompt
        assert "constitution" in prompt.lower() or "principle" in prompt.lower()

    def test_constitutional_prompt_longer_than_base(self):
        prompt = build_constitutional_system_prompt()
        assert len(prompt) > len(SYSTEM_PROMPT) + 200


class TestHedgingRate:
    def test_no_hedging(self):
        persona = {
            "summary": "I deploy infrastructure daily.",
            "goals": ["Automate deployments", "Reduce downtime"],
            "sample_quotes": ["I need this done now."],
        }
        assert compute_hedging_rate(persona) == 0.0

    def test_all_hedging(self):
        persona = {
            "goals": ["might reduce costs", "could improve speed"],
            "pains": ["potentially slow processes"],
        }
        rate = compute_hedging_rate(persona)
        assert rate == 1.0

    def test_partial_hedging(self):
        persona = {
            "goals": ["reduce costs", "might improve speed"],
        }
        rate = compute_hedging_rate(persona)
        assert rate == 0.5

    def test_empty(self):
        assert compute_hedging_rate({}) == 0.0

    def test_detects_various_hedge_words(self):
        for word in ["might", "could", "potentially", "perhaps", "possibly", "it seems", "tend to"]:
            persona = {"goals": [f"We {word} want this"]}
            assert compute_hedging_rate(persona) > 0, f"Failed to detect '{word}'"


class TestGroundedQuoteRate:
    def test_grounded_first_person(self):
        persona = {
            "sample_quotes": [
                "I need the API docs before I commit to anything.",
                "We deploy 15 times a day using CI/CD.",
            ],
        }
        assert compute_grounded_quote_rate(persona) == 1.0

    def test_third_person_not_grounded(self):
        persona = {
            "sample_quotes": [
                "The team values efficiency.",
                "Users prefer simple interfaces.",
            ],
        }
        assert compute_grounded_quote_rate(persona) == 0.0

    def test_first_person_but_vague(self):
        persona = {
            "sample_quotes": [
                "I like good things.",  # first person but no specifics
            ],
        }
        assert compute_grounded_quote_rate(persona) == 0.0

    def test_mixed(self):
        persona = {
            "sample_quotes": [
                "I need the API to handle 1000 requests per second.",  # grounded
                "Things should work better.",  # not grounded
            ],
        }
        assert compute_grounded_quote_rate(persona) == 0.5

    def test_empty(self):
        assert compute_grounded_quote_rate({}) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
