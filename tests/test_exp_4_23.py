"""Tests for Experiment 4.23: Persona wake-words.

Verifies:
1. Wake-word extraction from persona
2. Drift detection logic
3. Vocabulary overlap computation
4. Turns-to-recover calculation
5. Edge cases
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "twin"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from twin.chat import extract_wake_words, detect_drift
from experiment_4_23 import compute_vocab_overlap, find_turns_to_recover, TurnMetric


# ── Fixtures ─────────────────────────────────────────────────────────

PERSONA = {
    "name": "Alex the Engineer",
    "vocabulary": ["terraform", "CI/CD", "kubernetes", "infrastructure as code", "yaml"],
    "sample_quotes": [
        "If we can't automate it, it's not worth doing.",
        "I need something that just works out of the box.",
        "Show me the API docs first, then we'll talk.",
    ],
}


# ── Wake-word extraction tests ───────────────────────────────────────

class TestExtractWakeWords:
    def test_includes_vocabulary(self):
        ww = extract_wake_words(PERSONA)
        assert "terraform" in ww
        assert "kubernetes" in ww
        assert "yaml" in ww

    def test_includes_quote_fragments(self):
        ww = extract_wake_words(PERSONA)
        # Should include beginning of quotes
        assert any("automate" in w for w in ww)
        assert any("api docs" in w for w in ww)

    def test_empty_persona(self):
        ww = extract_wake_words({})
        assert ww == []

    def test_no_long_vocabulary(self):
        """Vocabulary phrases longer than 3 words should be excluded."""
        persona = {"vocabulary": ["short", "a longer phrase that is too long"]}
        ww = extract_wake_words(persona)
        assert "short" in ww
        # The long phrase should be excluded
        assert not any("a longer phrase" in w for w in ww)

    def test_lowercased(self):
        persona = {"vocabulary": ["Terraform", "CI/CD"]}
        ww = extract_wake_words(persona)
        assert "terraform" in ww
        assert "ci/cd" in ww


# ── Drift detection tests ───────────────────────────────────────────

class TestDetectDrift:
    def test_no_drift_with_wake_words(self):
        """Response containing wake-words should not be flagged."""
        ww = ["terraform", "kubernetes", "yaml"]
        response = "I use terraform and kubernetes daily for my yaml configs."
        assert not detect_drift(response, ww, threshold=0.1)

    def test_drift_without_wake_words(self):
        """Response missing wake-words should be flagged."""
        ww = ["terraform", "kubernetes", "yaml"]
        response = "The weather in Paris is lovely this time of year."
        assert detect_drift(response, ww, threshold=0.1)

    def test_empty_wake_words_no_drift(self):
        assert not detect_drift("anything", [], threshold=0.1)

    def test_threshold_sensitivity(self):
        ww = ["terraform", "kubernetes", "yaml", "ci/cd", "docker"]
        response = "I love terraform."  # 1/5 = 0.2
        assert not detect_drift(response, ww, threshold=0.1)  # 0.2 > 0.1
        assert detect_drift(response, ww, threshold=0.3)  # 0.2 < 0.3

    def test_case_insensitive(self):
        ww = ["terraform"]
        assert not detect_drift("I use TERRAFORM daily", ww, threshold=0.1)


# ── Vocab overlap tests ─────────────────────────────────────────────

class TestVocabOverlap:
    def test_full_overlap(self):
        overlap = compute_vocab_overlap("terraform kubernetes yaml", ["terraform", "kubernetes", "yaml"])
        assert overlap == 1.0

    def test_no_overlap(self):
        overlap = compute_vocab_overlap("hello world", ["terraform", "kubernetes"])
        assert overlap == 0.0

    def test_partial_overlap(self):
        overlap = compute_vocab_overlap("I use terraform", ["terraform", "kubernetes"])
        assert overlap == 0.5

    def test_empty_wake_words(self):
        assert compute_vocab_overlap("anything", []) == 0.0


# ── Turns to recover tests ──────────────────────────────────────────

class TestTurnsToRecover:
    def test_immediate_recovery(self):
        turns = [
            TurnMetric(turn=0, phase="recovery", question="q", response_snippet="r",
                       vocab_overlap=0.3, is_in_character=True),
        ]
        assert find_turns_to_recover(turns, warmup_baseline=0.3) == 1

    def test_delayed_recovery(self):
        turns = [
            TurnMetric(turn=0, phase="recovery", question="q", response_snippet="r",
                       vocab_overlap=0.0, is_in_character=False),
            TurnMetric(turn=1, phase="recovery", question="q", response_snippet="r",
                       vocab_overlap=0.0, is_in_character=False),
            TurnMetric(turn=2, phase="recovery", question="q", response_snippet="r",
                       vocab_overlap=0.2, is_in_character=True),
        ]
        assert find_turns_to_recover(turns, warmup_baseline=0.3) == 3

    def test_never_recovered(self):
        turns = [
            TurnMetric(turn=0, phase="recovery", question="q", response_snippet="r",
                       vocab_overlap=0.0, is_in_character=False),
            TurnMetric(turn=1, phase="recovery", question="q", response_snippet="r",
                       vocab_overlap=0.01, is_in_character=False),
        ]
        assert find_turns_to_recover(turns, warmup_baseline=0.3) == -1

    def test_ignores_non_recovery_turns(self):
        turns = [
            TurnMetric(turn=0, phase="warmup", question="q", response_snippet="r",
                       vocab_overlap=0.5, is_in_character=True),
            TurnMetric(turn=1, phase="recovery", question="q", response_snippet="r",
                       vocab_overlap=0.3, is_in_character=True),
        ]
        assert find_turns_to_recover(turns, warmup_baseline=0.4) == 1

    def test_zero_baseline(self):
        turns = [
            TurnMetric(turn=0, phase="recovery", question="q", response_snippet="r",
                       vocab_overlap=0.0, is_in_character=False),
        ]
        # threshold = 0 * 0.5 = 0, so any overlap >= 0 recovers
        assert find_turns_to_recover(turns, warmup_baseline=0.0) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
