"""Tests for Experiment 4.03: Drift over turn count.

Verifies:
1. Persona word extraction
2. Turn scoring
3. Drift curve computation
4. Half-life detection
5. Scripted question bank
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "evals"))
sys.path.insert(0, str(REPO_ROOT / "twin"))

from drift_curve import (
    CheckpointScore,
    DriftCurve,
    compute_drift_curve,
    compute_half_life,
    extract_persona_words,
    score_turn,
)
from twin.conversations import SCRIPTED_QUESTIONS, DRIFT_CHECKPOINTS


PERSONA = {
    "vocabulary": ["terraform", "CI/CD", "kubernetes", "webhook", "GraphQL"],
    "sample_quotes": [
        "If we can't automate it, it's not worth doing.",
        "Show me the API docs first.",
    ],
    "goals": ["automate deployments", "reduce downtime"],
    "pains": ["manual syncing", "legacy systems"],
}


class TestExtractPersonaWords:
    def test_includes_vocabulary(self):
        words = extract_persona_words(PERSONA)
        assert "terraform" in words
        assert "kubernetes" in words
        assert "graphql" in words

    def test_includes_goal_words(self):
        words = extract_persona_words(PERSONA)
        assert "automate" in words
        assert "deployments" in words

    def test_excludes_stopwords(self):
        words = extract_persona_words(PERSONA)
        assert "the" not in words
        assert "it" not in words
        assert "we" not in words

    def test_empty_persona(self):
        assert len(extract_persona_words({})) == 0


class TestScoreTurn:
    def test_high_overlap(self):
        words = extract_persona_words(PERSONA)
        response = "I use terraform and kubernetes for CI/CD to automate deployments."
        score = score_turn(response, words)
        assert score > 0.2

    def test_no_overlap(self):
        words = extract_persona_words(PERSONA)
        response = "The weather in Paris is lovely."
        score = score_turn(response, words)
        assert score == 0.0

    def test_empty_words(self):
        assert score_turn("anything", set()) == 0.0


class TestDriftCurve:
    def test_stable_curve(self):
        """Consistent responses should show stable overlap."""
        words = {"terraform", "kubernetes", "automate"}
        responses = [
            "I use terraform daily",  # T1
            "setup", "test", "build",  # T2-4
            "kubernetes and terraform are key",  # T5
            "a", "b", "c", "d",  # T6-9
            "automate everything with terraform",  # T10
        ]
        curve = compute_drift_curve(responses, words, [1, 5, 10])
        assert len(curve.checkpoints) == 3
        assert curve.half_life == -1  # never drops below 50%

    def test_declining_curve(self):
        """Responses that lose persona words should show decline."""
        words = {"terraform", "kubernetes", "automate", "deploy", "infrastructure"}
        responses = ["terraform kubernetes automate deploy infrastructure"]  # T1: full overlap
        # Pad with empty responses for T2-49
        responses += ["nothing relevant here"] * 49
        curve = compute_drift_curve(responses, words, [1, 50])
        assert curve.baseline_overlap == 1.0
        assert curve.final_overlap == 0.0
        assert curve.half_life == 50  # drops at T50

    def test_half_life_detection(self):
        words = {"terraform", "kubernetes", "automate"}
        responses = [
            "terraform kubernetes automate",  # T1: 3/3 = 1.0
            "a", "b", "c",  # T2-4
            "terraform",  # T5: 1/3 = 0.33 < 0.5 -> half-life
        ]
        curve = compute_drift_curve(responses, words, [1, 5])
        assert curve.half_life == 5

    def test_decay_rate(self):
        words = {"terraform"}
        responses = ["terraform"] + ["nothing"] * 9
        curve = compute_drift_curve(responses, words, [1, 10])
        assert curve.decay_rate > 0  # positive = declining

    def test_empty_responses(self):
        curve = compute_drift_curve([], {"a", "b"}, [1, 5])
        assert len(curve.checkpoints) == 0
        assert curve.half_life == -1


class TestComputeHalfLife:
    def test_returns_value(self):
        curve = DriftCurve(persona_name="test", persona_word_count=5, half_life=10)
        assert compute_half_life(curve) == 10

    def test_never_reached(self):
        curve = DriftCurve(persona_name="test", persona_word_count=5, half_life=-1)
        assert compute_half_life(curve) == -1


class TestScriptedQuestions:
    def test_has_50_questions(self):
        assert len(SCRIPTED_QUESTIONS) == 50

    def test_checkpoints_defined(self):
        assert DRIFT_CHECKPOINTS == [1, 5, 10, 25, 50]

    def test_questions_are_strings(self):
        for q in SCRIPTED_QUESTIONS:
            assert isinstance(q, str)
            assert len(q) > 10

    def test_questions_end_with_punctuation(self):
        for q in SCRIPTED_QUESTIONS:
            assert q[-1] in ".?!", f"Question missing punctuation: {q[:30]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
