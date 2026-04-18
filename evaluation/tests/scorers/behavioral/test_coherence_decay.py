"""Tests for D34 Multi-Turn Coherence Decay scorer."""

import pytest
from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext


def _ctx_with_turns(turns: list[str]) -> SourceContext:
    return SourceContext(id="s1", text="coherence test", extra_data={"conversation_turns": turns})


SAMPLE_PERSONA = Persona(id="p1", name="Alice", occupation="Teacher")

# Consistent topic throughout — no decay
CONSISTENT_TURNS = [
    f"Education is essential for building strong communities. Point {i}."
    for i in range(25)
]

# Starts on-topic, then drifts wildly — decay expected
DECAYING_TURNS = [
    "Education is important for building strong communities.",
    "Schools need adequate funding for quality teaching.",
    "Teachers should be paid fairly for their critical work.",
    "Good schools create informed and engaged citizens.",
    "I believe education transforms lives.",
] + [
    "I love cooking pasta with garlic and olive oil.",
    "My favorite movie is about space exploration.",
    "Basketball is the best sport in the world.",
    "The weather has been unusually warm this winter.",
    "I think dogs are better pets than cats.",
] * 2


def test_scorer_importable():
    from evaluation.testing.scorers.behavioral.coherence_decay import CoherenceDecayScorer
    assert CoherenceDecayScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.behavioral.coherence_decay import CoherenceDecayScorer
    s = CoherenceDecayScorer()
    assert s.dimension_id == "D34"
    assert s.tier == 5
    assert s.requires_set is False


@pytest.mark.slow
def test_consistent_passes():
    """Consistent conversation should show no decay and pass."""
    from evaluation.testing.scorers.behavioral.coherence_decay import CoherenceDecayScorer
    scorer = CoherenceDecayScorer(window_size=5)
    ctx = _ctx_with_turns(CONSISTENT_TURNS)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    assert result.passed is True
    assert result.score >= 0.5
    assert "decay_magnitude" in result.details


@pytest.mark.slow
def test_decaying_fails():
    """Conversation that drifts off-topic should show decay and fail."""
    from evaluation.testing.scorers.behavioral.coherence_decay import CoherenceDecayScorer
    scorer = CoherenceDecayScorer(window_size=5)
    ctx = _ctx_with_turns(DECAYING_TURNS)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    assert result.passed is False
    assert result.details["decay_magnitude"] > 0.0


def test_too_few_turns_skips():
    from evaluation.testing.scorers.behavioral.coherence_decay import CoherenceDecayScorer
    scorer = CoherenceDecayScorer(window_size=10)
    ctx = _ctx_with_turns(["Short conversation"] * 5)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    assert result.details.get("skipped") is True


@pytest.mark.slow
def test_details_present():
    from evaluation.testing.scorers.behavioral.coherence_decay import CoherenceDecayScorer
    scorer = CoherenceDecayScorer(window_size=5)
    ctx = _ctx_with_turns(CONSISTENT_TURNS)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    for key in ("decay_magnitude", "window_scores", "first_half_avg", "second_half_avg"):
        assert key in result.details, f"Missing key: {key}"
