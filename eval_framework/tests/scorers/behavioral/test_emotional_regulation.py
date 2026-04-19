"""Tests for D25 Emotional Self-Regulation scorer."""

from persona_eval.schemas import Persona, EmotionalProfile
from persona_eval.source_context import SourceContext


def _ctx_with_turns(turns: list[str]) -> SourceContext:
    return SourceContext(id="s1", text="emotional test", extra_data={"conversation_turns": turns})


CALM_PERSONA = Persona(
    id="p1", name="Alice",
    emotional_profile=EmotionalProfile(baseline_mood="calm"),
)

# Consistent calm responses
CALM_TURNS = [
    "I think we should consider all options carefully.",
    "That's an interesting perspective, let me think about it.",
    "I appreciate your input on this matter.",
    "Let's approach this step by step.",
    "I understand your concern and here's what I suggest.",
]

# Erratic emotional responses (anger, sadness, euphoria mixed randomly)
ERRATIC_TURNS = [
    "This is absolutely OUTRAGEOUS! I can't believe this!",
    "I'm so depressed about everything, nothing matters anymore.",
    "WOW this is the BEST thing EVER! I'm SO EXCITED!!!",
    "I HATE this, it makes me furious and angry!",
    "Everything is wonderful and perfect and amazing!",
]


def test_scorer_importable():
    from persona_eval.scorers.behavioral.emotional_regulation import EmotionalRegulationScorer
    assert EmotionalRegulationScorer is not None


def test_scorer_metadata():
    from persona_eval.scorers.behavioral.emotional_regulation import EmotionalRegulationScorer
    s = EmotionalRegulationScorer()
    assert s.dimension_id == "D25"
    assert s.tier == 5
    assert s.requires_set is False


def test_calm_persona_consistent_passes():
    """Calm persona with calm responses should pass."""
    from persona_eval.scorers.behavioral.emotional_regulation import EmotionalRegulationScorer
    scorer = EmotionalRegulationScorer()
    ctx = _ctx_with_turns(CALM_TURNS)
    result = scorer.score(CALM_PERSONA, ctx)
    assert result.passed is True
    assert result.score >= 0.5
    assert "emotion_variance" in result.details


def test_calm_persona_erratic_fails():
    """Calm persona with erratic responses should fail."""
    from persona_eval.scorers.behavioral.emotional_regulation import EmotionalRegulationScorer
    scorer = EmotionalRegulationScorer()
    ctx = _ctx_with_turns(ERRATIC_TURNS)
    result = scorer.score(CALM_PERSONA, ctx)
    assert result.passed is False
    assert result.score < 0.7


def test_no_turns_skips():
    from persona_eval.scorers.behavioral.emotional_regulation import EmotionalRegulationScorer
    scorer = EmotionalRegulationScorer()
    ctx = SourceContext(id="s1", text="no turns")
    result = scorer.score(CALM_PERSONA, ctx)
    assert result.details.get("skipped") is True


def test_details_present():
    from persona_eval.scorers.behavioral.emotional_regulation import EmotionalRegulationScorer
    scorer = EmotionalRegulationScorer()
    ctx = _ctx_with_turns(CALM_TURNS)
    result = scorer.score(CALM_PERSONA, ctx)
    for key in ("emotion_variance", "dominant_emotion", "turn_count"):
        assert key in result.details, f"Missing key: {key}"
