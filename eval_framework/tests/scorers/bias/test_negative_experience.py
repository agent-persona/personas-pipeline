"""Tests for D24 Negative Experience Representation scorer."""

from persona_eval.schemas import Persona, EmotionalProfile
from persona_eval.source_context import SourceContext

CTX = SourceContext(id="s1", text="negative experience test")


def _happy_persona(i: int) -> Persona:
    return Persona(
        id=f"ph{i}", name=f"Happy {i}",
        emotional_profile=EmotionalProfile(baseline_mood="cheerful"),
        bio="A successful and content person with a great life.",
        pain_points=[],
    )


def _adversity_persona(i: int) -> Persona:
    return Persona(
        id=f"pa{i}", name=f"Adversity {i}",
        emotional_profile=EmotionalProfile(
            baseline_mood="anxious",
            stress_triggers=["financial stress", "health concerns"],
        ),
        bio="Dealing with depression and unemployment after being laid off.",
        pain_points=["debt", "anxiety", "loneliness"],
    )


def test_scorer_importable():
    from persona_eval.scorers.bias.negative_experience import NegativeExperienceScorer
    assert NegativeExperienceScorer is not None


def test_scorer_metadata():
    from persona_eval.scorers.bias.negative_experience import NegativeExperienceScorer
    s = NegativeExperienceScorer()
    assert s.dimension_id == "D24"
    assert s.dimension_name == "Negative Experience Representation"
    assert s.tier == 4
    assert s.requires_set is True


def test_single_persona_skips():
    from persona_eval.scorers.bias.negative_experience import NegativeExperienceScorer
    scorer = NegativeExperienceScorer()
    result = scorer.score(_happy_persona(0), CTX)
    assert result.details.get("skipped") is True


def test_balanced_set_passes():
    """A set with ~20%+ negative experiences should pass."""
    from persona_eval.scorers.bias.negative_experience import NegativeExperienceScorer
    scorer = NegativeExperienceScorer()
    personas = [_happy_persona(i) for i in range(7)]
    personas.extend([_adversity_persona(i) for i in range(3)])  # 30%
    ctxs = [CTX] * len(personas)
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert result.passed is True
    assert result.score >= 0.5
    assert "adversity_rate" in result.details


def test_uniformly_happy_set_fails():
    """A set with no negative experiences should fail."""
    from persona_eval.scorers.bias.negative_experience import NegativeExperienceScorer
    scorer = NegativeExperienceScorer()
    personas = [_happy_persona(i) for i in range(10)]
    ctxs = [CTX] * len(personas)
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert result.passed is False
    assert result.details["adversity_rate"] < 0.15


def test_empty_set_skips():
    from persona_eval.scorers.bias.negative_experience import NegativeExperienceScorer
    scorer = NegativeExperienceScorer()
    results = scorer.score_set([], [])
    assert results[0].details.get("skipped") is True


def test_details_present():
    from persona_eval.scorers.bias.negative_experience import NegativeExperienceScorer
    scorer = NegativeExperienceScorer()
    personas = [_happy_persona(i) for i in range(5)]
    ctxs = [CTX] * len(personas)
    results = scorer.score_set(personas, ctxs)
    d = results[0].details
    for key in ("adversity_rate", "personas_with_adversity", "persona_count"):
        assert key in d, f"Missing key: {key}"
