"""Tests for D19 RLHF Positivity Bias scorer."""

from persona_eval.schemas import Persona, EmotionalProfile
from persona_eval.source_context import SourceContext

CTX = SourceContext(id="s1", text="bias test")


def _make_persona(i: int, mood: str = "happy", pain_points: list[str] | None = None) -> Persona:
    return Persona(
        id=f"p{i}",
        name=f"Person {i}",
        emotional_profile=EmotionalProfile(baseline_mood=mood),
        pain_points=pain_points or [],
        goals=["be successful", "thrive"],
        bio="A driven and passionate professional who loves their work.",
    )


def test_scorer_importable():
    from persona_eval.scorers.bias.positivity_bias import PositivityBiasScorer
    assert PositivityBiasScorer is not None


def test_scorer_metadata():
    from persona_eval.scorers.bias.positivity_bias import PositivityBiasScorer
    s = PositivityBiasScorer()
    assert s.dimension_id == "D19"
    assert s.dimension_name == "RLHF Positivity Bias"
    assert s.tier == 4
    assert s.requires_set is True


def test_single_persona_score_skips():
    from persona_eval.scorers.bias.positivity_bias import PositivityBiasScorer
    scorer = PositivityBiasScorer()
    p = _make_persona(0)
    result = scorer.score(p, CTX)
    assert result.details.get("skipped") is True


def test_balanced_set_passes():
    """A set with mixed positive/negative should pass."""
    from persona_eval.scorers.bias.positivity_bias import PositivityBiasScorer
    scorer = PositivityBiasScorer()
    personas = []
    # Half: positive mood, no challenges
    for i in range(5):
        personas.append(_make_persona(i, mood="happy", pain_points=[]))
    # Half: negative mood, with struggles
    for i in range(5, 10):
        personas.append(_make_persona(
            i, mood="stressed",
            pain_points=["struggling with debt", "anxious about the future"],
        ))
    ctxs = [CTX] * len(personas)
    results = scorer.score_set(personas, ctxs)
    assert len(results) == 1
    result = results[0]
    assert result.passed is True
    assert result.score > 0.5
    assert "positivity_ratio" in result.details
    assert "challenge_rate" in result.details


def test_uniformly_positive_set_fails():
    """A set where every persona is uniformly positive should fail."""
    from persona_eval.scorers.bias.positivity_bias import PositivityBiasScorer
    scorer = PositivityBiasScorer()
    personas = [
        Persona(
            id=f"p{i}",
            name=f"Happy {i}",
            emotional_profile=EmotionalProfile(baseline_mood="excited"),
            bio="Amazing, wonderful, passionate, thriving person who loves life!",
            goals=["excel", "be successful", "thrive"],
            pain_points=[],
        )
        for i in range(10)
    ]
    ctxs = [CTX] * len(personas)
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert result.passed is False
    assert result.details["positivity_ratio"] >= 0.8


def test_empty_set_skips():
    """Empty persona list should skip."""
    from persona_eval.scorers.bias.positivity_bias import PositivityBiasScorer
    scorer = PositivityBiasScorer()
    results = scorer.score_set([], [])
    assert results[0].details.get("skipped") is True


def test_details_present():
    """Result details must include all expected keys."""
    from persona_eval.scorers.bias.positivity_bias import PositivityBiasScorer
    scorer = PositivityBiasScorer()
    personas = [_make_persona(i) for i in range(5)]
    ctxs = [CTX] * len(personas)
    results = scorer.score_set(personas, ctxs)
    d = results[0].details
    for key in ("positivity_ratio", "positive_markers", "negative_markers",
                "challenge_rate", "personas_with_challenges", "persona_count"):
        assert key in d, f"Missing key: {key}"
