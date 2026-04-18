"""Tests for D32-D33 Engagement scorer (response diversity + consistency tradeoff)."""

from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext


def _ctx_with_turns(turns: list[str]) -> SourceContext:
    return SourceContext(id="s1", text="engagement test", extra_data={"conversation_turns": turns})


SAMPLE_PERSONA = Persona(id="p1", name="Alice")

# Diverse, engaging responses (varied vocabulary)
DIVERSE_TURNS = [
    "I believe education transforms communities through empowerment.",
    "Technology offers fascinating possibilities for healthcare innovation.",
    "Traveling abroad broadens our cultural perspectives enormously.",
    "Music provides therapeutic benefits for mental wellness.",
    "Sustainable agriculture is critical for environmental preservation.",
]

# Repetitive, monotone responses (low lexical diversity)
MONOTONE_TURNS = [
    "I think that is good. That is good.",
    "I think that is good. That is good.",
    "I think that is good. That is good.",
    "I think that is good. That is good.",
    "I think that is good. That is good.",
]


def test_scorer_importable():
    from evaluation.testing.scorers.behavioral.engagement import EngagementScorer
    assert EngagementScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.behavioral.engagement import EngagementScorer
    s = EngagementScorer()
    assert s.dimension_id == "D32-D33"
    assert s.tier == 5
    assert s.requires_set is False


def test_diverse_passes():
    """Engaging, diverse responses should pass."""
    from evaluation.testing.scorers.behavioral.engagement import EngagementScorer
    scorer = EngagementScorer()
    ctx = _ctx_with_turns(DIVERSE_TURNS)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    assert result.passed is True
    assert result.score >= 0.5
    assert "lexical_diversity" in result.details


def test_monotone_fails():
    """Repetitive, monotone responses should fail."""
    from evaluation.testing.scorers.behavioral.engagement import EngagementScorer
    scorer = EngagementScorer()
    ctx = _ctx_with_turns(MONOTONE_TURNS)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    assert result.passed is False
    assert result.details["lexical_diversity"] < 0.5


def test_no_turns_skips():
    from evaluation.testing.scorers.behavioral.engagement import EngagementScorer
    scorer = EngagementScorer()
    ctx = SourceContext(id="s1", text="no turns")
    result = scorer.score(SAMPLE_PERSONA, ctx)
    assert result.details.get("skipped") is True


def test_details_present():
    from evaluation.testing.scorers.behavioral.engagement import EngagementScorer
    scorer = EngagementScorer()
    ctx = _ctx_with_turns(DIVERSE_TURNS)
    result = scorer.score(SAMPLE_PERSONA, ctx)
    for key in ("lexical_diversity", "unique_words", "total_words", "turn_count"):
        assert key in result.details, f"Missing key: {key}"
