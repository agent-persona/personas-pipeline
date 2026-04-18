"""Tests for D11 Profile Coverage scorer."""

from evaluation.testing.schemas import Persona, CommunicationStyle
from evaluation.testing.source_context import SourceContext


def test_scorer_importable():
    from evaluation.testing.scorers.semantic.profile_coverage import ProfileCoverageScorer
    assert ProfileCoverageScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.semantic.profile_coverage import ProfileCoverageScorer
    s = ProfileCoverageScorer()
    assert s.dimension_id == "D11"
    assert s.tier == 2


def test_good_coverage_passes():
    from evaluation.testing.scorers.semantic.profile_coverage import ProfileCoverageScorer
    scorer = ProfileCoverageScorer()
    persona = Persona(
        id="p1", name="Alice", occupation="Product Manager",
        industry="SaaS", education="Master's degree", location="San Francisco",
        communication_style=CommunicationStyle(tone="professional but warm"),
    )
    ctx = SourceContext(id="s1", text="test", extra_data={
        "conversation": (
            "As a Product Manager in the SaaS industry, I work from San Francisco. "
            "I have a Master's degree and try to keep a professional but warm tone."
        ),
    })
    result = scorer.score(persona, ctx)
    assert result.passed is True
    assert result.score >= 0.6
    assert result.details["mentioned"] >= 3


def test_poor_coverage_fails():
    from evaluation.testing.scorers.semantic.profile_coverage import ProfileCoverageScorer
    scorer = ProfileCoverageScorer()
    persona = Persona(
        id="p2", name="Bob", occupation="Engineer",
        industry="Fintech", education="PhD", location="New York",
        lifestyle="urban commuter",
    )
    ctx = SourceContext(id="s1", text="test", extra_data={
        "conversation": "I like coffee and reading books.",
    })
    result = scorer.score(persona, ctx)
    assert result.passed is False
    assert result.score < 0.3


def test_no_conversation_skips():
    from evaluation.testing.scorers.semantic.profile_coverage import ProfileCoverageScorer
    scorer = ProfileCoverageScorer()
    persona = Persona(id="p3", name="Nobody", occupation="PM")
    ctx = SourceContext(id="s1", text="test")
    result = scorer.score(persona, ctx)
    assert result.details.get("skipped") is True


def test_minimal_persona_skips():
    from evaluation.testing.scorers.semantic.profile_coverage import ProfileCoverageScorer
    scorer = ProfileCoverageScorer()
    persona = Persona(id="p4", name="Minimal")
    ctx = SourceContext(id="s1", text="test", extra_data={
        "conversation": "Some conversation here.",
    })
    result = scorer.score(persona, ctx)
    assert result.details.get("skipped") is True
