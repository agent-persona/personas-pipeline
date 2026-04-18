"""Tests for D5 Behavioral Consistency scorer."""

import pytest
from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext


def test_scorer_importable():
    from evaluation.testing.scorers.semantic.behavioral_consistency import BehavioralConsistencyScorer
    assert BehavioralConsistencyScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.semantic.behavioral_consistency import BehavioralConsistencyScorer
    s = BehavioralConsistencyScorer()
    assert s.dimension_id == "D5"
    assert s.tier == 2
    assert s.requires_set is False


@pytest.mark.slow
def test_consistent_responses_score_well():
    from evaluation.testing.scorers.semantic.behavioral_consistency import BehavioralConsistencyScorer
    scorer = BehavioralConsistencyScorer()
    persona = Persona(id="test-1", name="Test")
    ctx = SourceContext(id="s1", text="test", extra_data={
        "responses": [
            "I prefer data-driven approaches to decision making.",
            "I like to base my decisions on solid data and metrics.",
            "Data and evidence guide my decision-making process.",
            "I rely on analytics and data when making key decisions.",
            "My decision making is rooted in quantitative evidence.",
        ],
    })
    result = scorer.score(persona, ctx)
    assert result.score >= 0.5
    assert "centroid_radius" in result.details


@pytest.mark.slow
def test_inconsistent_responses_score_poorly():
    from evaluation.testing.scorers.semantic.behavioral_consistency import BehavioralConsistencyScorer
    scorer = BehavioralConsistencyScorer()
    persona = Persona(id="test-1", name="Test")
    ctx = SourceContext(id="s1", text="test", extra_data={
        "responses": [
            "I love eating pizza with my family.",
            "Quantum mechanics is the foundation of modern physics.",
            "The stock market crashed yesterday.",
            "I prefer hiking in the mountains.",
            "Java is my favorite programming language.",
        ],
    })
    result = scorer.score(persona, ctx)
    assert result.score < 0.7


def test_empty_responses_skips():
    from evaluation.testing.scorers.semantic.behavioral_consistency import BehavioralConsistencyScorer
    scorer = BehavioralConsistencyScorer()
    persona = Persona(id="test-1", name="Test")
    ctx = SourceContext(id="s1", text="test")
    result = scorer.score(persona, ctx)
    assert result.details.get("skipped") is True
