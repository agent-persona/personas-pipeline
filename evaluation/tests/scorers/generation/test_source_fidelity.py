"""Tests for D43 Source Data Fidelity scorer."""

import pytest
from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext


CTX = SourceContext(id="s1", text="generation test")


def _make_persona(pid: str = "p1") -> Persona:
    return Persona(id=pid, name="Alice", occupation="analyst")


def test_scorer_importable():
    from evaluation.testing.scorers.generation.source_fidelity import SourceDataFidelityScorer
    assert SourceDataFidelityScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.generation.source_fidelity import SourceDataFidelityScorer
    s = SourceDataFidelityScorer()
    assert s.dimension_id == "D43"
    assert s.tier == 7
    assert s.requires_set is False


@pytest.mark.slow
def test_high_fidelity_passes():
    from evaluation.testing.scorers.generation.source_fidelity import SourceDataFidelityScorer
    scorer = SourceDataFidelityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "source_facts": [
            "Alice is a data analyst",
            "She works at a technology company",
            "She has 8 years of experience",
        ],
        "persona_text": "Alice is a data analyst at a tech firm with 8 years of experience in data analytics.",
    })
    result = scorer.score(p, ctx)
    assert result.passed is True
    assert result.score >= 0.5
    assert "retention_rate" in result.details


@pytest.mark.slow
def test_low_fidelity_fails():
    from evaluation.testing.scorers.generation.source_fidelity import SourceDataFidelityScorer
    scorer = SourceDataFidelityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "source_facts": [
            "Alice is a data analyst",
            "She works at a technology company",
            "She has 8 years of experience",
            "She lives in San Francisco",
            "She manages a team of 5 people",
        ],
        "persona_text": "Bob is a professional chef who runs a restaurant in Paris.",
    })
    result = scorer.score(p, ctx)
    assert result.passed is False
    assert result.score < 0.5


def test_no_source_facts_skips():
    from evaluation.testing.scorers.generation.source_fidelity import SourceDataFidelityScorer
    scorer = SourceDataFidelityScorer()
    p = _make_persona()
    result = scorer.score(p, CTX)
    assert result.details.get("skipped") is True


@pytest.mark.slow
def test_partial_retention():
    from evaluation.testing.scorers.generation.source_fidelity import SourceDataFidelityScorer
    scorer = SourceDataFidelityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "source_facts": [
            "Alice is a data analyst",
            "She works at a technology company",
            "She enjoys rock climbing",
            "She has a pet snake named Monty",
        ],
        "persona_text": "Alice is a data analyst at a tech company. She enjoys various outdoor activities.",
    })
    result = scorer.score(p, ctx)
    # Some facts retained, some lost
    assert 0.2 <= result.score <= 0.9
    assert result.details["retained_count"] > 0
    assert result.details["total_facts"] == 4
