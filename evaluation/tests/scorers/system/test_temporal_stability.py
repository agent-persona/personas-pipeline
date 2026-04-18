"""Tests for D37 Temporal Stability scorer."""

import pytest
from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext


CTX = SourceContext(id="s1", text="system test")


def _make_persona(pid: str = "p1") -> Persona:
    return Persona(id=pid, name="Alice", occupation="analyst", bio="Data analyst at a tech firm.")


def test_scorer_importable():
    from evaluation.testing.scorers.system.temporal_stability import TemporalStabilityScorer
    assert TemporalStabilityScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.system.temporal_stability import TemporalStabilityScorer
    s = TemporalStabilityScorer()
    assert s.dimension_id == "D37"
    assert s.tier == 6
    assert s.requires_set is False


@pytest.mark.slow
def test_stable_persona_passes():
    from evaluation.testing.scorers.system.temporal_stability import TemporalStabilityScorer
    scorer = TemporalStabilityScorer()
    p = _make_persona()
    # Baseline and current are semantically identical
    ctx = SourceContext(id="s1", text="test", extra_data={
        "baseline_text": "Alice is a data analyst at a tech company. She analyzes data trends.",
        "current_text": "Alice is a data analyst at a tech firm. She works on data analysis.",
    })
    result = scorer.score(p, ctx)
    assert result.passed is True
    assert result.score >= 0.7
    assert "semantic_similarity" in result.details


@pytest.mark.slow
def test_drifted_persona_fails():
    from evaluation.testing.scorers.system.temporal_stability import TemporalStabilityScorer
    scorer = TemporalStabilityScorer()
    p = _make_persona()
    # Baseline and current are semantically very different
    ctx = SourceContext(id="s1", text="test", extra_data={
        "baseline_text": "Alice is a data analyst at a tech company. She analyzes market trends and user behavior.",
        "current_text": "Bob is a professional chef who specializes in French cuisine and runs a restaurant in Paris.",
    })
    result = scorer.score(p, ctx)
    assert result.passed is False
    assert result.score < 0.5


def test_no_baseline_skips():
    from evaluation.testing.scorers.system.temporal_stability import TemporalStabilityScorer
    scorer = TemporalStabilityScorer()
    p = _make_persona()
    result = scorer.score(p, CTX)
    assert result.details.get("skipped") is True


@pytest.mark.slow
def test_psi_computed_when_distributions_provided():
    from evaluation.testing.scorers.system.temporal_stability import TemporalStabilityScorer
    scorer = TemporalStabilityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "baseline_text": "Alice is a data analyst.",
        "current_text": "Alice is a data analyst.",
        "baseline_distribution": [0.3, 0.3, 0.2, 0.2],
        "current_distribution": [0.3, 0.3, 0.2, 0.2],
    })
    result = scorer.score(p, ctx)
    assert result.passed is True
    assert "psi" in result.details
    assert result.details["psi"] < 0.1  # Identical distributions


@pytest.mark.slow
def test_psi_high_drift_detected():
    from evaluation.testing.scorers.system.temporal_stability import TemporalStabilityScorer
    scorer = TemporalStabilityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "baseline_text": "Alice is a data analyst.",
        "current_text": "Alice is a data analyst.",
        "baseline_distribution": [0.9, 0.05, 0.025, 0.025],
        "current_distribution": [0.1, 0.1, 0.4, 0.4],
    })
    result = scorer.score(p, ctx)
    assert result.details["psi"] > 0.2  # Significant shift
    assert result.details.get("psi_alert") is True
