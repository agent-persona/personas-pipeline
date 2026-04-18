"""Tests for D38 Cross-Model Stability scorer."""

import pytest
from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext


CTX = SourceContext(id="s1", text="system test")


def _make_persona(pid: str = "p1") -> Persona:
    return Persona(id=pid, name="Alice", occupation="analyst")


def test_scorer_importable():
    from evaluation.testing.scorers.system.cross_model_stability import CrossModelStabilityScorer
    assert CrossModelStabilityScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.system.cross_model_stability import CrossModelStabilityScorer
    s = CrossModelStabilityScorer()
    assert s.dimension_id == "D38"
    assert s.tier == 6
    assert s.requires_set is False


def test_stable_across_models_passes():
    from evaluation.testing.scorers.system.cross_model_stability import CrossModelStabilityScorer
    scorer = CrossModelStabilityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "model_scores": {
            "sonnet": {"D1": 0.95, "D2": 0.90, "D3": 0.85},
            "opus": {"D1": 0.93, "D2": 0.88, "D3": 0.87},
            "haiku": {"D1": 0.92, "D2": 0.89, "D3": 0.84},
        }
    })
    result = scorer.score(p, ctx)
    assert result.passed is True
    assert result.score >= 0.8
    assert "max_regression" in result.details


def test_regression_detected():
    from evaluation.testing.scorers.system.cross_model_stability import CrossModelStabilityScorer
    scorer = CrossModelStabilityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "model_scores": {
            "baseline_model": {"D1": 0.95, "D2": 0.90, "D3": 0.85},
            "new_model": {"D1": 0.50, "D2": 0.40, "D3": 0.30},
        }
    })
    result = scorer.score(p, ctx)
    assert result.passed is False
    assert result.score < 0.7


def test_no_model_scores_skips():
    from evaluation.testing.scorers.system.cross_model_stability import CrossModelStabilityScorer
    scorer = CrossModelStabilityScorer()
    p = _make_persona()
    result = scorer.score(p, CTX)
    assert result.details.get("skipped") is True


def test_single_model_skips():
    from evaluation.testing.scorers.system.cross_model_stability import CrossModelStabilityScorer
    scorer = CrossModelStabilityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "model_scores": {
            "sonnet": {"D1": 0.95, "D2": 0.90},
        }
    })
    result = scorer.score(p, ctx)
    assert result.details.get("skipped") is True


def test_dimension_level_regressions_reported():
    from evaluation.testing.scorers.system.cross_model_stability import CrossModelStabilityScorer
    scorer = CrossModelStabilityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "model_scores": {
            "model_a": {"D1": 0.95, "D2": 0.90},
            "model_b": {"D1": 0.50, "D2": 0.92},
        }
    })
    result = scorer.score(p, ctx)
    assert "dimension_ranges" in result.details
