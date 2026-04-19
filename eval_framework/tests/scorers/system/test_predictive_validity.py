"""Tests for D36 Predictive Validity scorer."""

import pytest
from persona_eval.schemas import Persona
from persona_eval.source_context import SourceContext


CTX = SourceContext(id="s1", text="system test")


def _make_persona(pid: str = "p1") -> Persona:
    return Persona(id=pid, name="Alice", occupation="analyst")


def test_scorer_importable():
    from persona_eval.scorers.system.predictive_validity import PredictiveValidityScorer
    assert PredictiveValidityScorer is not None


def test_scorer_metadata():
    from persona_eval.scorers.system.predictive_validity import PredictiveValidityScorer
    s = PredictiveValidityScorer()
    assert s.dimension_id == "D36"
    assert s.tier == 6
    assert s.requires_set is False


def test_high_correlation_passes():
    from persona_eval.scorers.system.predictive_validity import PredictiveValidityScorer
    scorer = PredictiveValidityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "predictions": [
            {"predicted": 1.0, "actual": 1.1},
            {"predicted": 2.0, "actual": 2.2},
            {"predicted": 3.0, "actual": 2.9},
            {"predicted": 4.0, "actual": 4.1},
            {"predicted": 5.0, "actual": 5.0},
        ]
    })
    result = scorer.score(p, ctx)
    assert result.passed is True
    assert result.score >= 0.8
    assert "correlation" in result.details


def test_low_correlation_fails():
    from persona_eval.scorers.system.predictive_validity import PredictiveValidityScorer
    scorer = PredictiveValidityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "predictions": [
            {"predicted": 1.0, "actual": 5.0},
            {"predicted": 2.0, "actual": 1.0},
            {"predicted": 3.0, "actual": 4.0},
            {"predicted": 4.0, "actual": 2.0},
            {"predicted": 5.0, "actual": 3.0},
        ]
    })
    result = scorer.score(p, ctx)
    assert result.passed is False
    assert result.score < 0.5


def test_sign_flip_detected():
    from persona_eval.scorers.system.predictive_validity import PredictiveValidityScorer
    scorer = PredictiveValidityScorer()
    p = _make_persona()
    # Perfect negative correlation = sign flip
    ctx = SourceContext(id="s1", text="test", extra_data={
        "predictions": [
            {"predicted": 1.0, "actual": 5.0},
            {"predicted": 2.0, "actual": 4.0},
            {"predicted": 3.0, "actual": 3.0},
            {"predicted": 4.0, "actual": 2.0},
            {"predicted": 5.0, "actual": 1.0},
        ]
    })
    result = scorer.score(p, ctx)
    assert result.passed is False
    assert result.details["sign_flip"] is True


def test_no_predictions_skips():
    from persona_eval.scorers.system.predictive_validity import PredictiveValidityScorer
    scorer = PredictiveValidityScorer()
    p = _make_persona()
    result = scorer.score(p, CTX)
    assert result.details.get("skipped") is True


def test_too_few_predictions_skips():
    from persona_eval.scorers.system.predictive_validity import PredictiveValidityScorer
    scorer = PredictiveValidityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "predictions": [
            {"predicted": 1.0, "actual": 1.0},
            {"predicted": 2.0, "actual": 2.0},
        ]
    })
    result = scorer.score(p, ctx)
    assert result.details.get("skipped") is True


def test_constant_predictions_handled():
    from persona_eval.scorers.system.predictive_validity import PredictiveValidityScorer
    scorer = PredictiveValidityScorer()
    p = _make_persona()
    # All same predictions — correlation undefined
    ctx = SourceContext(id="s1", text="test", extra_data={
        "predictions": [
            {"predicted": 3.0, "actual": 1.0},
            {"predicted": 3.0, "actual": 2.0},
            {"predicted": 3.0, "actual": 3.0},
            {"predicted": 3.0, "actual": 4.0},
            {"predicted": 3.0, "actual": 5.0},
        ]
    })
    result = scorer.score(p, ctx)
    assert result.passed is False
    assert result.score == 0.0
