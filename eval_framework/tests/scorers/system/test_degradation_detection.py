"""Tests for D41 Degradation Detection scorer."""

import pytest
from persona_eval.schemas import Persona
from persona_eval.source_context import SourceContext


CTX = SourceContext(id="s1", text="system test")


def _make_persona(pid: str = "p1") -> Persona:
    return Persona(id=pid, name="Alice", occupation="analyst")


def test_scorer_importable():
    from persona_eval.scorers.system.degradation_detection import DegradationDetectionScorer
    assert DegradationDetectionScorer is not None


def test_scorer_metadata():
    from persona_eval.scorers.system.degradation_detection import DegradationDetectionScorer
    s = DegradationDetectionScorer()
    assert s.dimension_id == "D41"
    assert s.tier == 6
    assert s.requires_set is False


def test_stable_metrics_pass():
    from persona_eval.scorers.system.degradation_detection import DegradationDetectionScorer
    scorer = DegradationDetectionScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "historical_scores": [0.90, 0.91, 0.89, 0.90, 0.92, 0.91, 0.90, 0.89],
        "current_score": 0.90,
    })
    result = scorer.score(p, ctx)
    assert result.passed is True
    assert result.score >= 0.8
    assert "z_score" in result.details


def test_degraded_metrics_fail():
    from persona_eval.scorers.system.degradation_detection import DegradationDetectionScorer
    scorer = DegradationDetectionScorer()
    p = _make_persona()
    # Current score significantly below historical mean
    ctx = SourceContext(id="s1", text="test", extra_data={
        "historical_scores": [0.90, 0.91, 0.89, 0.90, 0.92, 0.91, 0.90, 0.89],
        "current_score": 0.50,
    })
    result = scorer.score(p, ctx)
    assert result.passed is False
    assert result.details["z_score"] < -1.0
    assert result.details["drift_detected"] is True


def test_no_historical_data_skips():
    from persona_eval.scorers.system.degradation_detection import DegradationDetectionScorer
    scorer = DegradationDetectionScorer()
    p = _make_persona()
    result = scorer.score(p, CTX)
    assert result.details.get("skipped") is True


def test_too_few_historical_skips():
    from persona_eval.scorers.system.degradation_detection import DegradationDetectionScorer
    scorer = DegradationDetectionScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "historical_scores": [0.90, 0.91],
        "current_score": 0.50,
    })
    result = scorer.score(p, ctx)
    assert result.details.get("skipped") is True


def test_step_function_drop_detected():
    from persona_eval.scorers.system.degradation_detection import DegradationDetectionScorer
    scorer = DegradationDetectionScorer()
    p = _make_persona()
    # Sudden drop from stable baseline
    ctx = SourceContext(id="s1", text="test", extra_data={
        "historical_scores": [0.95, 0.95, 0.95, 0.95, 0.95],
        "current_score": 0.60,
    })
    result = scorer.score(p, ctx)
    assert result.passed is False
    assert result.details["drift_detected"] is True


def test_improving_metrics_pass():
    from persona_eval.scorers.system.degradation_detection import DegradationDetectionScorer
    scorer = DegradationDetectionScorer()
    p = _make_persona()
    # Current score above historical mean = improvement, not degradation
    ctx = SourceContext(id="s1", text="test", extra_data={
        "historical_scores": [0.70, 0.72, 0.71, 0.73, 0.70],
        "current_score": 0.85,
    })
    result = scorer.score(p, ctx)
    assert result.passed is True


def test_constant_historical_handled():
    from persona_eval.scorers.system.degradation_detection import DegradationDetectionScorer
    scorer = DegradationDetectionScorer()
    p = _make_persona()
    # All identical historical scores (zero std dev)
    ctx = SourceContext(id="s1", text="test", extra_data={
        "historical_scores": [0.90, 0.90, 0.90, 0.90, 0.90],
        "current_score": 0.85,
    })
    result = scorer.score(p, ctx)
    # With zero std dev, any deviation should be detected
    assert result.passed is False
    assert result.details["drift_detected"] is True
