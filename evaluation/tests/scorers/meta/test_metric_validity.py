"""Tests for M3 Evaluation Metric Validity scorer."""

import pytest
from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext


CTX = SourceContext(id="s1", text="meta test")


def _make_persona(pid: str = "p1") -> Persona:
    return Persona(id=pid, name="Alice", occupation="analyst")


def test_scorer_importable():
    from evaluation.testing.scorers.meta.metric_validity import MetricValidityScorer
    assert MetricValidityScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.meta.metric_validity import MetricValidityScorer
    s = MetricValidityScorer()
    assert s.dimension_id == "M3"
    assert s.tier == 8
    assert s.requires_set is False


def test_valid_metric_passes():
    from evaluation.testing.scorers.meta.metric_validity import MetricValidityScorer
    scorer = MetricValidityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "metric_scores": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        "human_ratings": [0.88, 0.79, 0.72, 0.61, 0.48, 0.42, 0.31],
        "metric_name": "D1_schema_compliance",
    })
    result = scorer.score(p, ctx)
    assert result.passed is True
    assert result.score >= 0.8
    assert "correlation" in result.details


def test_invalid_metric_fails():
    from evaluation.testing.scorers.meta.metric_validity import MetricValidityScorer
    scorer = MetricValidityScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "metric_scores": [0.9, 0.8, 0.7, 0.6, 0.5],
        "human_ratings": [0.2, 0.9, 0.1, 0.8, 0.3],
        "metric_name": "D12_narrative",
    })
    result = scorer.score(p, ctx)
    assert result.passed is False


def test_no_data_skips():
    from evaluation.testing.scorers.meta.metric_validity import MetricValidityScorer
    scorer = MetricValidityScorer()
    p = _make_persona()
    result = scorer.score(p, CTX)
    assert result.details.get("skipped") is True


def test_sensitivity_detected():
    from evaluation.testing.scorers.meta.metric_validity import MetricValidityScorer
    scorer = MetricValidityScorer()
    p = _make_persona()
    # Metric changes when quality changes = sensitive
    ctx = SourceContext(id="s1", text="test", extra_data={
        "metric_scores": [0.9, 0.1, 0.8, 0.2, 0.7],
        "human_ratings": [0.85, 0.15, 0.75, 0.25, 0.65],
        "metric_name": "test_metric",
    })
    result = scorer.score(p, ctx)
    assert result.details["sensitive"] is True


def test_insensitive_metric_flagged():
    from evaluation.testing.scorers.meta.metric_validity import MetricValidityScorer
    scorer = MetricValidityScorer()
    p = _make_persona()
    # Metric doesn't change even when quality varies
    ctx = SourceContext(id="s1", text="test", extra_data={
        "metric_scores": [0.5, 0.5, 0.5, 0.5, 0.5],
        "human_ratings": [0.9, 0.1, 0.8, 0.2, 0.7],
        "metric_name": "insensitive_metric",
    })
    result = scorer.score(p, ctx)
    assert result.details["sensitive"] is False
    assert result.passed is False
