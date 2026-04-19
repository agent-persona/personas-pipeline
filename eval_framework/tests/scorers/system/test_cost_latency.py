"""Tests for D40 Cost/Latency Bounds scorer."""

import pytest
from persona_eval.schemas import Persona
from persona_eval.source_context import SourceContext


CTX = SourceContext(id="s1", text="system test")


def _make_persona(pid: str = "p1") -> Persona:
    return Persona(id=pid, name="Alice", occupation="analyst")


def test_scorer_importable():
    from persona_eval.scorers.system.cost_latency import CostLatencyScorer
    assert CostLatencyScorer is not None


def test_scorer_metadata():
    from persona_eval.scorers.system.cost_latency import CostLatencyScorer
    s = CostLatencyScorer()
    assert s.dimension_id == "D40"
    assert s.tier == 6
    assert s.requires_set is False


def test_within_budget_passes():
    from persona_eval.scorers.system.cost_latency import CostLatencyScorer
    scorer = CostLatencyScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "cost_usd": 0.05,
        "latency_seconds": 10.0,
        "token_counts": {"input_tokens": 500, "output_tokens": 1000},
    })
    result = scorer.score(p, ctx)
    assert result.passed is True
    assert result.score >= 0.9
    assert result.details["cost_within_budget"] is True
    assert result.details["latency_within_budget"] is True


def test_over_budget_fails():
    from persona_eval.scorers.system.cost_latency import CostLatencyScorer
    scorer = CostLatencyScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "cost_usd": 5.0,
        "latency_seconds": 120.0,
    })
    result = scorer.score(p, ctx)
    assert result.passed is False
    assert result.score < 0.5


def test_no_metrics_skips():
    from persona_eval.scorers.system.cost_latency import CostLatencyScorer
    scorer = CostLatencyScorer()
    p = _make_persona()
    result = scorer.score(p, CTX)
    assert result.details.get("skipped") is True


def test_custom_thresholds():
    from persona_eval.scorers.system.cost_latency import CostLatencyScorer
    scorer = CostLatencyScorer(max_cost_per_persona=0.01, max_latency_seconds=5.0)
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "cost_usd": 0.05,
        "latency_seconds": 10.0,
    })
    result = scorer.score(p, ctx)
    assert result.passed is False


def test_zero_cost_passes():
    from persona_eval.scorers.system.cost_latency import CostLatencyScorer
    scorer = CostLatencyScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "cost_usd": 0.0,
        "latency_seconds": 0.0,
    })
    result = scorer.score(p, ctx)
    assert result.passed is True
    assert result.score == 1.0


def test_token_counts_reported():
    from persona_eval.scorers.system.cost_latency import CostLatencyScorer
    scorer = CostLatencyScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "cost_usd": 0.1,
        "latency_seconds": 5.0,
        "token_counts": {"input_tokens": 1000, "output_tokens": 2000},
    })
    result = scorer.score(p, ctx)
    assert result.details["token_counts"] == {"input_tokens": 1000, "output_tokens": 2000}
