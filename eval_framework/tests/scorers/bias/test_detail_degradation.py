"""Tests for Detail Degradation scorer."""

import pytest
from persona_eval.schemas import Persona
from persona_eval.source_context import SourceContext
from tests.fixtures.persona_set import generate_test_persona_set


CTX = SourceContext(id="s1", text="test")


def test_scorer_importable():
    from persona_eval.scorers.bias.detail_degradation import DetailDegradationScorer
    assert DetailDegradationScorer is not None


def test_scorer_metadata():
    from persona_eval.scorers.bias.detail_degradation import DetailDegradationScorer
    s = DetailDegradationScorer()
    assert s.dimension_id == "D24b"
    assert s.tier == 4
    assert s.requires_set is True


def test_monotonic_degradation_detected():
    """Accuracy decreases with more detail → degradation detected."""
    from persona_eval.scorers.bias.detail_degradation import DetailDegradationScorer
    scorer = DetailDegradationScorer()
    personas = generate_test_persona_set(n=5, seed=42)
    ctxs = [SourceContext(id=f"s{i}", text="test", extra_data={
        "detail_level_results": [
            {"detail_level": 1, "label": "demographic_only", "accuracy": 0.65},
            {"detail_level": 2, "label": "plus_backstory", "accuracy": 0.55},
            {"detail_level": 3, "label": "full_llm_generated", "accuracy": 0.40},
        ]
    }) for i in range(5)]
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert result.passed is False
    assert result.details["is_monotonically_decreasing"] is True
    assert result.details["degradation_rate"] > 0.0


def test_stable_accuracy_passes():
    """Accuracy stays stable or improves → no degradation."""
    from persona_eval.scorers.bias.detail_degradation import DetailDegradationScorer
    scorer = DetailDegradationScorer()
    personas = generate_test_persona_set(n=5, seed=42)
    ctxs = [SourceContext(id=f"s{i}", text="test", extra_data={
        "detail_level_results": [
            {"detail_level": 1, "label": "demographic_only", "accuracy": 0.55},
            {"detail_level": 2, "label": "plus_backstory", "accuracy": 0.60},
            {"detail_level": 3, "label": "full_llm_generated", "accuracy": 0.62},
        ]
    }) for i in range(5)]
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert result.passed is True
    assert result.details["is_monotonically_decreasing"] is False


def test_skipped_when_no_data():
    """No detail_level_results → skipped."""
    from persona_eval.scorers.bias.detail_degradation import DetailDegradationScorer
    scorer = DetailDegradationScorer()
    personas = generate_test_persona_set(n=5, seed=42)
    ctxs = [CTX] * 5
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert result.details.get("skipped") is True
