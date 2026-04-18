"""Tests for D42 Generation Bias Amplification scorer."""

import pytest
from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext


CTX = SourceContext(id="s1", text="generation test")


def _make_persona(pid: str = "p1") -> Persona:
    return Persona(id=pid, name="Alice", occupation="analyst")


def test_scorer_importable():
    from evaluation.testing.scorers.generation.generation_bias import GenerationBiasAmplificationScorer
    assert GenerationBiasAmplificationScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.generation.generation_bias import GenerationBiasAmplificationScorer
    s = GenerationBiasAmplificationScorer()
    assert s.dimension_id == "D42"
    assert s.tier == 7
    assert s.requires_set is False


def test_monotonic_degradation_detected():
    from evaluation.testing.scorers.generation.generation_bias import GenerationBiasAmplificationScorer
    scorer = GenerationBiasAmplificationScorer()
    p = _make_persona()
    # Quality decreases as LLM involvement increases — expected pattern
    ctx = SourceContext(id="s1", text="test", extra_data={
        "ablation_scores": {
            "meta": 0.95,
            "objective_tabular": 0.85,
            "subjective_tabular": 0.70,
            "descriptive": 0.50,
        }
    })
    result = scorer.score(p, ctx)
    assert result.passed is True  # Monotonic = expected behavior
    assert result.details["monotonic_degradation"] is True
    assert result.details["total_degradation"] == pytest.approx(0.45, abs=0.01)


def test_non_monotonic_flags_anomaly():
    from evaluation.testing.scorers.generation.generation_bias import GenerationBiasAmplificationScorer
    scorer = GenerationBiasAmplificationScorer()
    p = _make_persona()
    # Non-monotonic: descriptive is better than subjective — unexpected
    ctx = SourceContext(id="s1", text="test", extra_data={
        "ablation_scores": {
            "meta": 0.95,
            "objective_tabular": 0.85,
            "subjective_tabular": 0.50,
            "descriptive": 0.80,
        }
    })
    result = scorer.score(p, ctx)
    assert result.passed is False  # Non-monotonic should fail
    assert result.details["monotonic_degradation"] is False


def test_severe_degradation_fails():
    from evaluation.testing.scorers.generation.generation_bias import GenerationBiasAmplificationScorer
    scorer = GenerationBiasAmplificationScorer()
    p = _make_persona()
    # Massive quality drop
    ctx = SourceContext(id="s1", text="test", extra_data={
        "ablation_scores": {
            "meta": 0.95,
            "objective_tabular": 0.30,
            "subjective_tabular": 0.10,
            "descriptive": 0.05,
        }
    })
    result = scorer.score(p, ctx)
    assert result.passed is False
    assert result.score < 0.5


def test_no_ablation_data_skips():
    from evaluation.testing.scorers.generation.generation_bias import GenerationBiasAmplificationScorer
    scorer = GenerationBiasAmplificationScorer()
    p = _make_persona()
    result = scorer.score(p, CTX)
    assert result.details.get("skipped") is True


def test_single_level_skips():
    from evaluation.testing.scorers.generation.generation_bias import GenerationBiasAmplificationScorer
    scorer = GenerationBiasAmplificationScorer()
    p = _make_persona()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "ablation_scores": {"meta": 0.95}
    })
    result = scorer.score(p, ctx)
    assert result.details.get("skipped") is True


def test_stable_quality_passes():
    from evaluation.testing.scorers.generation.generation_bias import GenerationBiasAmplificationScorer
    scorer = GenerationBiasAmplificationScorer()
    p = _make_persona()
    # Quality stays high across all levels
    ctx = SourceContext(id="s1", text="test", extra_data={
        "ablation_scores": {
            "meta": 0.92,
            "objective_tabular": 0.90,
            "subjective_tabular": 0.88,
            "descriptive": 0.85,
        }
    })
    result = scorer.score(p, ctx)
    assert result.passed is True
    assert result.score >= 0.8
