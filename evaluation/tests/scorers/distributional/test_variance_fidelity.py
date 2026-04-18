"""Tests for D14 Variance Fidelity scorer."""

import pytest
from evaluation.testing.source_context import SourceContext
from tests.fixtures.persona_set import generate_test_persona_set, generate_homogeneous_set


CTX = SourceContext(id="s1", text="distributional test")


@pytest.fixture
def diverse_personas():
    return generate_test_persona_set(n=50, seed=42)


@pytest.fixture
def homogeneous_personas():
    return generate_homogeneous_set(n=50)


def test_scorer_importable():
    from evaluation.testing.scorers.distributional.variance_fidelity import VarianceFidelityScorer
    assert VarianceFidelityScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.distributional.variance_fidelity import VarianceFidelityScorer
    s = VarianceFidelityScorer()
    assert s.dimension_id == "D14"
    assert s.tier == 3
    assert s.requires_set is True


def test_diverse_set_has_variance(diverse_personas):
    from evaluation.testing.scorers.distributional.variance_fidelity import VarianceFidelityScorer
    scorer = VarianceFidelityScorer()
    ctxs = [CTX] * len(diverse_personas)
    results = scorer.score_set(diverse_personas, ctxs)
    result = results[0]
    assert result.score >= 0.5
    assert "iqr_scores" in result.details


def test_homogeneous_set_zero_variance(homogeneous_personas):
    from evaluation.testing.scorers.distributional.variance_fidelity import VarianceFidelityScorer
    scorer = VarianceFidelityScorer()
    ctxs = [CTX] * len(homogeneous_personas)
    results = scorer.score_set(homogeneous_personas, ctxs)
    result = results[0]
    assert result.passed is False
    assert result.score < 0.3
    assert "zero_variance_fields" in result.details


def test_ks_test_with_reference(diverse_personas):
    from evaluation.testing.scorers.distributional.variance_fidelity import VarianceFidelityScorer
    scorer = VarianceFidelityScorer()
    ctxs = [CTX] * len(diverse_personas)
    ref = {"age": {"mean": 40, "std": 12}}
    results = scorer.score_set(diverse_personas, ctxs, reference_distributions=ref)
    result = results[0]
    assert "ks_tests" in result.details
