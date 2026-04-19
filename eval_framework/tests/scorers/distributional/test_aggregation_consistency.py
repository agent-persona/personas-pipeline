"""Tests for D15 Structural Aggregation Consistency scorer."""

import pytest
from persona_eval.schemas import Persona, CommunicationStyle, EmotionalProfile
from persona_eval.source_context import SourceContext
from tests.fixtures.persona_set import generate_test_persona_set


CTX = SourceContext(id="s1", text="distributional test")


@pytest.fixture
def diverse_personas():
    return generate_test_persona_set(n=50, seed=42)


def _make_skewed_set() -> list[Persona]:
    """Create a set with strong demographic-opinion correlation (inconsistency).

    All females -> Healthcare industry, all males -> SaaS industry.
    This should produce low cross-group consistency for industry distribution.
    """
    personas = []
    for i in range(25):
        personas.append(Persona(
            id=f"skew-f-{i}",
            name=f"Female {i}",
            age=30,
            gender="female",
            occupation="Analyst",
            industry="Healthcare",
            lifestyle="urban commuter",
            education="Bachelor's degree",
            bio="An analyst with a strong background in healthcare data and patient outcomes research.",
        ))
    for i in range(25):
        personas.append(Persona(
            id=f"skew-m-{i}",
            name=f"Male {i}",
            age=30,
            gender="male",
            occupation="Engineer",
            industry="SaaS",
            lifestyle="remote-first",
            education="Master's degree",
            bio="A software engineer working in SaaS with experience across multiple cloud platforms.",
        ))
    return personas


def test_scorer_importable():
    from persona_eval.scorers.distributional.aggregation_consistency import AggregationConsistencyScorer
    assert AggregationConsistencyScorer is not None


def test_scorer_metadata():
    from persona_eval.scorers.distributional.aggregation_consistency import AggregationConsistencyScorer
    s = AggregationConsistencyScorer()
    assert s.dimension_id == "D15"
    assert s.tier == 3
    assert s.requires_set is True


def test_diverse_set_consistent(diverse_personas):
    from persona_eval.scorers.distributional.aggregation_consistency import AggregationConsistencyScorer
    scorer = AggregationConsistencyScorer()
    ctxs = [CTX] * len(diverse_personas)
    results = scorer.score_set(diverse_personas, ctxs)
    result = results[0]
    assert result.score >= 0.5
    assert "consistency_ratio" in result.details


def test_group_comparisons_present(diverse_personas):
    from persona_eval.scorers.distributional.aggregation_consistency import AggregationConsistencyScorer
    scorer = AggregationConsistencyScorer()
    ctxs = [CTX] * len(diverse_personas)
    results = scorer.score_set(diverse_personas, ctxs)
    result = results[0]
    assert "group_comparisons" in result.details
    assert "split_half_scores" in result.details


def test_skewed_set_detects_inconsistency():
    """A set where gender perfectly predicts industry should score lower."""
    from persona_eval.scorers.distributional.aggregation_consistency import AggregationConsistencyScorer
    scorer = AggregationConsistencyScorer()
    skewed = _make_skewed_set()
    ctxs = [CTX] * len(skewed)
    results = scorer.score_set(skewed, ctxs)
    result = results[0]
    # Cross-group consistency should be low because gender->industry is deterministic
    assert result.score < 0.9
    # Check that group comparisons detected the skew
    group_comps = result.details.get("group_comparisons", [])
    gender_industry = [g for g in group_comps if g["group_by"] == "gender" and g["field"] == "industry"]
    assert len(gender_industry) > 0
    # JSD-based consistency for perfectly skewed data is ~0.69 (JSD max = log2(2) = 1.0)
    assert gender_industry[0]["consistency"] < 0.75


def test_small_set_skipped():
    from persona_eval.scorers.distributional.aggregation_consistency import AggregationConsistencyScorer
    scorer = AggregationConsistencyScorer()
    small = generate_test_persona_set(n=5, seed=1)
    results = scorer.score_set(small, [CTX] * len(small))
    assert results[0].details.get("skipped") is True
