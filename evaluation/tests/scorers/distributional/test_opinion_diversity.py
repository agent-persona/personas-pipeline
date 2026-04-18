"""Tests for D13 Opinion Diversity scorer."""

import random

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
    from evaluation.testing.scorers.distributional.opinion_diversity import OpinionDiversityScorer
    assert OpinionDiversityScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.distributional.opinion_diversity import OpinionDiversityScorer
    s = OpinionDiversityScorer()
    assert s.dimension_id == "D13"
    assert s.tier == 3
    assert s.requires_set is True


def test_diverse_set_passes(diverse_personas):
    from evaluation.testing.scorers.distributional.opinion_diversity import OpinionDiversityScorer
    scorer = OpinionDiversityScorer()
    ctxs = [CTX] * len(diverse_personas)
    results = scorer.score_set(diverse_personas, ctxs)
    assert len(results) == 1
    result = results[0]
    assert result.passed is True
    assert result.score >= 0.7
    assert "entropies" in result.details or "variation_ratios" in result.details


def test_homogeneous_set_fails(homogeneous_personas):
    from evaluation.testing.scorers.distributional.opinion_diversity import OpinionDiversityScorer
    scorer = OpinionDiversityScorer()
    ctxs = [CTX] * len(homogeneous_personas)
    results = scorer.score_set(homogeneous_personas, ctxs)
    result = results[0]
    assert result.passed is False
    assert result.score < 0.5


def test_single_persona_skipped():
    from evaluation.testing.scorers.distributional.opinion_diversity import OpinionDiversityScorer
    scorer = OpinionDiversityScorer()
    # Gate is now n < 20; use n=10 to confirm any sub-threshold set is skipped
    personas = generate_test_persona_set(n=10, seed=1)
    results = scorer.score_set(personas, [CTX] * len(personas))
    assert results[0].details.get("skipped") is True


def test_modal_collapse_detected(homogeneous_personas):
    from evaluation.testing.scorers.distributional.opinion_diversity import OpinionDiversityScorer
    scorer = OpinionDiversityScorer()
    ctxs = [CTX] * len(homogeneous_personas)
    results = scorer.score_set(homogeneous_personas, ctxs)
    result = results[0]
    assert "modal_collapse_fields" in result.details
    assert len(result.details["modal_collapse_fields"]) > 0


def test_entrenchment_detected_when_responses_identical():
    """All personas give same answer regardless of demographics → entrenchment."""
    from evaluation.testing.scorers.distributional.opinion_diversity import OpinionDiversityScorer
    scorer = OpinionDiversityScorer()
    n = 25  # must be >= 20 to pass the min-size gate
    personas = generate_test_persona_set(n=n, seed=42)
    # Every persona gives identical response to each question
    ctxs = [
        SourceContext(id=f"s{i}", text="test", extra_data={
            "survey_responses": [
                {"question_id": "q1", "response": "Strongly agree"},
                {"question_id": "q2", "response": "Option A"},
                {"question_id": "q3", "response": "Yes"},
            ]
        })
        for i in range(n)
    ]
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert "entrenchment_rate" in result.details
    assert result.details["entrenchment_rate"] >= 0.9  # all questions entrenched


def test_no_entrenchment_when_responses_vary():
    """Different personas give different answers → no entrenchment."""
    from evaluation.testing.scorers.distributional.opinion_diversity import OpinionDiversityScorer
    scorer = OpinionDiversityScorer()
    n = 25  # must be >= 20 to pass the min-size gate
    personas = generate_test_persona_set(n=n, seed=42)
    responses_pool = ["Strongly agree", "Agree", "Neutral", "Disagree", "Strongly disagree"]
    rng = random.Random(99)
    ctxs = [
        SourceContext(id=f"s{i}", text="test", extra_data={
            "survey_responses": [
                {"question_id": "q1", "response": rng.choice(responses_pool)},
                {"question_id": "q2", "response": rng.choice(responses_pool)},
                {"question_id": "q3", "response": rng.choice(responses_pool)},
            ]
        })
        for i in range(n)
    ]
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert "entrenchment_rate" in result.details
    assert result.details["entrenchment_rate"] < 0.5


def test_entrenchment_skipped_when_no_survey_data():
    """No survey_responses → entrenchment analysis skipped, existing behavior unchanged."""
    from evaluation.testing.scorers.distributional.opinion_diversity import OpinionDiversityScorer
    scorer = OpinionDiversityScorer()
    personas = generate_test_persona_set(n=25, seed=42)  # >= 20 to pass min-size gate
    ctxs = [CTX] * len(personas)
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert "entrenchment_rate" not in result.details or result.details.get("entrenchment_skipped")
