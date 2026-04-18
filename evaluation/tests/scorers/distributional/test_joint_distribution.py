"""Tests for D18 Joint Distribution Fidelity scorer."""

import pytest
from evaluation.testing.schemas import Persona, CommunicationStyle, EmotionalProfile
from evaluation.testing.source_context import SourceContext
from tests.fixtures.persona_set import generate_test_persona_set


CTX = SourceContext(id="s1", text="distributional test")


@pytest.fixture
def diverse_personas():
    return generate_test_persona_set(n=50, seed=42)


def test_scorer_importable():
    from evaluation.testing.scorers.distributional.joint_distribution import JointDistributionScorer
    assert JointDistributionScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.distributional.joint_distribution import JointDistributionScorer
    s = JointDistributionScorer()
    assert s.dimension_id == "D18"
    assert s.tier == 3
    assert s.requires_set is True


def test_diverse_set_computes_correlations(diverse_personas):
    from evaluation.testing.scorers.distributional.joint_distribution import JointDistributionScorer
    scorer = JointDistributionScorer()
    ctxs = [CTX] * len(diverse_personas)
    results = scorer.score_set(diverse_personas, ctxs)
    assert len(results) == 1
    result = results[0]
    assert result.passed is True
    assert result.score >= 0.5
    assert "correlation_matrix" in result.details
    assert len(result.details["correlation_matrix"]) > 0


def test_detects_stereotypical_correlation():
    """Forced stereotypical set: all females collaborative/low-risk, all males competitive/high-risk."""
    from evaluation.testing.scorers.distributional.joint_distribution import JointDistributionScorer
    scorer = JointDistributionScorer()

    stereotyped = []
    for i in range(25):
        stereotyped.append(Persona(
            id=f"stereo-f-{i}", name=f"Female {i}", age=30,
            gender="female", location="NY", education="Master's degree",
            occupation="PM", industry="SaaS", income_bracket="middle",
            ethnicity="White", marital_status="single",
            lifestyle="urban commuter",
            communication_style=CommunicationStyle(
                tone="warm", formality="semi-formal", vocabulary_level="advanced",
            ),
            emotional_profile=EmotionalProfile(
                baseline_mood="calm", stress_triggers=["t"], coping_mechanisms=["c"],
            ),
        ))
    for i in range(25):
        stereotyped.append(Persona(
            id=f"stereo-m-{i}", name=f"Male {i}", age=30,
            gender="male", location="NY", education="PhD",
            occupation="CTO", industry="SaaS", income_bracket="high",
            ethnicity="White", marital_status="single",
            lifestyle="remote-first",
            communication_style=CommunicationStyle(
                tone="direct", formality="informal", vocabulary_level="technical",
            ),
            emotional_profile=EmotionalProfile(
                baseline_mood="driven", stress_triggers=["t"], coping_mechanisms=["c"],
            ),
        ))

    ctxs = [CTX] * len(stereotyped)
    results = scorer.score_set(stereotyped, ctxs)
    result = results[0]
    assert "stereotypical_pairs" in result.details
    assert len(result.details["stereotypical_pairs"]) > 0
    assert result.passed is False


def test_small_set_skipped():
    from evaluation.testing.scorers.distributional.joint_distribution import JointDistributionScorer
    scorer = JointDistributionScorer()
    personas = generate_test_persona_set(n=10, seed=1)
    ctxs = [CTX] * len(personas)
    results = scorer.score_set(personas, ctxs)
    assert results[0].details.get("skipped") is True


def test_single_persona_score_skips():
    from evaluation.testing.scorers.distributional.joint_distribution import JointDistributionScorer
    scorer = JointDistributionScorer()
    personas = generate_test_persona_set(n=1, seed=1)
    result = scorer.score(personas[0], CTX)
    assert result.details.get("skipped") is True


def test_regression_accuracy_good_coefficients():
    """Synthetic coefficients close to real → good score."""
    from evaluation.testing.scorers.distributional.joint_distribution import JointDistributionScorer
    scorer = JointDistributionScorer()
    personas = generate_test_persona_set(n=50, seed=42)
    ctxs = [SourceContext(id=f"s{i}", text="test") for i in range(50)]
    ctxs[0].extra_data["regression_reference"] = {
        "coefficients": [
            {"predictor": "age", "outcome": "satisfaction", "real_coeff": 0.3, "synthetic_coeff": 0.28},
            {"predictor": "income", "outcome": "satisfaction", "real_coeff": -0.2, "synthetic_coeff": -0.18},
            {"predictor": "education", "outcome": "adoption", "real_coeff": 0.5, "synthetic_coeff": 0.45},
        ]
    }
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert "regression_sign_flip_rate" in result.details
    assert result.details["regression_sign_flip_rate"] == 0.0
    assert result.details["regression_wrong_rate"] < 0.5


def test_regression_detects_sign_flips():
    """Sign-flipped coefficients → flagged."""
    from evaluation.testing.scorers.distributional.joint_distribution import JointDistributionScorer
    scorer = JointDistributionScorer()
    personas = generate_test_persona_set(n=50, seed=42)
    ctxs = [SourceContext(id=f"s{i}", text="test") for i in range(50)]
    ctxs[0].extra_data["regression_reference"] = {
        "coefficients": [
            {"predictor": "age", "outcome": "vote", "real_coeff": 0.3, "synthetic_coeff": -0.1},
            {"predictor": "income", "outcome": "vote", "real_coeff": -0.2, "synthetic_coeff": 0.3},
            {"predictor": "education", "outcome": "vote", "real_coeff": 0.5, "synthetic_coeff": 0.45},
        ]
    }
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert result.details["regression_sign_flip_rate"] > 0.5


def test_regression_skipped_when_no_reference():
    """No regression_reference → analysis skipped."""
    from evaluation.testing.scorers.distributional.joint_distribution import JointDistributionScorer
    scorer = JointDistributionScorer()
    personas = generate_test_persona_set(n=20, seed=42)
    ctxs = [CTX] * len(personas)
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert "regression_sign_flip_rate" not in result.details
