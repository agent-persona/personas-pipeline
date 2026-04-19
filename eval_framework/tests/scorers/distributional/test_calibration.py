"""Tests for D17 Calibration scorer."""

from persona_eval.source_context import SourceContext
from tests.fixtures.persona_set import generate_test_persona_set


CTX = SourceContext(id="s1", text="distributional test")


def test_scorer_importable():
    from persona_eval.scorers.distributional.calibration import CalibrationScorer
    assert CalibrationScorer is not None


def test_scorer_metadata():
    from persona_eval.scorers.distributional.calibration import CalibrationScorer
    s = CalibrationScorer()
    assert s.dimension_id == "D17"
    assert s.tier == 3
    assert s.requires_set is True


def test_well_calibrated_predictions():
    from persona_eval.scorers.distributional.calibration import CalibrationScorer
    scorer = CalibrationScorer()
    # Reasonably calibrated: high confidence → mostly correct, low → mostly wrong
    predictions = [
        {"confidence": 0.9, "correct": True},
        {"confidence": 0.9, "correct": True},
        {"confidence": 0.9, "correct": True},
        {"confidence": 0.9, "correct": False},
        {"confidence": 0.5, "correct": True},
        {"confidence": 0.5, "correct": False},
        {"confidence": 0.1, "correct": False},
        {"confidence": 0.1, "correct": False},
    ]
    personas = generate_test_persona_set(n=8, seed=1)
    ctxs = [CTX] * len(personas)
    results = scorer.score_set(personas, ctxs, predictions=predictions)
    assert len(results) == 1
    result = results[0]
    assert result.passed is True
    assert result.score >= 0.5
    assert "ece" in result.details


def test_terrible_calibration():
    from persona_eval.scorers.distributional.calibration import CalibrationScorer
    scorer = CalibrationScorer()
    # High confidence, always wrong
    predictions = [
        {"confidence": 0.99, "correct": False},
        {"confidence": 0.95, "correct": False},
        {"confidence": 0.90, "correct": False},
        {"confidence": 0.85, "correct": False},
    ]
    personas = generate_test_persona_set(n=4, seed=1)
    ctxs = [CTX] * len(personas)
    results = scorer.score_set(personas, ctxs, predictions=predictions)
    result = results[0]
    assert result.passed is False
    assert result.score < 0.3


def test_empty_predictions_skipped():
    from persona_eval.scorers.distributional.calibration import CalibrationScorer
    scorer = CalibrationScorer()
    personas = generate_test_persona_set(n=2, seed=1)
    ctxs = [CTX] * len(personas)
    results = scorer.score_set(personas, ctxs, predictions=[])
    assert results[0].details.get("skipped") is True


def test_single_persona_score_skips():
    from persona_eval.scorers.distributional.calibration import CalibrationScorer
    scorer = CalibrationScorer()
    personas = generate_test_persona_set(n=1, seed=1)
    result = scorer.score(personas[0], CTX)
    assert result.details.get("skipped") is True
