"""Tests for Strategic Reasoning (NRA) scorer."""

import pytest
from evaluation.testing.schemas import Persona
from evaluation.testing.source_context import SourceContext


PERSONA = Persona(id="p1", name="Test Player")


def test_scorer_importable():
    from evaluation.testing.scorers.behavioral.strategic_reasoning import StrategicReasoningScorer
    assert StrategicReasoningScorer is not None


def test_scorer_metadata():
    from evaluation.testing.scorers.behavioral.strategic_reasoning import StrategicReasoningScorer
    s = StrategicReasoningScorer()
    assert s.dimension_id == "D33"
    assert s.tier == 5
    assert s.requires_set is False


def test_optimal_play_scores_high():
    """Persona plays optimally → NRA near 1.0."""
    from evaluation.testing.scorers.behavioral.strategic_reasoning import StrategicReasoningScorer
    scorer = StrategicReasoningScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "game_results": [
            {"game": "ultimatum_proposer", "reward": 7.0, "optimal_reward": 8.0, "random_reward": 5.0},
            {"game": "dictator", "reward": 9.0, "optimal_reward": 10.0, "random_reward": 5.0},
            {"game": "guess_2_3", "reward": 0.9, "optimal_reward": 1.0, "random_reward": 0.3},
        ]
    })
    result = scorer.score(PERSONA, ctx)
    assert result.passed is True
    assert result.score >= 0.6
    assert result.details["mean_nra"] > 0.5


def test_random_play_scores_zero():
    """Persona plays at random → NRA near 0."""
    from evaluation.testing.scorers.behavioral.strategic_reasoning import StrategicReasoningScorer
    scorer = StrategicReasoningScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "game_results": [
            {"game": "ultimatum_proposer", "reward": 5.0, "optimal_reward": 8.0, "random_reward": 5.0},
            {"game": "dictator", "reward": 5.0, "optimal_reward": 10.0, "random_reward": 5.0},
        ]
    })
    result = scorer.score(PERSONA, ctx)
    assert result.details["mean_nra"] <= 0.1


def test_worse_than_random_detected():
    """Persona plays worse than random → negative NRA (GTBench finding)."""
    from evaluation.testing.scorers.behavioral.strategic_reasoning import StrategicReasoningScorer
    scorer = StrategicReasoningScorer()
    ctx = SourceContext(id="s1", text="test", extra_data={
        "game_results": [
            {"game": "nim", "reward": 0.0, "optimal_reward": 10.0, "random_reward": 5.0},
            {"game": "connect4", "reward": 1.0, "optimal_reward": 10.0, "random_reward": 5.0},
        ]
    })
    result = scorer.score(PERSONA, ctx)
    assert result.passed is False
    assert result.details["mean_nra"] < 0.0


def test_skipped_when_no_data():
    """No game_results → skipped."""
    from evaluation.testing.scorers.behavioral.strategic_reasoning import StrategicReasoningScorer
    scorer = StrategicReasoningScorer()
    ctx = SourceContext(id="s1", text="test")
    result = scorer.score(PERSONA, ctx)
    assert result.details.get("skipped") is True
