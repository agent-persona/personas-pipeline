"""Tests for Tail Insight Detection scorer (D-NEW)."""

import pytest

from persona_eval.schemas import Persona
from persona_eval.source_context import SourceContext
from tests.fixtures.persona_set import generate_test_persona_set
from tests.fixtures.tail_insights import SAMPLE_BENCHMARK, MAJORITY_RESPONSES


CTX = SourceContext(id="s1", text="test")


def test_scorer_importable():
    from persona_eval.scorers.distributional.tail_insight_detection import (
        TailInsightDetectionScorer,
    )
    assert TailInsightDetectionScorer is not None


def test_scorer_metadata():
    from persona_eval.scorers.distributional.tail_insight_detection import (
        TailInsightDetectionScorer,
    )
    s = TailInsightDetectionScorer()
    assert s.dimension_id == "D-NEW"
    assert s.tier == 6
    assert s.requires_set is True


def test_skipped_when_no_benchmark():
    """No tail_insights_benchmark → skipped."""
    from persona_eval.scorers.distributional.tail_insight_detection import (
        TailInsightDetectionScorer,
    )
    scorer = TailInsightDetectionScorer()
    personas = generate_test_persona_set(n=5, seed=42)
    ctxs = [CTX] * 5
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert result.details.get("skipped") is True


def test_compute_metrics_all_detected():
    """All insights have high similarity → full recall."""
    from persona_eval.scorers.distributional.tail_insight_detection import (
        TailInsightDetectionScorer,
    )
    scorer = TailInsightDetectionScorer()
    benchmark = [
        {"insight_id": "ti_001", "novelty_score": 5, "semantic_threshold": 0.65},
        {"insight_id": "ti_002", "novelty_score": 3, "semantic_threshold": 0.65},
        {"insight_id": "ti_003", "novelty_score": 4, "semantic_threshold": 0.65},
    ]
    max_similarities = {"ti_001": 0.82, "ti_002": 0.71, "ti_003": 0.90}

    metrics = scorer._compute_detection_metrics(benchmark, max_similarities)
    assert metrics["tail_insight_recall"] == 1.0
    assert set(metrics["detected_insights"]) == {"ti_001", "ti_002", "ti_003"}
    assert metrics["missed_insights"] == []


def test_compute_metrics_partial_detection():
    """Some insights detected, some missed → partial recall."""
    from persona_eval.scorers.distributional.tail_insight_detection import (
        TailInsightDetectionScorer,
    )
    scorer = TailInsightDetectionScorer()
    benchmark = [
        {"insight_id": "ti_001", "novelty_score": 5, "semantic_threshold": 0.65},
        {"insight_id": "ti_002", "novelty_score": 3, "semantic_threshold": 0.65},
        {"insight_id": "ti_003", "novelty_score": 4, "semantic_threshold": 0.65},
    ]
    max_similarities = {"ti_001": 0.40, "ti_002": 0.71, "ti_003": 0.80}

    metrics = scorer._compute_detection_metrics(benchmark, max_similarities)
    assert abs(metrics["tail_insight_recall"] - 2 / 3) < 0.01
    assert "ti_001" in metrics["missed_insights"]
    assert "ti_002" in metrics["detected_insights"]
    assert "ti_003" in metrics["detected_insights"]


def test_compute_metrics_none_detected():
    """All similarities below threshold → zero recall."""
    from persona_eval.scorers.distributional.tail_insight_detection import (
        TailInsightDetectionScorer,
    )
    scorer = TailInsightDetectionScorer()
    benchmark = [
        {"insight_id": "ti_001", "novelty_score": 5, "semantic_threshold": 0.65},
        {"insight_id": "ti_002", "novelty_score": 3, "semantic_threshold": 0.65},
    ]
    max_similarities = {"ti_001": 0.30, "ti_002": 0.20}

    metrics = scorer._compute_detection_metrics(benchmark, max_similarities)
    assert metrics["tail_insight_recall"] == 0.0
    assert len(metrics["missed_insights"]) == 2
    assert metrics["detected_insights"] == []


def test_compute_metrics_per_insight_threshold():
    """Each insight can have its own semantic_threshold."""
    from persona_eval.scorers.distributional.tail_insight_detection import (
        TailInsightDetectionScorer,
    )
    scorer = TailInsightDetectionScorer()
    benchmark = [
        {"insight_id": "ti_001", "novelty_score": 5, "semantic_threshold": 0.80},
        {"insight_id": "ti_002", "novelty_score": 3, "semantic_threshold": 0.50},
    ]
    max_similarities = {"ti_001": 0.75, "ti_002": 0.55}

    metrics = scorer._compute_detection_metrics(benchmark, max_similarities)
    assert metrics["tail_insight_recall"] == 0.5
    assert "ti_001" in metrics["missed_insights"]
    assert "ti_002" in metrics["detected_insights"]


def test_compute_metrics_exact_threshold_counts_as_detected():
    """Similarity exactly equal to threshold → detected (>= not >)."""
    from persona_eval.scorers.distributional.tail_insight_detection import (
        TailInsightDetectionScorer,
    )
    scorer = TailInsightDetectionScorer()
    benchmark = [
        {"insight_id": "ti_001", "novelty_score": 3, "semantic_threshold": 0.65},
    ]
    max_similarities = {"ti_001": 0.65}  # exactly at threshold

    metrics = scorer._compute_detection_metrics(benchmark, max_similarities)
    assert metrics["tail_insight_recall"] == 1.0
    assert "ti_001" in metrics["detected_insights"]


def test_novelty_weighted_recall_favors_hard_insights():
    """Detecting high-novelty insights gives higher weighted recall than low-novelty."""
    from persona_eval.scorers.distributional.tail_insight_detection import (
        TailInsightDetectionScorer,
    )
    scorer = TailInsightDetectionScorer()

    benchmark = [
        {"insight_id": "ti_easy", "novelty_score": 1, "semantic_threshold": 0.65},
        {"insight_id": "ti_hard", "novelty_score": 5, "semantic_threshold": 0.65},
    ]

    # Scenario A: detect only the easy insight
    sims_easy = {"ti_easy": 0.80, "ti_hard": 0.30}
    metrics_easy = scorer._compute_detection_metrics(benchmark, sims_easy)

    # Scenario B: detect only the hard insight
    sims_hard = {"ti_easy": 0.30, "ti_hard": 0.80}
    metrics_hard = scorer._compute_detection_metrics(benchmark, sims_hard)

    # Both have 50% raw recall, but detecting the hard one should give higher weighted recall
    assert metrics_easy["tail_insight_recall"] == metrics_hard["tail_insight_recall"]
    assert metrics_hard["novelty_weighted_recall"] > metrics_easy["novelty_weighted_recall"]


def test_detection_by_novelty_tier():
    """Detection broken down by novelty tier."""
    from persona_eval.scorers.distributional.tail_insight_detection import (
        TailInsightDetectionScorer,
    )
    scorer = TailInsightDetectionScorer()

    benchmark = [
        {"insight_id": "ti_1", "novelty_score": 1, "semantic_threshold": 0.65},
        {"insight_id": "ti_2", "novelty_score": 2, "semantic_threshold": 0.65},
        {"insight_id": "ti_3", "novelty_score": 3, "semantic_threshold": 0.65},
        {"insight_id": "ti_4", "novelty_score": 4, "semantic_threshold": 0.65},
        {"insight_id": "ti_5", "novelty_score": 5, "semantic_threshold": 0.65},
    ]
    # Detect low-novelty only
    sims = {
        "ti_1": 0.80, "ti_2": 0.70,  # detected (low tier)
        "ti_3": 0.40,                  # missed (medium tier)
        "ti_4": 0.30, "ti_5": 0.20,   # missed (high tier)
    }
    metrics = scorer._compute_detection_metrics(benchmark, sims)

    tiers = metrics["detection_by_novelty"]
    assert tiers["low (1-2)"]["recall"] == 1.0   # 2/2 detected
    assert tiers["medium (3)"]["recall"] == 0.0   # 0/1 detected
    assert tiers["high (4-5)"]["recall"] == 0.0   # 0/2 detected


def test_novelty_weighted_recall_all_detected():
    """All detected → novelty_weighted_recall == 1.0."""
    from persona_eval.scorers.distributional.tail_insight_detection import (
        TailInsightDetectionScorer,
    )
    scorer = TailInsightDetectionScorer()

    benchmark = [
        {"insight_id": "ti_1", "novelty_score": 5, "semantic_threshold": 0.65},
        {"insight_id": "ti_2", "novelty_score": 3, "semantic_threshold": 0.65},
    ]
    sims = {"ti_1": 0.90, "ti_2": 0.80}
    metrics = scorer._compute_detection_metrics(benchmark, sims)
    assert metrics["novelty_weighted_recall"] == 1.0


@pytest.mark.slow
def test_full_pipeline_majority_responses_miss_tail_insights():
    """Generic majority responses should NOT detect tail insights (the whole point)."""
    from persona_eval.scorers.distributional.tail_insight_detection import (
        TailInsightDetectionScorer,
    )
    scorer = TailInsightDetectionScorer()
    personas = generate_test_persona_set(n=5, seed=42)
    ctxs = [SourceContext(id=f"s{i}", text="test") for i in range(5)]
    ctxs[0].extra_data["tail_insights_benchmark"] = SAMPLE_BENCHMARK
    ctxs[0].extra_data["persona_responses"] = [
        {"persona_id": f"p{i}", "response_text": resp}
        for i, resp in enumerate(MAJORITY_RESPONSES)
    ]
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    # Majority responses should have LOW recall on tail insights
    assert result.details["tail_insight_recall"] < 0.5
    assert "detection_by_novelty" in result.details


@pytest.mark.slow
def test_full_pipeline_matching_responses_detect_insights():
    """Responses that closely match insight text → high recall."""
    from persona_eval.scorers.distributional.tail_insight_detection import (
        TailInsightDetectionScorer,
    )
    scorer = TailInsightDetectionScorer()
    personas = generate_test_persona_set(n=5, seed=42)
    ctxs = [SourceContext(id=f"s{i}", text="test") for i in range(5)]
    # Use the insight texts themselves as responses (perfect match)
    ctxs[0].extra_data["tail_insights_benchmark"] = SAMPLE_BENCHMARK
    ctxs[0].extra_data["persona_responses"] = [
        {"persona_id": f"p{i}", "response_text": ins["insight_text"]}
        for i, ins in enumerate(SAMPLE_BENCHMARK)
    ]
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    # Near-exact match responses should detect most/all insights
    assert result.details["tail_insight_recall"] >= 0.8
    assert result.passed is True


@pytest.mark.slow
def test_full_pipeline_score_in_bounds():
    """Score should always be in [0.0, 1.0]."""
    from persona_eval.scorers.distributional.tail_insight_detection import (
        TailInsightDetectionScorer,
    )
    scorer = TailInsightDetectionScorer()
    personas = generate_test_persona_set(n=3, seed=42)
    ctxs = [SourceContext(id=f"s{i}", text="test") for i in range(3)]
    ctxs[0].extra_data["tail_insights_benchmark"] = SAMPLE_BENCHMARK[:2]
    ctxs[0].extra_data["persona_responses"] = [
        {"persona_id": "p1", "response_text": "I like shopping online."},
    ]
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert 0.0 <= result.score <= 1.0
