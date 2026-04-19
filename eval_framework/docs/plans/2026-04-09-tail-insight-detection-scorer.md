# Tail Insight Detection Scorer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a D-NEW TailInsightDetectionScorer that measures whether synthetic persona responses surface rare, non-obvious insights from a benchmark dataset, plus a skeleton benchmark with 5 example scenarios.

**Architecture:** Set-level scorer in `distributional/`. Core detection logic (similarity → metrics) is a pure function testable without numpy. The embedding integration layer uses lazy `Embedder` import. Benchmark data lives as a test fixture. Novelty-weighted recall is the primary metric.

**Tech Stack:** Python 3.11+, pytest, Pydantic v2, sentence-transformers (Embedder) for semantic similarity

---

## Task 1: Scorer Skeleton + Benchmark Fixture Data

**Files:**
- Create: `persona_eval/scorers/distributional/tail_insight_detection.py`
- Create: `tests/scorers/distributional/test_tail_insight_detection.py`
- Create: `tests/fixtures/tail_insights.py`

**What it does:** Define the scorer class with metadata, the skip-when-no-data path, and a reusable benchmark fixture with 5 sample scenarios.

### Step 1: Write the benchmark fixture

Create `tests/fixtures/tail_insights.py`:

```python
"""Fixture: sample tail insight benchmark data for testing D-NEW scorer.

Each scenario has:
- research_question: the prompt context
- benchmark_insights: annotated insights with novelty scores
- majority_response: what a typical/expected persona would say

These are synthetic examples for testing. Real benchmark data requires
expert annotation per the design doc at docs/plans/2026-04-09-tail-insight-detection-design.md.
"""

from __future__ import annotations

SAMPLE_BENCHMARK = [
    {
        "insight_id": "ti_001",
        "insight_text": (
            "Users with chronic conditions prefer voice interfaces not for "
            "convenience but because screen fatigue from medical portals "
            "causes them to skip medication refills"
        ),
        "novelty_score": 5,
        "prevalence": 0.06,
        "domain": "healthcare_ux",
        "semantic_threshold": 0.65,
    },
    {
        "insight_id": "ti_002",
        "insight_text": (
            "First-generation college students avoid office hours not from "
            "shyness but because they interpret professor availability as "
            "a sign the course is too hard for them"
        ),
        "novelty_score": 4,
        "prevalence": 0.09,
        "domain": "education",
        "semantic_threshold": 0.65,
    },
    {
        "insight_id": "ti_003",
        "insight_text": (
            "Budget-conscious families buy premium organic baby food not for "
            "health beliefs but as guilt compensation for not being able to "
            "afford other enrichment activities"
        ),
        "novelty_score": 5,
        "prevalence": 0.04,
        "domain": "consumer_behavior",
        "semantic_threshold": 0.65,
    },
    {
        "insight_id": "ti_004",
        "insight_text": (
            "Remote workers in rural areas report higher job satisfaction "
            "than urban remote workers despite worse internet because the "
            "social status of a tech salary in a small town changes their "
            "entire community standing"
        ),
        "novelty_score": 3,
        "prevalence": 0.12,
        "domain": "workplace",
        "semantic_threshold": 0.65,
    },
    {
        "insight_id": "ti_005",
        "insight_text": (
            "Elderly users prefer physical bank branches not for technophobia "
            "but because the branch visit is their primary social interaction "
            "of the week"
        ),
        "novelty_score": 3,
        "prevalence": 0.11,
        "domain": "fintech",
        "semantic_threshold": 0.65,
    },
]

# Majority-pattern responses that any generic persona would produce
MAJORITY_RESPONSES = [
    "I prefer digital banking because it's convenient and saves time.",
    "Voice assistants are useful for hands-free convenience.",
    "I choose organic food because I care about my family's health.",
    "Remote work is great because of the flexibility and no commute.",
    "I think office hours are important for getting help with coursework.",
]
```

### Step 2: Write the failing tests

Create `tests/scorers/distributional/test_tail_insight_detection.py`:

```python
"""Tests for Tail Insight Detection scorer (D-NEW)."""

import pytest
from persona_eval.schemas import Persona
from persona_eval.source_context import SourceContext
from tests.fixtures.persona_set import generate_test_persona_set


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
```

### Step 3: Run tests to verify they fail

Run: `cd /Users/ivanma/Desktop/gauntlet/Capstone && python3 -m pytest tests/scorers/distributional/test_tail_insight_detection.py -v -x --tb=short 2>&1 | tail -15`
Expected: FAIL (module not found)

### Step 4: Implement scorer skeleton

Create `persona_eval/scorers/distributional/tail_insight_detection.py`:

```python
"""D-NEW Tail Insight Detection — measures ability to surface rare, non-obvious findings.

Trustworthiness: EXPERIMENTAL (requires curated benchmark; semantic similarity is a proxy).
Method: Compare persona responses against a benchmark of annotated tail insights
using embedding similarity. Compute recall, novelty-weighted recall, and detection-by-tier.
Evidence: Speero/NN/g practitioner evaluations — 0% tail insight detection across all cases.
Zero academic measurement exists. See docs/plans/2026-04-09-tail-insight-detection-design.md.
"""

from __future__ import annotations

from typing import Any

from persona_eval.scorer import BaseScorer
from persona_eval.schemas import Persona, EvalResult
from persona_eval.source_context import SourceContext

# Default similarity threshold for detecting an insight
DEFAULT_SEMANTIC_THRESHOLD = 0.65

# Pass if recall exceeds this (very low bar — current state is 0%)
RECALL_PASS_THRESHOLD = 0.10


class TailInsightDetectionScorer(BaseScorer):
    """Measures whether synthetic personas surface tail insights from a benchmark."""

    dimension_id = "D-NEW"
    dimension_name = "Tail Insight Detection"
    tier = 6
    requires_set = True

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        return self._result(
            persona, passed=True, score=1.0,
            details={"skipped": True, "reason": "D-NEW is a set-level dimension"},
        )

    def score_set(
        self, personas: list[Persona], source_contexts: list[SourceContext]
    ) -> list[EvalResult]:
        # Collect benchmark and responses from extra_data
        benchmark: list[dict[str, Any]] = []
        responses: list[dict[str, Any]] = []
        for ctx in source_contexts:
            b = ctx.extra_data.get("tail_insights_benchmark", [])
            if b:
                benchmark = b
            r = ctx.extra_data.get("persona_responses", [])
            if r:
                responses = r

        if not benchmark:
            return [EvalResult(
                dimension_id=self.dimension_id,
                dimension_name=self.dimension_name,
                persona_id="__set__",
                passed=True, score=1.0,
                details={"skipped": True, "reason": "No tail_insights_benchmark provided"},
            )]

        if not responses:
            return [EvalResult(
                dimension_id=self.dimension_id,
                dimension_name=self.dimension_name,
                persona_id="__set__",
                passed=False, score=0.0,
                details={"skipped": True, "reason": "No persona_responses provided"},
            )]

        # Placeholder: detection logic added in Task 2
        return [EvalResult(
            dimension_id=self.dimension_id,
            dimension_name=self.dimension_name,
            persona_id="__set__",
            passed=False, score=0.0,
            details={"not_implemented": True},
        )]
```

### Step 5: Run tests to verify they pass

Run: `cd /Users/ivanma/Desktop/gauntlet/Capstone && python3 -m pytest tests/scorers/distributional/test_tail_insight_detection.py -v --tb=short 2>&1 | tail -15`
Expected: ALL 3 tests PASS

### Step 6: Commit

```bash
git add tests/fixtures/tail_insights.py persona_eval/scorers/distributional/tail_insight_detection.py tests/scorers/distributional/test_tail_insight_detection.py
git commit -m "feat(D-NEW): scaffold tail insight detection scorer + benchmark fixture"
```

---

## Task 2: Core Detection Metrics (Pure Logic, No Embeddings)

**Files:**
- Modify: `persona_eval/scorers/distributional/tail_insight_detection.py`
- Modify: `tests/scorers/distributional/test_tail_insight_detection.py`

**What it does:** Implement `_compute_detection_metrics()` as a pure function that takes pre-computed similarity scores and benchmark data, and returns recall/precision/detection details. This is testable without numpy/sentence-transformers.

### Step 1: Write the failing tests

Add to `tests/scorers/distributional/test_tail_insight_detection.py`:

```python
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
    # Simulate: max similarity for each insight across all responses
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
    # ti_001 missed (below threshold), ti_002 and ti_003 detected
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
    # ti_001 at 0.75 is BELOW its threshold of 0.80 → missed
    # ti_002 at 0.55 is ABOVE its threshold of 0.50 → detected
    max_similarities = {"ti_001": 0.75, "ti_002": 0.55}

    metrics = scorer._compute_detection_metrics(benchmark, max_similarities)
    assert metrics["tail_insight_recall"] == 0.5
    assert "ti_001" in metrics["missed_insights"]
    assert "ti_002" in metrics["detected_insights"]
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/ivanma/Desktop/gauntlet/Capstone && python3 -m pytest tests/scorers/distributional/test_tail_insight_detection.py -v -x --tb=short 2>&1 | tail -15`
Expected: New tests FAIL (method not found)

### Step 3: Implement `_compute_detection_metrics`

Add to the `TailInsightDetectionScorer` class in `persona_eval/scorers/distributional/tail_insight_detection.py`:

```python
def _compute_detection_metrics(
    self,
    benchmark: list[dict[str, Any]],
    max_similarities: dict[str, float],
) -> dict[str, Any]:
    """Pure detection logic: given similarity scores, compute recall metrics.

    Args:
        benchmark: list of insight dicts with insight_id, novelty_score, semantic_threshold
        max_similarities: dict mapping insight_id → max similarity score across all responses

    Returns:
        Detection metrics dict.
    """
    detected = []
    missed = []

    for insight in benchmark:
        iid = insight["insight_id"]
        threshold = insight.get("semantic_threshold", DEFAULT_SEMANTIC_THRESHOLD)
        sim = max_similarities.get(iid, 0.0)
        if sim >= threshold:
            detected.append(iid)
        else:
            missed.append(iid)

    total = len(benchmark)
    recall = len(detected) / total if total > 0 else 0.0

    return {
        "tail_insight_recall": round(recall, 4),
        "detected_insights": detected,
        "missed_insights": missed,
        "benchmark_size": total,
        "detected_count": len(detected),
    }
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/ivanma/Desktop/gauntlet/Capstone && python3 -m pytest tests/scorers/distributional/test_tail_insight_detection.py -v --tb=short 2>&1 | tail -20`
Expected: ALL 7 tests PASS

### Step 5: Commit

```bash
git add persona_eval/scorers/distributional/tail_insight_detection.py tests/scorers/distributional/test_tail_insight_detection.py
git commit -m "feat(D-NEW): add core detection metrics — recall, detected/missed insights"
```

---

## Task 3: Novelty-Weighted Recall + Detection-by-Tier

**Files:**
- Modify: `persona_eval/scorers/distributional/tail_insight_detection.py`
- Modify: `tests/scorers/distributional/test_tail_insight_detection.py`

**What it does:** Extend `_compute_detection_metrics()` to compute novelty-weighted recall and break down detection by novelty tier (low 1-2, medium 3, high 4-5). Still pure logic, no embeddings.

### Step 1: Write the failing tests

Add to `tests/scorers/distributional/test_tail_insight_detection.py`:

```python
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
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/ivanma/Desktop/gauntlet/Capstone && python3 -m pytest tests/scorers/distributional/test_tail_insight_detection.py -v -x --tb=short 2>&1 | tail -15`
Expected: New tests FAIL (missing keys in metrics dict)

### Step 3: Extend `_compute_detection_metrics` with novelty weighting

Update the method in `persona_eval/scorers/distributional/tail_insight_detection.py`. Replace the return dict construction with:

```python
    # Novelty-weighted recall: Σ(detected_i × novelty_i) / Σ(novelty_i)
    total_novelty = sum(i["novelty_score"] for i in benchmark)
    detected_novelty = sum(
        i["novelty_score"] for i in benchmark if i["insight_id"] in detected
    )
    novelty_weighted_recall = (
        detected_novelty / total_novelty if total_novelty > 0 else 0.0
    )

    # Detection by novelty tier
    tiers = {
        "low (1-2)": {"detected": 0, "total": 0},
        "medium (3)": {"detected": 0, "total": 0},
        "high (4-5)": {"detected": 0, "total": 0},
    }
    for insight in benchmark:
        ns = insight["novelty_score"]
        if ns <= 2:
            tier_key = "low (1-2)"
        elif ns == 3:
            tier_key = "medium (3)"
        else:
            tier_key = "high (4-5)"
        tiers[tier_key]["total"] += 1
        if insight["insight_id"] in detected:
            tiers[tier_key]["detected"] += 1

    detection_by_novelty = {}
    for tier_key, counts in tiers.items():
        detection_by_novelty[tier_key] = {
            "recall": round(counts["detected"] / counts["total"], 4) if counts["total"] > 0 else 0.0,
            "detected": counts["detected"],
            "total": counts["total"],
        }

    return {
        "tail_insight_recall": round(recall, 4),
        "novelty_weighted_recall": round(novelty_weighted_recall, 4),
        "detection_by_novelty": detection_by_novelty,
        "detected_insights": detected,
        "missed_insights": missed,
        "benchmark_size": total,
        "detected_count": len(detected),
    }
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/ivanma/Desktop/gauntlet/Capstone && python3 -m pytest tests/scorers/distributional/test_tail_insight_detection.py -v --tb=short 2>&1 | tail -20`
Expected: ALL 10 tests PASS

### Step 5: Commit

```bash
git add persona_eval/scorers/distributional/tail_insight_detection.py tests/scorers/distributional/test_tail_insight_detection.py
git commit -m "feat(D-NEW): add novelty-weighted recall + detection-by-tier breakdown"
```

---

## Task 4: Embedding Integration — Full `score_set` Pipeline

**Files:**
- Modify: `persona_eval/scorers/distributional/tail_insight_detection.py`
- Modify: `tests/scorers/distributional/test_tail_insight_detection.py`

**What it does:** Wire up the Embedder to compute actual similarity between benchmark insights and persona responses. Replace the placeholder in `score_set()` with the full pipeline: embed → compute max similarities → detection metrics → EvalResult.

### Step 1: Write the failing tests

Add to `tests/scorers/distributional/test_tail_insight_detection.py`:

```python
import pytest
from tests.fixtures.tail_insights import SAMPLE_BENCHMARK, MAJORITY_RESPONSES


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
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/ivanma/Desktop/gauntlet/Capstone && python3 -m pytest tests/scorers/distributional/test_tail_insight_detection.py -v -x -k "full_pipeline" --tb=short 2>&1 | tail -15`
Expected: FAIL (either numpy missing or placeholder return)

### Step 3: Wire up embedding integration

Update `persona_eval/scorers/distributional/tail_insight_detection.py`. Add lazy Embedder pattern and replace the `score_set` placeholder:

Add `__init__` and `_get_embedder` to the class (before `score`):

```python
def __init__(self) -> None:
    self._embedder = None  # Embedder, lazily imported

def _get_embedder(self):
    if self._embedder is None:
        from persona_eval.embeddings import Embedder
        self._embedder = Embedder()
    return self._embedder
```

Replace the `# Placeholder: detection logic added in Task 2` block and everything after it in `score_set()` with:

```python
        # Embed benchmark insights and persona responses
        embedder = self._get_embedder()
        insight_texts = [ins["insight_text"] for ins in benchmark]
        response_texts = [r["response_text"] for r in responses]

        insight_vecs = embedder.embed_batch(insight_texts)
        response_vecs = embedder.embed_batch(response_texts)

        # Compute max similarity for each insight across all responses
        max_similarities: dict[str, float] = {}
        for i, insight in enumerate(benchmark):
            max_sim = 0.0
            for j in range(len(response_vecs)):
                sim = embedder.vector_similarity(insight_vecs[i], response_vecs[j])
                if sim > max_sim:
                    max_sim = sim
            max_similarities[insight["insight_id"]] = max_sim

        # Compute detection metrics
        metrics = self._compute_detection_metrics(benchmark, max_similarities)

        # Score: use novelty-weighted recall as the primary score
        score = min(1.0, max(0.0, metrics["novelty_weighted_recall"]))
        passed = metrics["tail_insight_recall"] >= RECALL_PASS_THRESHOLD

        return [EvalResult(
            dimension_id=self.dimension_id,
            dimension_name=self.dimension_name,
            persona_id="__set__",
            passed=passed,
            score=round(score, 4),
            details={
                **metrics,
                "recall_pass_threshold": RECALL_PASS_THRESHOLD,
            },
        )]
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/ivanma/Desktop/gauntlet/Capstone && python3 -m pytest tests/scorers/distributional/test_tail_insight_detection.py -v --tb=short 2>&1 | tail -25`
Expected: ALL tests PASS (pure-logic tests always pass; slow tests pass if numpy/sentence-transformers installed, skip otherwise)

Note: If numpy is not installed, the `@pytest.mark.slow` tests will fail with `ModuleNotFoundError`. This is pre-existing behavior for all embedding-dependent tests in this project. The 10 pure-logic tests should always pass.

### Step 5: Commit

```bash
git add persona_eval/scorers/distributional/tail_insight_detection.py tests/scorers/distributional/test_tail_insight_detection.py
git commit -m "feat(D-NEW): wire embedding pipeline in score_set — full tail insight detection [ship]

First ever scorer measuring tail insight detection. The metric that every
practitioner evaluation scored 0% on now has a measurable dimension. Skeleton
benchmark with 5 scenarios, novelty-weighted recall, detection-by-tier breakdown.
"
```

---

## Summary

| Task | What | Tests | Lines |
|------|------|-------|-------|
| 1 | Scorer skeleton + benchmark fixture | 3 | ~100 |
| 2 | Core detection metrics (pure logic) | 4 | ~40 |
| 3 | Novelty-weighted recall + tiers | 3 | ~40 |
| 4 | Embedding integration pipeline | 3 | ~30 |

**Total:** 13 new tests, ~210 new lines, 1 new scorer, 1 benchmark fixture.

**All pure-logic tests (10) run without numpy. Embedding tests (3) require numpy/sentence-transformers.**
