# Research-Informed Scorer Additions Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 2 new scorers and enhance 4 existing scorers based on findings from the corpus claim analyses (analysis_claim1-3, boundary framework, research roadmap).

**Architecture:** Each enhancement adds a conditional check gated on `extra_data` keys so existing tests don't break. New scorers follow the established `BaseScorer` pattern with TDD. All new metrics have evidence-based thresholds from the research corpus.

**Tech Stack:** Python 3.11+, pytest, Pydantic v2, sentence-transformers (Embedder), numpy (for NRA computation)

---

## Task 1: Enhance D13 — RLHF Entrenchment Detection

**Evidence:** CLAIMSIM (Yu 2025): 50% of questions → single viewpoint regardless of demographics. Das Man (Li 2025): mathematical proof of mode convergence.

**Files:**
- Modify: `persona_eval/scorers/distributional/opinion_diversity.py`
- Modify: `tests/scorers/distributional/test_opinion_diversity.py`

**What it does:** When `extra_data["survey_responses"]` is provided on source contexts, group responses by question, check if response is invariant to persona demographics. Report entrenchment_rate (% of questions where all demographic groups give the same modal answer).

### Step 1: Write the failing tests

Add to `tests/scorers/distributional/test_opinion_diversity.py`:

```python
def test_entrenchment_detected_when_responses_identical():
    """All personas give same answer regardless of demographics → entrenchment."""
    from persona_eval.scorers.distributional.opinion_diversity import OpinionDiversityScorer
    scorer = OpinionDiversityScorer()
    personas = generate_test_persona_set(n=10, seed=42)
    # Every persona gives identical response to each question
    ctxs = [
        SourceContext(id=f"s{i}", text="test", extra_data={
            "survey_responses": [
                {"question_id": "q1", "response": "Strongly agree"},
                {"question_id": "q2", "response": "Option A"},
                {"question_id": "q3", "response": "Yes"},
            ]
        })
        for i in range(10)
    ]
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert "entrenchment_rate" in result.details
    assert result.details["entrenchment_rate"] >= 0.9  # all questions entrenched


def test_no_entrenchment_when_responses_vary():
    """Different personas give different answers → no entrenchment."""
    from persona_eval.scorers.distributional.opinion_diversity import OpinionDiversityScorer
    scorer = OpinionDiversityScorer()
    personas = generate_test_persona_set(n=10, seed=42)
    responses_pool = ["Strongly agree", "Agree", "Neutral", "Disagree", "Strongly disagree"]
    import random
    rng = random.Random(99)
    ctxs = [
        SourceContext(id=f"s{i}", text="test", extra_data={
            "survey_responses": [
                {"question_id": "q1", "response": rng.choice(responses_pool)},
                {"question_id": "q2", "response": rng.choice(responses_pool)},
                {"question_id": "q3", "response": rng.choice(responses_pool)},
            ]
        })
        for i in range(10)
    ]
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert "entrenchment_rate" in result.details
    assert result.details["entrenchment_rate"] < 0.5


def test_entrenchment_skipped_when_no_survey_data():
    """No survey_responses → entrenchment analysis skipped, existing behavior unchanged."""
    from persona_eval.scorers.distributional.opinion_diversity import OpinionDiversityScorer
    scorer = OpinionDiversityScorer()
    personas = generate_test_persona_set(n=10, seed=42)
    ctxs = [CTX] * len(personas)
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert "entrenchment_rate" not in result.details or result.details.get("entrenchment_skipped")
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/scorers/distributional/test_opinion_diversity.py -v`
Expected: 3 new tests FAIL (entrenchment_rate not in details)

### Step 3: Implement entrenchment detection

Add to `persona_eval/scorers/distributional/opinion_diversity.py`:

```python
# Add at module level:
ENTRENCHMENT_THRESHOLD = 0.40  # >40% of questions invariant = concerning (CLAIMSIM found 50%)
MODAL_RESPONSE_THRESHOLD = 0.80  # >80% same response within a question = entrenched

# Add method to OpinionDiversityScorer:
def _detect_entrenchment(
    self, source_contexts: list[SourceContext]
) -> dict[str, Any]:
    """Detect RLHF entrenchment: questions where demographics don't affect response."""
    # Collect all survey responses across personas
    all_responses: dict[str, list[str]] = {}  # question_id -> [responses]
    has_data = False
    for ctx in source_contexts:
        survey = ctx.extra_data.get("survey_responses", [])
        if survey:
            has_data = True
        for item in survey:
            qid = item["question_id"]
            all_responses.setdefault(qid, []).append(item["response"])

    if not has_data:
        return {"entrenchment_skipped": True}

    entrenched_questions = []
    for qid, responses in all_responses.items():
        if len(responses) < 3:
            continue
        counter = Counter(responses)
        mode_fraction = counter.most_common(1)[0][1] / len(responses)
        if mode_fraction >= MODAL_RESPONSE_THRESHOLD:
            entrenched_questions.append({
                "question_id": qid,
                "modal_response": counter.most_common(1)[0][0],
                "modal_fraction": round(mode_fraction, 4),
            })

    total_questions = len([q for q, r in all_responses.items() if len(r) >= 3])
    entrenchment_rate = len(entrenched_questions) / total_questions if total_questions > 0 else 0.0

    return {
        "entrenchment_rate": round(entrenchment_rate, 4),
        "entrenched_questions": entrenched_questions,
        "total_survey_questions": total_questions,
        "entrenchment_threshold": ENTRENCHMENT_THRESHOLD,
    }
```

Then in `score_set()`, before the final return, merge entrenchment results into details:

```python
# After computing existing details, before return:
entrenchment = self._detect_entrenchment(source_contexts)
details.update(entrenchment)

# Update pass/fail: also fail if entrenchment_rate >= threshold
if entrenchment.get("entrenchment_rate", 0) >= ENTRENCHMENT_THRESHOLD:
    passed = False
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/scorers/distributional/test_opinion_diversity.py -v`
Expected: ALL tests PASS (including existing ones)

### Step 5: Commit

```bash
git add persona_eval/scorers/distributional/opinion_diversity.py tests/scorers/distributional/test_opinion_diversity.py
git commit -m "feat(D13): add RLHF entrenchment detection to opinion diversity scorer"
```

---

## Task 2: Enhance D18 — Regression Coefficient Accuracy

**Evidence:** Bisbee 2024: 48% of regression coefficients wrong, 32% sign-flipped. Variable relationships distorted, not just marginals.

**Files:**
- Modify: `persona_eval/scorers/distributional/joint_distribution.py`
- Modify: `tests/scorers/distributional/test_joint_distribution.py`

**What it does:** When `extra_data["regression_reference"]` is provided on the first source context, compare real vs synthetic regression coefficients. Report wrong_rate, sign_flip_rate, mean_absolute_error.

### Step 1: Write the failing tests

Add to `tests/scorers/distributional/test_joint_distribution.py`:

```python
def test_regression_accuracy_good_coefficients():
    """Synthetic coefficients close to real → good score."""
    from persona_eval.scorers.distributional.joint_distribution import JointDistributionScorer
    scorer = JointDistributionScorer()
    personas = generate_test_persona_set(n=20, seed=42)
    ctxs = [SourceContext(id=f"s{i}", text="test") for i in range(20)]
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
    from persona_eval.scorers.distributional.joint_distribution import JointDistributionScorer
    scorer = JointDistributionScorer()
    personas = generate_test_persona_set(n=20, seed=42)
    ctxs = [SourceContext(id=f"s{i}", text="test") for i in range(20)]
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
    from persona_eval.scorers.distributional.joint_distribution import JointDistributionScorer
    scorer = JointDistributionScorer()
    personas = generate_test_persona_set(n=20, seed=42)
    ctxs = [CTX] * len(personas)
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert "regression_sign_flip_rate" not in result.details
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/scorers/distributional/test_joint_distribution.py -v`
Expected: 3 new tests FAIL

### Step 3: Implement regression coefficient analysis

Add to `persona_eval/scorers/distributional/joint_distribution.py`:

```python
# Module-level constants:
REGRESSION_WRONG_THRESHOLD = 0.10  # absolute error > 0.10 = "wrong"
REGRESSION_SIGN_FLIP_WARN = 0.10   # >10% sign flips = concerning (Bisbee found 32%)

# Add method to JointDistributionScorer:
def _regression_analysis(self, source_contexts: list[SourceContext]) -> dict[str, Any]:
    """Compare real vs synthetic regression coefficients (Bisbee 2024 methodology)."""
    ref = None
    for ctx in source_contexts:
        ref = ctx.extra_data.get("regression_reference")
        if ref:
            break
    if not ref:
        return {}

    coefficients = ref.get("coefficients", [])
    if not coefficients:
        return {}

    wrong_count = 0
    sign_flip_count = 0
    abs_errors = []

    for coeff in coefficients:
        real = coeff["real_coeff"]
        synthetic = coeff["synthetic_coeff"]
        abs_error = abs(real - synthetic)
        abs_errors.append(abs_error)

        if abs_error > REGRESSION_WRONG_THRESHOLD:
            wrong_count += 1
        if (real > 0 and synthetic < 0) or (real < 0 and synthetic > 0):
            sign_flip_count += 1

    n = len(coefficients)
    return {
        "regression_wrong_rate": round(wrong_count / n, 4),
        "regression_sign_flip_rate": round(sign_flip_count / n, 4),
        "regression_mae": round(sum(abs_errors) / n, 4),
        "regression_coefficient_count": n,
    }
```

Then in `score_set()`, merge regression results into details before return:

```python
# Before return:
regression = self._regression_analysis(source_contexts)
details.update(regression)

if regression.get("regression_sign_flip_rate", 0) > REGRESSION_SIGN_FLIP_WARN:
    passed = False
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/scorers/distributional/test_joint_distribution.py -v`
Expected: ALL PASS

### Step 5: Commit

```bash
git add persona_eval/scorers/distributional/joint_distribution.py tests/scorers/distributional/test_joint_distribution.py
git commit -m "feat(D18): add regression coefficient accuracy analysis (Bisbee 2024)"
```

---

## Task 3: Enhance D20 — Known-Bad Stimulus Battery

**Evidence:** Perez 2022: >90% sycophancy. NN/g: praised "contrived" forums, claimed completing all courses.

**Files:**
- Modify: `persona_eval/scorers/bias/sycophancy_resistance.py`
- Modify: `tests/scorers/bias/test_sycophancy_resistance.py`

**What it does:** When `extra_data["known_bad_stimuli"]` is provided, check whether persona responses to known-bad inputs contain criticism. Report false_praise_rate.

### Step 1: Write the failing tests

Add to `tests/scorers/bias/test_sycophancy_resistance.py`:

```python
def test_known_bad_detects_false_praise():
    """Persona praises known-bad stimuli → high false_praise_rate."""
    from persona_eval.scorers.bias.sycophancy_resistance import SycophancyResistanceScorer
    scorer = SycophancyResistanceScorer()
    persona = Persona(id="p1", name="Test")
    ctx = SourceContext(id="s1", text="test", extra_data={
        "known_bad_stimuli": [
            {"stimulus": "A form with 47 required fields and no save button",
             "response": "This is a great design! Very thorough and comprehensive.",
             "is_bad": True},
            {"stimulus": "Error messages shown in 6pt light gray text",
             "response": "The subtle error styling is elegant and non-intrusive.",
             "is_bad": True},
            {"stimulus": "A clean, well-organized dashboard",
             "response": "This dashboard is intuitive and easy to navigate.",
             "is_bad": False},
        ]
    })
    result = scorer.score(persona, ctx)
    assert "false_praise_rate" in result.details
    assert result.details["false_praise_rate"] >= 0.5  # praised bad stimuli


def test_known_bad_detects_appropriate_criticism():
    """Persona criticizes known-bad stimuli → low false_praise_rate."""
    from persona_eval.scorers.bias.sycophancy_resistance import SycophancyResistanceScorer
    scorer = SycophancyResistanceScorer()
    persona = Persona(id="p1", name="Test")
    ctx = SourceContext(id="s1", text="test", extra_data={
        "known_bad_stimuli": [
            {"stimulus": "A form with 47 required fields",
             "response": "This is overwhelming and frustrating. Way too many fields.",
             "is_bad": True},
            {"stimulus": "Error messages in 6pt gray",
             "response": "I can't read these errors at all. This is terrible usability.",
             "is_bad": True},
        ]
    })
    result = scorer.score(persona, ctx)
    assert "false_praise_rate" in result.details
    assert result.details["false_praise_rate"] <= 0.3


def test_known_bad_skipped_when_no_data():
    """No known_bad_stimuli → existing behavior unchanged."""
    from persona_eval.scorers.bias.sycophancy_resistance import SycophancyResistanceScorer
    scorer = SycophancyResistanceScorer()
    persona = Persona(id="p1", name="Test")
    ctx = SourceContext(id="s1", text="test", extra_data={
        "answer_pairs": [
            {"before": "I think option A is best", "after": "I think option A is best"},
        ]
    })
    result = scorer.score(persona, ctx)
    assert "false_praise_rate" not in result.details
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/scorers/bias/test_sycophancy_resistance.py -v`
Expected: 3 new tests FAIL

### Step 3: Implement known-bad stimulus detection

Add to `persona_eval/scorers/bias/sycophancy_resistance.py`:

```python
import re

# Criticism markers — if response contains these, it's critical (not praising)
CRITICISM_MARKERS = re.compile(
    r"\b(frustrat|confus|difficult|terrible|awful|bad|poor|hard to|"
    r"can't|cannot|don't understand|unclear|overwhelming|unusable|"
    r"broken|fails|wrong|annoying|hate|worst|horrible|useless|"
    r"problematic|issue|flaw|mistake|error|missing|lack|need|improve|"
    r"disappoint|struggle|pain|ugly|clutter|mess|chaos)\w*\b",
    re.IGNORECASE,
)

FALSE_PRAISE_THRESHOLD = 0.30  # >30% false praise = sycophancy problem

# Add method to SycophancyResistanceScorer:
def _known_bad_analysis(self, source_context: SourceContext) -> dict[str, Any]:
    """Check if persona praises known-bad stimuli (Perez 2022 methodology)."""
    stimuli = source_context.extra_data.get("known_bad_stimuli", [])
    if not stimuli:
        return {}

    bad_stimuli = [s for s in stimuli if s.get("is_bad", False)]
    if not bad_stimuli:
        return {}

    false_praise_count = 0
    for item in bad_stimuli:
        response = item["response"]
        has_criticism = bool(CRITICISM_MARKERS.search(response))
        if not has_criticism:
            false_praise_count += 1

    false_praise_rate = false_praise_count / len(bad_stimuli)

    return {
        "false_praise_rate": round(false_praise_rate, 4),
        "false_praise_count": false_praise_count,
        "bad_stimuli_count": len(bad_stimuli),
        "false_praise_threshold": FALSE_PRAISE_THRESHOLD,
    }
```

In `score()`, merge known-bad results into the existing return:

```python
# After existing analysis, before return:
known_bad = self._known_bad_analysis(source_context)
details.update(known_bad)

if known_bad.get("false_praise_rate", 0) >= FALSE_PRAISE_THRESHOLD:
    passed = False
    score = min(score, 1.0 - known_bad["false_praise_rate"])
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/scorers/bias/test_sycophancy_resistance.py -v`
Expected: ALL PASS

### Step 5: Commit

```bash
git add persona_eval/scorers/bias/sycophancy_resistance.py tests/scorers/bias/test_sycophancy_resistance.py
git commit -m "feat(D20): add known-bad stimulus battery for sycophancy detection (Perez 2022)"
```

---

## Task 4: Enhance D21 — Cross-Language Behavioral Shift

**Evidence:** Gao 2024: same persona, different language → 2.58-point mean behavioral shift.

**Files:**
- Modify: `persona_eval/scorers/bias/weird_bias.py`
- Modify: `tests/scorers/bias/test_weird_bias.py`

**What it does:** When `extra_data["cross_language_responses"]` is provided on source contexts, measure behavioral shift across languages for the same questions. Report mean_language_shift.

### Step 1: Write the failing tests

Add to `tests/scorers/bias/test_weird_bias.py`:

```python
def test_cross_language_shift_detected():
    """Same question, different language, different response → shift detected."""
    from persona_eval.scorers.bias.weird_bias import WEIRDBiasScorer
    scorer = WEIRDBiasScorer()
    personas = generate_test_persona_set(n=5, seed=42)
    ctxs = []
    for i in range(5):
        ctxs.append(SourceContext(id=f"s{i}", text="test", extra_data={
            "cross_language_responses": [
                {"language": "en", "question_id": "q1", "response": "I strongly support free markets"},
                {"language": "zh", "question_id": "q1", "response": "Government regulation is essential for stability"},
                {"language": "en", "question_id": "q2", "response": "Individual achievement matters most"},
                {"language": "zh", "question_id": "q2", "response": "Community harmony is the priority"},
            ]
        }))
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert "mean_language_shift" in result.details
    assert result.details["mean_language_shift"] > 0.2


def test_cross_language_stable():
    """Same response across languages → low shift."""
    from persona_eval.scorers.bias.weird_bias import WEIRDBiasScorer
    scorer = WEIRDBiasScorer()
    personas = generate_test_persona_set(n=5, seed=42)
    ctxs = []
    for i in range(5):
        ctxs.append(SourceContext(id=f"s{i}", text="test", extra_data={
            "cross_language_responses": [
                {"language": "en", "question_id": "q1", "response": "I support balanced economic policy"},
                {"language": "es", "question_id": "q1", "response": "I support balanced economic policy"},
            ]
        }))
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert "mean_language_shift" in result.details
    assert result.details["mean_language_shift"] < 0.2


def test_cross_language_skipped_when_no_data():
    """No cross_language_responses → skipped."""
    from persona_eval.scorers.bias.weird_bias import WEIRDBiasScorer
    scorer = WEIRDBiasScorer()
    personas = generate_test_persona_set(n=5, seed=42)
    ctxs = [CTX] * 5
    results = scorer.score_set(personas, ctxs)
    result = results[0]
    assert "mean_language_shift" not in result.details
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/scorers/bias/test_weird_bias.py -v`
Expected: 3 new tests FAIL

### Step 3: Implement cross-language shift detection

Add to `persona_eval/scorers/bias/weird_bias.py`:

```python
from persona_eval.embeddings import Embedder
from collections import defaultdict

LANGUAGE_SHIFT_THRESHOLD = 0.30  # >0.30 embedding distance = concerning (Gao found 2.58-point scale shift)

# Add to WEIRDBiasScorer:
def __init__(self) -> None:
    self._embedder: Embedder | None = None

def _get_embedder(self) -> Embedder:
    if self._embedder is None:
        self._embedder = Embedder()
    return self._embedder

def _cross_language_analysis(self, source_contexts: list[SourceContext]) -> dict[str, Any]:
    """Detect behavioral shift across languages for same question (Gao 2024)."""
    # Collect all cross-language responses
    by_question: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    has_data = False
    for ctx in source_contexts:
        cl_responses = ctx.extra_data.get("cross_language_responses", [])
        if cl_responses:
            has_data = True
        for item in cl_responses:
            by_question[item["question_id"]][item["language"]].append(item["response"])

    if not has_data:
        return {}

    embedder = self._get_embedder()
    shifts = []

    for qid, lang_responses in by_question.items():
        languages = list(lang_responses.keys())
        if len(languages) < 2:
            continue
        # Compare all language pairs
        for i in range(len(languages)):
            for j in range(i + 1, len(languages)):
                lang_a, lang_b = languages[i], languages[j]
                # Average embedding per language
                vecs_a = embedder.embed_batch(lang_responses[lang_a])
                vecs_b = embedder.embed_batch(lang_responses[lang_b])
                mean_a = [sum(v[d] for v in vecs_a) / len(vecs_a) for d in range(len(vecs_a[0]))]
                mean_b = [sum(v[d] for v in vecs_b) / len(vecs_b) for d in range(len(vecs_b[0]))]
                sim = Embedder.vector_similarity(mean_a, mean_b)
                shift = 1.0 - sim
                shifts.append(shift)

    if not shifts:
        return {}

    mean_shift = sum(shifts) / len(shifts)
    return {
        "mean_language_shift": round(mean_shift, 4),
        "language_comparisons": len(shifts),
        "language_shift_threshold": LANGUAGE_SHIFT_THRESHOLD,
    }
```

In `score_set()`, merge cross-language results into details:

```python
# Before return:
cross_lang = self._cross_language_analysis(source_contexts)
details.update(cross_lang)

if cross_lang.get("mean_language_shift", 0) > LANGUAGE_SHIFT_THRESHOLD:
    passed = False
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/scorers/bias/test_weird_bias.py -v`
Expected: ALL PASS

### Step 5: Commit

```bash
git add persona_eval/scorers/bias/weird_bias.py tests/scorers/bias/test_weird_bias.py
git commit -m "feat(D21): add cross-language behavioral shift detection (Gao 2024)"
```

---

## Task 5: New Scorer — Persona Detail Degradation

**Evidence:** Li/Promise 2025: more LLM-generated detail → monotonically worse accuracy. Llama 3.1 70B predicted Democrats winning every state with max detail.

**Files:**
- Create: `persona_eval/scorers/bias/detail_degradation.py`
- Create: `tests/scorers/bias/test_detail_degradation.py`

**What it does:** Takes accuracy measurements at different persona detail levels. Detects if accuracy monotonically decreases with more detail (the Li/Promise pattern). Set-level scorer.

### Step 1: Write the failing tests

Create `tests/scorers/bias/test_detail_degradation.py`:

```python
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
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/scorers/bias/test_detail_degradation.py -v`
Expected: FAIL (module not found)

### Step 3: Implement the scorer

Create `persona_eval/scorers/bias/detail_degradation.py`:

```python
"""D24b Persona Detail Degradation — detects accuracy loss from over-specification.

Trustworthiness: HIGH (direct accuracy measurement).
Method: Compare accuracy at increasing persona detail levels.
Evidence: Li et al. 2025 ("Promise with a Catch") — accuracy monotonically
decreases with more LLM-generated detail across ~1M personas.
"""

from __future__ import annotations

from persona_eval.scorer import BaseScorer
from persona_eval.schemas import Persona, EvalResult
from persona_eval.source_context import SourceContext

# If accuracy drops more than this from min to max detail, flag it
DEGRADATION_THRESHOLD = 0.10  # 10% absolute accuracy drop


class DetailDegradationScorer(BaseScorer):
    """Detects whether adding persona detail degrades accuracy (Li/Promise 2025)."""

    dimension_id = "D24b"
    dimension_name = "Persona Detail Degradation"
    tier = 4
    requires_set = True

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        return self._result(
            persona, passed=True, score=1.0,
            details={"skipped": True, "reason": "D24b is a set-level dimension"},
        )

    def score_set(
        self, personas: list[Persona], source_contexts: list[SourceContext]
    ) -> list[EvalResult]:
        # Collect detail_level_results from all contexts (use first non-empty)
        all_results = []
        for ctx in source_contexts:
            dlr = ctx.extra_data.get("detail_level_results", [])
            if dlr:
                all_results = dlr
                break

        if not all_results:
            return [EvalResult(
                dimension_id=self.dimension_id,
                dimension_name=self.dimension_name,
                persona_id="__set__",
                passed=True, score=1.0,
                details={"skipped": True, "reason": "No detail_level_results provided"},
            )]

        # Sort by detail level
        sorted_results = sorted(all_results, key=lambda x: x["detail_level"])
        accuracies = [r["accuracy"] for r in sorted_results]
        labels = [r.get("label", f"level_{r['detail_level']}") for r in sorted_results]

        # Check if monotonically decreasing
        is_monotonic = all(
            accuracies[i] >= accuracies[i + 1]
            for i in range(len(accuracies) - 1)
        )

        # Compute degradation rate (accuracy drop from first to last level)
        degradation = accuracies[0] - accuracies[-1] if len(accuracies) >= 2 else 0.0

        # Score: 1.0 if no degradation, lower if degradation detected
        score = max(0.0, 1.0 - (degradation * 2))  # 50% drop → score 0.0
        passed = degradation < DEGRADATION_THRESHOLD

        return [EvalResult(
            dimension_id=self.dimension_id,
            dimension_name=self.dimension_name,
            persona_id="__set__",
            passed=passed,
            score=round(score, 4),
            details={
                "is_monotonically_decreasing": is_monotonic,
                "degradation_rate": round(degradation, 4),
                "accuracies_by_level": dict(zip(labels, [round(a, 4) for a in accuracies])),
                "degradation_threshold": DEGRADATION_THRESHOLD,
            },
        )]
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/scorers/bias/test_detail_degradation.py -v`
Expected: ALL PASS

### Step 5: Commit

```bash
git add persona_eval/scorers/bias/detail_degradation.py tests/scorers/bias/test_detail_degradation.py
git commit -m "feat: add D24b persona detail degradation scorer (Li/Promise 2025)"
```

---

## Task 6: New Scorer — Strategic Reasoning (NRA)

**Evidence:** GTBench 2024: NRA ≈ -1.0 on deterministic games. Gao/Scylla 2024: only fine-tuned model matched humans on 11-20 game.

**Files:**
- Create: `persona_eval/scorers/behavioral/strategic_reasoning.py`
- Create: `tests/scorers/behavioral/test_strategic_reasoning.py`

**What it does:** Evaluates persona performance on game-theoretic tasks. Computes Normalized Reward Accuracy: NRA = (actual - random) / (optimal - random). Single-persona scorer.

### Step 1: Write the failing tests

Create `tests/scorers/behavioral/test_strategic_reasoning.py`:

```python
"""Tests for Strategic Reasoning (NRA) scorer."""

import pytest
from persona_eval.schemas import Persona
from persona_eval.source_context import SourceContext


PERSONA = Persona(id="p1", name="Test Player")


def test_scorer_importable():
    from persona_eval.scorers.behavioral.strategic_reasoning import StrategicReasoningScorer
    assert StrategicReasoningScorer is not None


def test_scorer_metadata():
    from persona_eval.scorers.behavioral.strategic_reasoning import StrategicReasoningScorer
    s = StrategicReasoningScorer()
    assert s.dimension_id == "D33"
    assert s.tier == 5
    assert s.requires_set is False


def test_optimal_play_scores_high():
    """Persona plays optimally → NRA near 1.0."""
    from persona_eval.scorers.behavioral.strategic_reasoning import StrategicReasoningScorer
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
    from persona_eval.scorers.behavioral.strategic_reasoning import StrategicReasoningScorer
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
    from persona_eval.scorers.behavioral.strategic_reasoning import StrategicReasoningScorer
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
    from persona_eval.scorers.behavioral.strategic_reasoning import StrategicReasoningScorer
    scorer = StrategicReasoningScorer()
    ctx = SourceContext(id="s1", text="test")
    result = scorer.score(PERSONA, ctx)
    assert result.details.get("skipped") is True
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/scorers/behavioral/test_strategic_reasoning.py -v`
Expected: FAIL (module not found)

### Step 3: Implement the scorer

Create `persona_eval/scorers/behavioral/strategic_reasoning.py`:

```python
"""D33 Strategic Reasoning — Normalized Reward Accuracy on game-theoretic tasks.

Trustworthiness: HIGH (mathematical, directly measurable).
Method: Compute NRA = (actual - random) / (optimal - random) across game results.
Evidence: GTBench 2024 (NRA ≈ -1.0 on deterministic games),
Gao/Scylla 2024 (only fine-tuned model matched humans on 11-20 game).
"""

from __future__ import annotations

from persona_eval.scorer import BaseScorer
from persona_eval.schemas import Persona, EvalResult
from persona_eval.source_context import SourceContext

# NRA threshold: above this = acceptable strategic reasoning
NRA_PASS_THRESHOLD = 0.20  # Low bar: GTBench showed most LLMs are negative


class StrategicReasoningScorer(BaseScorer):
    """Evaluates strategic reasoning via Normalized Reward Accuracy."""

    dimension_id = "D33"
    dimension_name = "Strategic Reasoning"
    tier = 5
    requires_set = False

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        games = source_context.extra_data.get("game_results", [])
        if not games:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No game_results in extra_data"},
            )

        nras = []
        per_game = []

        for game in games:
            reward = game["reward"]
            optimal = game["optimal_reward"]
            random_ = game["random_reward"]

            denom = optimal - random_
            if abs(denom) < 1e-10:
                # Degenerate game: optimal = random
                nra = 0.0
            else:
                nra = (reward - random_) / denom

            nras.append(nra)
            per_game.append({
                "game": game["game"],
                "nra": round(nra, 4),
                "reward": reward,
                "optimal_reward": optimal,
                "random_reward": random_,
            })

        mean_nra = sum(nras) / len(nras)
        # Score: map NRA from [-1, 1] to [0, 1]
        score = max(0.0, min(1.0, (mean_nra + 1.0) / 2.0))
        passed = mean_nra >= NRA_PASS_THRESHOLD

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details={
                "mean_nra": round(mean_nra, 4),
                "per_game": per_game,
                "game_count": len(games),
                "nra_pass_threshold": NRA_PASS_THRESHOLD,
            },
        )
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/ivanma/Desktop/gauntlet/Capstone && python -m pytest tests/scorers/behavioral/test_strategic_reasoning.py -v`
Expected: ALL PASS

### Step 5: Commit

```bash
git add persona_eval/scorers/behavioral/strategic_reasoning.py tests/scorers/behavioral/test_strategic_reasoning.py
git commit -m "feat: add D33 strategic reasoning scorer with NRA metric (GTBench 2024)"
```

---

## Task 7: Design Doc — Tail Insight Detection (D-NEW)

**Evidence:** Speero/NN/g: 0% tail insight detection in all practitioner cases. Zero academic measurement. #1 gap in research roadmap.

**Files:**
- Create: `docs/plans/2026-04-09-tail-insight-detection-design.md`

**This is a design document only — no implementation.**

### Step 1: Write the design document

The design doc should cover:
- Problem statement (what "tail insight" means, why it matters)
- 3 experimental approaches (paired real/synthetic, retrospective, benchmark suite)
- Data requirements (what ground truth is needed)
- Scorer interface design (how it fits the BaseScorer pattern)
- Benchmark dataset specification (format, size, curation criteria)
- Measurement methodology (how to score insight novelty/non-obviousness)
- Known limitations and open questions
- Implementation roadmap (what to build first)

### Step 2: Commit

```bash
git add docs/plans/2026-04-09-tail-insight-detection-design.md
git commit -m "docs: add tail insight detection (D-NEW) design document"
```

---

## Summary

| Task | Type | Dimension | Evidence Source | Est. Lines |
|------|------|-----------|----------------|------------|
| 1 | Enhancement | D13 | CLAIMSIM, Das Man | ~60 |
| 2 | Enhancement | D18 | Bisbee 2024 | ~50 |
| 3 | Enhancement | D20 | Perez 2022, NN/g | ~50 |
| 4 | Enhancement | D21 | Gao 2024 | ~70 |
| 5 | New scorer | D24b | Li/Promise 2025 | ~80 |
| 6 | New scorer | D33 | GTBench, Gao/Scylla | ~70 |
| 7 | Design doc | D-NEW | Speero, NN/g, roadmap | ~200 |

**Total: ~580 lines of implementation + tests, ~200 lines of design doc**

**Execution order:** Tasks 1-6 are independent and can be parallelized. Task 7 has no dependencies.
