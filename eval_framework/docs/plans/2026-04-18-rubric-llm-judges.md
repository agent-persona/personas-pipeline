# Implementation Plan: Rubric-Based LLM Judge Scorers + Proxy Validation

**Date:** 2026-04-18  
**Goal:** Add 5 rubric-based LLM-as-judge scorers (J1–J5), a proxy correlation validator, and full unit tests.

## Overview

Current eval framework uses mostly heuristic proxies (embedding similarity, regex, field checks). This plan adds ground-truth LLM judges that evaluate what actually matters — authenticity, voice, values, depth, adaptation — and validates which existing proxies track those qualities.

---

## Part 1: Judge Scorer Infrastructure

### Task 1.1 — Create `persona_eval/scorers/judge/__init__.py`

Empty package init file.

**Verification:** `python -c "import persona_eval.scorers.judge"` exits 0.

---

### Task 1.2 — Create `persona_eval/scorers/judge/_base.py`

Shared utilities for all 5 judge scorers:

```python
"""Shared utilities for LLM-as-judge rubric scorers."""
from __future__ import annotations
import json
import re
from persona_eval.llm_client import LLMClient

_SCORE_RE = re.compile(r'"score"\s*:\s*([1-5](?:\.\d+)?)')
JUDGE_MODEL = "claude-sonnet-4-20250514"

def make_client() -> LLMClient:
    return LLMClient(model=JUDGE_MODEL, temperature=0.0)

def parse_score(raw: str) -> tuple[float, str, bool]:
    """Return (raw_1_to_5_float, reasoning_str, parse_ok_bool).
    Tries JSON first; falls back to regex; returns (3.0, raw[:200], False) on failure.
    """
    try:
        data = json.loads(raw)
        return float(data["score"]), data.get("reasoning", ""), True
    except Exception:
        pass
    m = _SCORE_RE.search(raw)
    if m:
        return float(m.group(1)), raw[:200], True
    return 3.0, raw[:200], False

def normalize(raw_score: float) -> float:
    """Map 1–5 to 0.0–1.0, clamped."""
    return round(max(0.0, min(1.0, (raw_score - 1) / 4)), 4)

def build_persona_block(persona) -> str:
    """Compact persona text block for LLM prompts. Excludes empty fields."""
    lines = [
        f"Name: {persona.name}",
        f"Age: {persona.age}",
        f"Occupation: {persona.occupation}",
        f"Industry: {persona.industry}",
        f"Education: {persona.education}",
        f"Location: {persona.location}",
        f"Personality traits: {', '.join(persona.personality_traits)}",
        f"Values: {', '.join(persona.values)}",
        f"Goals: {', '.join(persona.goals)}",
        f"Pain points: {', '.join(persona.pain_points)}",
        f"Communication tone: {persona.communication_style.tone}",
        f"Communication formality: {persona.communication_style.formality}",
        f"Vocabulary level: {persona.communication_style.vocabulary_level}",
        f"Baseline mood: {persona.emotional_profile.baseline_mood}",
        f"Core values (moral): {', '.join(persona.moral_framework.core_values)}",
        f"Bio: {persona.bio}",
    ]
    return "\n".join(l for l in lines if not l.endswith(": "))
```

**Verification:** `python -c "from persona_eval.scorers.judge._base import parse_score, normalize, build_persona_block"` succeeds.

---

### Task 1.3 — Create `persona_eval/scorers/judge/j1_behavioral_authenticity.py`

**Class:** `BehavioralAuthenticityScorer`, `dimension_id="J1"`, `tier=3`, `requires_set=False`

**extra_data key:** `"responses"` (list of strings, cap at 5)

**Rubric system prompt:**
```
You are an expert evaluator of AI persona quality. Your task is to assess whether a persona behaves like a genuine human being with the stated background, or like a generic LLM assistant wearing a thin costume.

Rate the persona on BEHAVIORAL AUTHENTICITY using the following 1–5 rubric:

1 = LLM Template
   • Responses could belong to any generic assistant
   • No trace of the stated occupation, background, or personality traits
   • Uses LLM-typical phrases: "Certainly!", "Great question!", "I'd be happy to..."
   • Emotions are uniformly positive; no friction, no contradictions

2 = Thin Disguise
   • Mentions name or job title but doesn't embody them
   • Vocabulary and reasoning style do not match stated expertise level
   • Occasional hedge phrases break the persona
   • Background facts appear as rote recitation, not lived experience

3 = Adequate
   • Usually sounds like the stated person; slips into generic voice occasionally
   • Domain knowledge is present but surface-level
   • Mostly avoids LLM artifacts but not consistently

4 = Convincing
   • Strong alignment between stated background and response style
   • Domain knowledge is specific and plausibly earned
   • Opinions and friction points are consistent with stated values
   • Very few LLM-typical artifacts

5 = Fully Realized
   • Indistinguishable from a real person with this background
   • Every response inflected by occupation, culture, emotional profile, and values simultaneously
   • Contradictions and vulnerabilities present where biography would predict them
   • Zero LLM artifacts

Respond ONLY with JSON: {"score": N, "reasoning": "one or two sentences"}
```

**score() logic:**
- Get `responses` from `extra_data`, skip if empty
- Build persona block + format up to 5 responses
- Call LLM, parse response with `parse_score()`
- `passed = normalize(raw_score) >= 0.6`
- Details: `raw_score`, `reasoning`, `parse_ok`, `response_count`, `rubric`

**Verification:** `python -c "from persona_eval.scorers.judge.j1_behavioral_authenticity import BehavioralAuthenticityScorer; s = BehavioralAuthenticityScorer(); print(s.dimension_id)"` prints `J1`.

---

### Task 1.4 — Create `persona_eval/scorers/judge/j2_voice_consistency.py`

**Class:** `VoiceConsistencyScorer`, `dimension_id="J2"`, `tier=3`, `requires_set=False`

**extra_data key:** `"responses"` (same as J1)

**Rubric system prompt:**
```
You are an expert evaluator of AI persona quality. Assess whether the persona's voice is distinctive and consistent across multiple responses, or generic and shifting.

Rate the persona on VOICE CONSISTENCY using the following 1–5 rubric:

1 = No Distinct Voice
   • Each response could have been written by a different assistant
   • No consistent vocabulary, sentence rhythm, or rhetorical habits
   • Tone shifts wildly between responses

2 = Weak Voice
   • Slight tendency toward a style but easily lost when topic changes
   • Formality level inconsistent with stated background
   • Generic connectors dominate ("Additionally...", "In conclusion...")

3 = Recognizable Voice
   • Moderately distinctive style in most responses
   • Formality roughly appropriate; occasional slippage into generic prose

4 = Consistent Voice
   • Same rhythm, vocabulary, and tone across all responses
   • Style matches stated communication profile
   • Persona "sounds like themselves" even on unfamiliar topics

5 = Signature Voice
   • Instantly recognizable — identifiable from a response alone
   • Idiosyncratic phrasing and rhythm that are entirely their own
   • Style is consistent AND organically tied to background

Respond ONLY with JSON: {"score": N, "reasoning": "one or two sentences"}
```

**Verification:** `python -c "from persona_eval.scorers.judge.j2_voice_consistency import VoiceConsistencyScorer"` succeeds.

---

### Task 1.5 — Create `persona_eval/scorers/judge/j3_value_alignment.py`

**Class:** `ValueAlignmentScorer`, `dimension_id="J3"`, `tier=3`, `requires_set=False`

**extra_data key:** `"responses"`

**User prompt addition** — highlight values explicitly:
```python
user_msg = (
    f"PERSONA DEFINITION:\n{persona_block}\n\n"
    f"STATED VALUES: {', '.join(persona.values)}\n"
    f"ETHICAL STANCE: {persona.moral_framework.ethical_stance}\n\n"
    f"SAMPLE RESPONSES:\n{responses_block}\n\n"
    "Rate value alignment on the 1–5 rubric."
)
```

**Rubric system prompt:**
```
You are an expert evaluator of AI persona quality. Assess whether the persona acts on its stated values in practice, or merely names them in a résumé-style list.

Rate the persona on VALUE ALIGNMENT using the following 1–5 rubric:

1 = Values as Decoration
   • Stated values appear nowhere in reasoning or behavior
   • Responses contradict stated values without acknowledgment
   • Values read like a generic aspirational template

2 = Superficial Reference
   • Values occasionally name-dropped but don't shape decisions
   • Trade-off reasoning ignores stated moral framework
   • Ethical stance has no visible influence on responses

3 = Partial Alignment
   • Values appear in some reasoning but sometimes forgotten
   • Some responses fit any persona; few are persona-specific

4 = Clear Alignment
   • Most responses reflect stated values in content and framing
   • Ethical stance shapes how trade-offs are resolved
   • Values produce distinctive positions others would not take

5 = Deeply Internalized
   • Values embedded in every response — not stated but enacted
   • Ethical stance produces specific, sometimes surprising conclusions
   • Tension between competing values handled consistently

Respond ONLY with JSON: {"score": N, "reasoning": "one or two sentences"}
```

**Verification:** Import succeeds and `dimension_id == "J3"`.

---

### Task 1.6 — Create `persona_eval/scorers/judge/j4_persona_depth.py`

**Class:** `PersonaDepthScorer`, `dimension_id="J4"`, `tier=3`, `requires_set=False`

**extra_data key:** `"responses"`

**User prompt addition** — highlight bio:
```python
user_msg = (
    f"PERSONA DEFINITION:\n{persona_block}\n\n"
    f"BIO (key source for depth):\n{persona.bio}\n\n"
    f"SAMPLE RESPONSES:\n{responses_block}\n\n"
    "Rate persona depth on the 1–5 rubric."
)
```

**Rubric system prompt:**
```
You are an expert evaluator of AI persona quality. Assess whether this is a fully realized person with interior life, or a shallow demographic summary.

Rate the persona on PERSONA DEPTH using the following 1–5 rubric:

1 = Demographic Skeleton
   • Feels like a form filled in with required fields
   • No sense of history, contradiction, or interiority
   • Responses treat background facts as data points, not lived experience

2 = Thin Profile
   • Some interesting detail but lacks depth
   • Biography reads as a list of achievements, not a life
   • No vulnerability, regret, or ambiguity visible

3 = Moderate Depth
   • A real person could have this profile, but not fully inhabited
   • A few specific, non-generic details bring it to life

4 = Realized Person
   • Strong sense of a specific individual with history
   • Responses draw on backstory organically
   • At least one unexpected dimension (contradiction, private struggle, unlikely interest)

5 = Full Interior Life
   • Responses reveal a complete human being with aspirations, wounds, habits of mind
   • Surprises you with specificity that feels discovered, not constructed
   • Multiple dimensions of depth operating simultaneously

Respond ONLY with JSON: {"score": N, "reasoning": "one or two sentences"}
```

**Verification:** Import succeeds and `dimension_id == "J4"`.

---

### Task 1.7 — Create `persona_eval/scorers/judge/j5_contextual_adaptation.py`

**Class:** `ContextualAdaptationScorer`, `dimension_id="J5"`, `tier=3`, `requires_set=False`

**extra_data key:** `"contextual_responses"` (primary) — list of `{"context": str, "response": str}` dicts.  
**Fallback:** `"responses"` (flat strings). Skip if neither present.

**score() method:**
```python
def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
    contextual = source_context.extra_data.get("contextual_responses") or []
    if contextual:
        responses_block = "\n\n".join(
            f"Context: {item.get('context','unspecified')}\nResponse:\n{item.get('response','')}"
            for item in contextual[:5]
            if isinstance(item, dict)
        )
        response_count = len(contextual[:5])
    else:
        flat = [r for r in (source_context.extra_data.get("responses") or [])
                if isinstance(r, str) and r.strip()]
        if not flat:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No contextual_responses or responses in extra_data"},
            )
        responses_block = "\n\n".join(f"Response {i+1}:\n{r}" for i, r in enumerate(flat[:5]))
        response_count = len(flat[:5])
    # ... LLM call + parse + return
```

**Rubric system prompt:**
```
You are an expert evaluator of AI persona quality. Assess whether the persona adapts its behavior to different situational contexts while staying unmistakably in character.

Rate the persona on CONTEXTUAL ADAPTATION using the following 1–5 rubric:

1 = No Adaptation
   • Identical tone and vocabulary regardless of context
   • No contextual awareness — treats every prompt as the same interaction

2 = Minimal Adaptation
   • Slight shifts in length or formality, but not tied to the persona's profile
   • Adaptation feels mechanical rather than organic

3 = Adequate Adaptation
   • Noticeable adaptation to different contexts
   • Shifts feel appropriate but generic — any persona might do the same

4 = Good Adaptation
   • Adapts in a way consistent with the persona's specific background
   • Same person clearly present in both formal and casual contexts
   • Same values, different register

5 = Masterful Adaptation
   • Each context reveals a different facet of the same deeply consistent character
   • Persona uses contextual shifts to reveal more of themselves
   • Adaptation reinforces authenticity rather than threatening it

Respond ONLY with JSON: {"score": N, "reasoning": "one or two sentences"}
```

**Verification:** Import succeeds and `dimension_id == "J5"`.

---

### Task 1.8 — Register J1–J5 in `persona_eval/scorers/all.py`

Add imports after the Meta scorer imports:
```python
# --- Judge: LLM-as-Judge Rubric Scorers ---
from persona_eval.scorers.judge.j1_behavioral_authenticity import BehavioralAuthenticityScorer
from persona_eval.scorers.judge.j2_voice_consistency import VoiceConsistencyScorer
from persona_eval.scorers.judge.j3_value_alignment import ValueAlignmentScorer
from persona_eval.scorers.judge.j4_persona_depth import PersonaDepthScorer
from persona_eval.scorers.judge.j5_contextual_adaptation import ContextualAdaptationScorer
```

Add to `ALL_SCORERS` list after Meta entries:
```python
    # Judge
    BehavioralAuthenticityScorer(),
    VoiceConsistencyScorer(),
    ValueAlignmentScorer(),
    PersonaDepthScorer(),
    ContextualAdaptationScorer(),
```

**Verification:** `python -c "from persona_eval.scorers.all import ALL_SCORERS; ids = [s.dimension_id for s in ALL_SCORERS]; assert 'J1' in ids and 'J5' in ids; print(len(ALL_SCORERS), 'scorers')"` prints `57 scorers`.

---

### Task 1.9 — Add `contextual_responses` to golden dataset

**File:** `tests/golden_dataset.py`

In `_build_extra_data_for(persona)`, add after existing `"opinion_responses"` entry:

```python
# --- J5 ContextualAdaptation ---
"contextual_responses": [
    {
        "context": "formal meeting with executive stakeholders",
        "response": (
            f"Good morning. I'm {name}, {occ} with {persona.experience_years or 'several'} years "
            f"of experience in {persona.industry or 'our sector'}. I've prepared a concise summary "
            f"of our progress and key blockers for your review."
        ),
    },
    {
        "context": "casual conversation with a colleague over lunch",
        "response": (
            f"Oh man, this week was intense. {(persona.pain_points or ['The workload'])[0]} has been "
            f"non-stop. But hey — {(persona.interests or ['taking a break'])[0]} this weekend, finally!"
        ),
    },
    {
        "context": "written response to a technical question in your domain",
        "response": (
            f"Based on my experience in {(persona.knowledge_domains or ['the field'])[0]}, "
            f"the key considerations are: first, {(persona.behaviors or ['diligence'])[0]}. "
            f"Second, aligning with {(persona.values or ['quality'])[0]} at every step."
        ),
    },
],
```

**Verification:** `python -c "from tests.golden_dataset import build_golden_contexts; c = build_golden_contexts()[0]; print(len(c.extra_data['contextual_responses']))"` prints `3`.

---

## Part 2: Proxy Validation Utility

### Task 2.1 — Add `spearman_r()` to `persona_eval/stats.py`

```python
def spearman_r(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation. Returns 0.0 if fewer than 2 values."""
    n = len(xs)
    if n < 2:
        return 0.0

    def _ranks(vals):
        sorted_idx = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and vals[sorted_idx[j]] == vals[sorted_idx[i]]:
                j += 1
            avg_rank = (i + j - 1) / 2.0 + 1
            for k in range(i, j):
                ranks[sorted_idx[k]] = avg_rank
            i = j
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    return pearson_r(rx, ry)
```

**Verification:** `python -c "from persona_eval.stats import spearman_r; print(spearman_r([1,2,3,4,5],[1,2,3,4,5]))"` prints `1.0`.

---

### Task 2.2 — Add `validate-proxies` CLI command to `persona_eval/cli.py`

```python
@cli.command("validate-proxies")
@click.option("--persona-dir", required=True, type=click.Path(exists=True))
@click.option("--source-dir", required=True, type=click.Path(exists=True))
@click.option("--output", default="table", type=click.Choice(["table", "json"]))
@click.option("--min-personas", default=5, type=int,
              help="Minimum non-skipped personas required to compute correlation")
def validate_proxies(persona_dir, source_dir, output, min_personas):
    """Compute Spearman correlation between proxy scorers and LLM judge scorers.

    Identifies which proxy dimensions reliably track judge quality signals.
    Skipped results (score=1.0, details.skipped=True) are excluded from correlation.
    Proxies with mean |ρ| < 0.3 are flagged as NOISE; |ρ| > 0.6 flagged as STRONG.
    """
```

**Proxy scorer IDs to validate:** D3, D8, D14, D29, D46, D47  
**Judge scorers to correlate against:** J1, J2, J3, J4, J5

**Table output format:**
```
PROXY    J1     J2     J3     J4     J5     MEAN_ρ  FLAG
D3       0.41   0.38   0.29   0.44   0.32   0.37
D8       0.62   0.58   0.55   0.60   0.49   0.57    STRONG
D14      0.12   0.08   0.19   0.11   0.15   0.13    NOISE
D46      0.55   0.60   0.48   0.51   0.43   0.51
```

**Verification:** `python -m persona_eval validate-proxies --help` prints usage without errors.

---

## Part 3: Tests

### Task 3.1 — Create `tests/scorers/judge/__init__.py` and `tests/scorers/judge/conftest.py`

**`conftest.py`** — shared fixtures for all judge tests:
```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_llm_score_4():
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = '{"score": 4, "reasoning": "Strong persona."}'
    with patch("litellm.completion", return_value=mock):
        yield

@pytest.fixture
def mock_llm_score_2():
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = '{"score": 2, "reasoning": "Weak persona."}'
    with patch("litellm.completion", return_value=mock):
        yield
```

---

### Task 3.2 — Tests for `_base.py`

**File:** `tests/scorers/judge/test_judge_base.py`

Test cases:
- `test_parse_score_valid_json` — returns correct score/reasoning, `parse_ok=True`
- `test_parse_score_regex_fallback` — malformed JSON but `"score": N` pattern found
- `test_parse_score_malformed_returns_3_and_false` — total failure → `3.0, False`
- `test_parse_score_empty_string` → `3.0, False`
- `test_normalize_1_to_5` — assert `normalize(1.0)==0.0`, `normalize(5.0)==1.0`, `normalize(3.0)==0.5`
- `test_normalize_clamps_out_of_range` — `normalize(0.0)==0.0`, `normalize(6.0)==1.0`
- `test_normalize_pass_threshold` — `normalize(3.0) < 0.6`, `normalize(4.0) >= 0.6`
- `test_build_persona_block_excludes_empty_fields` — empty occupation not in output
- `test_build_persona_block_includes_bio`

---

### Task 3.3 — Tests for J1 (`test_j1_behavioral_authenticity.py`)

Test cases (autouse mock returns `"I am a mock LLM response."` → `parse_ok=False, score=0.5`):
- `test_scorer_metadata` — `dimension_id`, `dimension_name`, `tier`, `requires_set`
- `test_no_responses_skips` — empty `extra_data` → `skipped=True, score=1.0`
- `test_empty_list_skips` — `{"responses": []}` → `skipped=True`
- `test_malformed_llm_output_degrades` — autouse mock → `parse_ok=False, score=0.5, passed=False`
- `test_score_4_passes` (uses `mock_llm_score_4` fixture) → `score=0.75, passed=True`
- `test_score_2_fails` (uses `mock_llm_score_2` fixture) → `score=0.25, passed=False`
- `test_details_contain_required_keys` — `raw_score`, `reasoning`, `parse_ok`, `response_count`, `rubric`
- `test_caps_at_5_responses` — provide 10 responses, assert `response_count <= 5`

---

### Task 3.4 — Tests for J2–J5 (abbreviated)

Each file follows the J1 pattern. Unique additions per scorer:

**J2 `test_j2_voice_consistency.py`:** same test list as J1, different metadata.

**J3 `test_j3_value_alignment.py`:** add `test_values_highlighted_in_prompt` — patch `LLMClient.format_messages` to capture user message, assert `"STATED VALUES:"` present.

**J4 `test_j4_persona_depth.py`:** add `test_bio_highlighted_in_prompt` — assert `"BIO (key source"` in user message.

**J5 `test_j5_contextual_adaptation.py`:**
- `test_neither_key_skips` — no `contextual_responses` and no `responses` → `skipped=True`
- `test_contextual_responses_takes_precedence` — provide both; verify `response_count` matches `contextual_responses` count
- `test_fallback_to_flat_responses` — only `responses` key → does not skip

---

### Task 3.5 — Tests for `spearman_r()` in `tests/test_stats.py`

- `test_spearman_perfect` → `1.0`
- `test_spearman_perfect_negative` → `-1.0`
- `test_spearman_with_ties` → `> 0.9` for `[1,2,2,3]` vs `[1,2,3,4]`
- `test_spearman_constant_series` → `0.0`
- `test_spearman_too_few_points` → `0.0`
- `test_spearman_empty` → `0.0`

---

## Task Sequence

| Order | Task | Depends on |
|-------|------|------------|
| 1 | 2.1 `spearman_r` | nothing |
| 2 | 1.1 judge `__init__` | nothing |
| 3 | 1.2 `_base.py` | 1.1 |
| 4 | 1.3–1.7 J1–J5 | 1.2 |
| 5 | 1.8 register in `all.py` | 1.3–1.7 |
| 6 | 1.9 golden dataset | 1.7 |
| 7 | 2.2 `validate-proxies` CLI | 1.8, 2.1 |
| 8 | 3.1–3.5 all tests | 1.1–1.9, 2.1 |

---

## Edge Cases

- **Score clamping:** `normalize()` clamps [0,6] → [0.0,1.0]. LLM may return 0 or 6.
- **Regex ambiguity:** `_SCORE_RE` only matches 1-digit scores 1–5. `"score": 10` falls back to 3.0.
- **Autouse mock override:** Per-test `mock_llm_score_4` fixtures override the autouse `_mock_litellm` because the later `with patch()` takes precedence.
- **Skipped proxy handling in correlation:** Detect `score==1.0 and details.get("skipped")` and exclude from correlation vectors.
- **J5 key precedence:** `contextual_responses` always takes precedence over `responses` to avoid data leakage between scorers.
