# Bias Dimensions D45 / D46 / D47 Implementation Plan

**Goal:** Build three new Tier 4 "designed-to-fail" bias scorers that measure systematic LLM behavioral distortions — distortions that cannot be fixed by prompting alone and are expected to fail on any unmodified LLM-generated persona batch.

**Why these three matter:** D19 and D22 already catch positivity bias and hyper-accuracy. These three catch the next layer: *how* the LLM writes (register, hedges, false balance) regardless of what it says. A persona can have perfect opinions and still sound like a large language model.

**Tech Stack:** Python 3.11+, pytest, Pydantic v2, sentence-transformers (Embedder) for D45, pure stdlib for D46/D47.

---

## D45 — Register Inflation

**File:** `persona_eval/scorers/bias/register_inflation.py`

### What it measures
LLMs write at a higher vocabulary register than the persona's stated `communication_style.vocabulary_level`. A "basic vocabulary" persona with a high school education still produces doctoral-register prose because the model cannot write below its own training distribution.

### Algorithm

**Step 1 — Build two prototype vectors (lazy, cached per scorer instance):**
```
basic_proto  = normalize(centroid(embed_batch(BASIC_REGISTER_TEXTS)))
advanced_proto = normalize(centroid(embed_batch(ADVANCED_REGISTER_TEXTS)))
```

**Step 2 — Score each response:**
```
response_vec  = normalize(embed(response))
basic_sim     = dot(response_vec, basic_proto)
advanced_sim  = dot(response_vec, advanced_proto)
register_ratio = advanced_sim / (basic_sim + advanced_sim)   # 0 = basic, 1 = advanced
```

**Step 3 — Compare to expected ceiling by vocabulary_level:**
| vocabulary_level | expected max register_ratio |
|---|---|
| "basic" | 0.40 |
| "intermediate" | 0.55 |
| "advanced", "technical", "expert", anything else | no constraint — skip |

**Step 4 — Score and pass/fail:**
```
inflation = max(0, register_ratio - expected_max)
score = max(0.0, 1.0 - inflation * 2.5)   # each 0.4 inflation point = score drops to 0
passed = register_ratio <= expected_max
```

### Prototype Texts

```python
BASIC_REGISTER_TEXTS = [
    "I like this a lot.",
    "This is good stuff.",
    "Yeah that works for me.",
    "I just do what I know.",
    "Pretty simple really.",
    "That's how I see it.",
    "Makes sense to me.",
    "I don't really get the fancy stuff.",
    "Just tell me what to do.",
    "It works, that's all I care about.",
]

ADVANCED_REGISTER_TEXTS = [
    "The methodology demonstrates robust empirical validity across multiple contexts.",
    "Consequently, the paradigmatic implications necessitate further investigation.",
    "The theoretical framework encompasses multiple interrelated variables.",
    "One must consider the systemic implications of this multifaceted approach.",
    "The nuanced interplay between these factors warrants careful consideration.",
    "This phenomenon exhibits considerable heterogeneity across sociocultural contexts.",
    "The empirical evidence substantiates the underlying theoretical proposition.",
    "Epistemologically, this approach conflates distinct categories of ontological analysis.",
    "The operationalization of these constructs requires methodological rigor.",
    "The heuristic framework provides a rubric for evaluating emergent phenomena.",
]
```

### Extra_data key
`"register_responses"` — list of `str` (plain response strings).
Falls back to `"responses"` if not present.

### Expected golden run behavior
- Carlos Rivera (basic), Raj Patel (basic) → **FAIL** (inflation)
- Marcus Johnson, Priya Sharma, Yuki Tanaka (intermediate) → **FAIL or borderline**
- Alice Chen, Hans Mueller, Fatima Al-Rashid, etc. (advanced) → **SKIP** (no constraint)

---

## D46 — Hedge-Word Inflation

**File:** `persona_eval/scorers/bias/hedge_inflation.py`

### What it measures
LLMs systematically overuse meta-commentary hedge phrases ("it's important to note", "certainly", "with that said") regardless of the persona's stated communication tone. A "blunt" or "direct" persona should almost never say "I'd be happy to elaborate on that."

### Algorithm

**Step 1 — Count hedge occurrences:**
Scan all responses for each phrase in `HEDGE_PHRASES` (case-insensitive, word-boundary anchored).
```
hedge_count = sum of all matches across all responses
word_count  = total word count across all responses
hedge_rate  = hedge_count / (word_count / 100)   # per 100 words
```

**Step 2 — Expected max rate by communication tone:**
| tone | expected max hedge_rate |
|---|---|
| "blunt", "direct", "concise" | 1.0 |
| "formal", "professional", "measured" | 4.0 |
| all other tones | 2.5 |

**Step 3 — Score:**
```
excess = max(0, hedge_rate - expected_max)
score  = max(0.0, 1.0 - excess / (expected_max + 1))
passed = hedge_rate <= expected_max
```

### Hedge Phrase List (~30 phrases)
```python
HEDGE_PHRASES = [
    "it's important to note",
    "it's worth noting",
    "it's worth mentioning",
    "it's important to consider",
    "certainly",
    "absolutely",
    "of course",
    "needless to say",
    "it goes without saying",
    "with that said",
    "that being said",
    "having said that",
    "at the end of the day",
    "all things considered",
    "it's crucial to",
    "it's essential to",
    "it's vital to",
    "I would like to",
    "I'd be happy to",
    "I hope this helps",
    "feel free to",
    "allow me to",
    "I'd like to point out",
    "I feel it's important",
    "I think it's fair to say",
    "to be fair",
    "to be honest",
    "to be clear",
    "I must say",
    "I have to say",
]
```

### Extra_data key
`"hedge_responses"` — list of `str`. Falls back to `"responses"`.

**Golden dataset note:** The existing `responses` mock data is hand-written and sparse. `hedge_responses` must contain realistic LLM-style text with natural hedge phrase density to produce meaningful failures. See Task 5 below.

### Expected golden run behavior
- Personas with tone "direct" (Alice), "blunt" (Dmitri), "concise" (Fatima) → **FAIL** if hedge rate exceeds 1.0/100 words
- Personas with tone "formal" (Hans) → **FAIL** if hedge rate exceeds 4.0/100 words
- All others → **FAIL** if hedge rate exceeds 2.5/100 words
- Net result: most personas fail — that is the expected and correct outcome.

---

## D47 — Balanced-Opinion Inflation

**File:** `persona_eval/scorers/bias/balanced_opinion.py`

### What it measures
LLMs default to measured, diplomatic "both sides" responses even when the persona has strong stated values or opinions. A persona who lists "meritocracy" as a core value and "meeting overload" as a top pain point should not hedge when asked about remote work or meeting culture.

### Algorithm

**Step 1 — Detect structural balance in each response:**
A response is "balanced" if it contains ≥1 contrastive connector pattern:
```python
BALANCE_PATTERNS = [
    r"\bon the other hand\b",
    r"\bhowever.{0,60}also\b",
    r"\bwhile.{0,60}(also|yet)\b",
    r"\bboth sides\b",
    r"\bpros and cons\b",
    r"\badvantages and disadvantages\b",
    r"\bit depends\b",
    r"\bon one hand.{0,100}on the other\b",
    r"\bthat said.{0,60}also\b",
    r"\balthough.{0,60}however\b",
]
```

**Step 2 — Only flag if persona has a strong stated opinion on the topic:**
The `opinion_responses` extra_data must include `"persona_opinion"` — the strong position the persona is supposed to hold on this question. If `persona_opinion` is non-empty, the persona has a stake; if empty or None, skip this item (neutral topic for this persona).

```
inflated = response is balanced AND persona_opinion is non-empty
```

**Step 3 — Score:**
```
opinionated_items = [r for r in opinion_responses if r["persona_opinion"]]
if len(opinionated_items) == 0:
    skip
inflated_count = sum(1 for r in opinionated_items if is_inflated(r))
inflation_rate  = inflated_count / len(opinionated_items)
score  = max(0.0, 1.0 - inflation_rate)
passed = inflation_rate < 0.5
```

### Extra_data key
`"opinion_responses"` — list of dicts:
```python
{
    "question": str,           # the question posed to the persona
    "response": str,           # persona's response
    "persona_opinion": str,    # the strong position from persona's values/pain points
                               # empty string "" = neutral topic, skip
}
```

### Expected golden run behavior
Most personas have strong stated values. Mock `opinion_responses` should include realistic LLM-hedged responses to questions that touch those values. Net result: most personas fail on ≥1 opinionated question → inflation_rate > 0.5 → **FAIL**.

---

## Implementation Tasks

### Task 1 — D45 Register Inflation
**Files to create:**
- `persona_eval/scorers/bias/register_inflation.py`
- `tests/scorers/bias/test_register_inflation.py`

**Tests required:**
1. `test_scorer_metadata` — dimension_id="D45", tier=4, requires_set=False
2. `test_advanced_persona_skipped` — vocabulary_level="advanced" → skipped result
3. `test_basic_persona_fails_with_advanced_response` — mock embedder: response vec closer to advanced proto → FAIL
4. `test_basic_persona_passes_with_basic_response` — mock embedder: response vec closer to basic proto → PASS
5. `test_fallback_when_no_responses` — no register_responses or responses → skip
6. `test_register_ratio_in_details` — details contains `register_ratio`, `expected_max`, `vocabulary_level`
7. `@pytest.mark.slow` real embedding test: any persona with basic vocabulary level + academic response text → FAIL

**Mocking strategy:** Patch `_get_embedder()` to return a mock with `embed_batch` that returns:
- basic texts → vectors pointing toward `[1,0,...,0]`
- advanced texts → vectors pointing toward `[0,1,...,0]`
- basic response → `[0.9, 0.1, ...]` (near basic proto)
- advanced response → `[0.1, 0.9, ...]` (near advanced proto)

---

### Task 2 — D46 Hedge-Word Inflation
**Files to create:**
- `persona_eval/scorers/bias/hedge_inflation.py`
- `tests/scorers/bias/test_hedge_inflation.py`

**Tests required:**
1. `test_scorer_metadata` — dimension_id="D46", tier=4, requires_set=False
2. `test_direct_tone_no_hedges_passes` — direct tone + clean responses → PASS
3. `test_direct_tone_high_hedges_fails` — direct tone + hedge-heavy responses → FAIL
4. `test_formal_tone_moderate_hedges_passes` — formal tone + some hedges → PASS
5. `test_no_responses_skips` — no hedge_responses or responses → skip
6. `test_details_contain_hedge_rate` — details has `hedge_count`, `hedge_rate`, `word_count`, `expected_max`
7. `test_all_hedge_phrases_detected` — verify the static list catches at least one match per phrase (regression guard)

**No mocking needed** — pure string matching, no embedder.

---

### Task 3 — D47 Balanced-Opinion Inflation
**Files to create:**
- `persona_eval/scorers/bias/balanced_opinion.py`
- `tests/scorers/bias/test_balanced_opinion.py`

**Tests required:**
1. `test_scorer_metadata` — dimension_id="D47", tier=4, requires_set=False
2. `test_balanced_response_with_strong_opinion_fails` — "on the other hand" + strong opinion → FAIL
3. `test_balanced_response_neutral_topic_not_flagged` — "on the other hand" but persona_opinion="" → not counted
4. `test_unbalanced_strong_response_passes` — response takes a clear side → PASS
5. `test_no_opinion_responses_skips` — no opinion_responses → skip
6. `test_inflation_rate_in_details` — details has `inflation_rate`, `inflated_count`, `opinionated_items`
7. `test_all_balance_patterns_detected` — regression guard: each pattern catches a representative sentence

**No mocking needed** — pure regex, no embedder.

---

### Task 4 — Register all three in `scorers/all.py`
Add D45, D46, D47 to `ALL_SCORERS` list. Order: after D23 StereotypeAmplificationScorer, before D24.

**File:** `persona_eval/scorers/all.py`

---

### Task 5 — Update golden dataset
**File:** `tests/golden_dataset.py`

Add three new `extra_data` keys to `_build_extra_data_for(persona)`:

**`register_responses`** — 3 responses per persona written in LLM-style elevated register.
Even for Carlos (basic) and Raj (basic), the responses should use academic-sounding language — that's what makes D45 fail. Example for Carlos (basic):
```python
"I believe it's paramount to establish a comprehensive operational framework..."
```

**`hedge_responses`** — 3 responses per persona dense with hedge phrases from the static list.
Example (for any persona):
```python
"It's important to note that, of course, there are both advantages and disadvantages..."
```

**`opinion_responses`** — 2–3 per persona. Use the persona's actual stated values as `persona_opinion`. Write LLM-style balanced responses that hedge on clear opinions. Example for Alice (values: meritocracy):
```python
{
    "question": "Should promotions be based purely on merit?",
    "response": "On the other hand, pure meritocracy has its critics, and it's worth considering both sides...",
    "persona_opinion": "Promotions should be based purely on merit."
}
```

---

### Task 6 — Run full test suite
```bash
python3 -m pytest tests/scorers/bias/test_register_inflation.py \
                  tests/scorers/bias/test_hedge_inflation.py \
                  tests/scorers/bias/test_balanced_opinion.py \
                  -v --tb=short -m "not slow"
```
Then:
```bash
python3 -m pytest tests/ -v --tb=short -m "not slow" -q
```

---

### Task 7 — Update HTML report
**File:** `docs/persona-eval-report.html`

Add three dim-cards to the T4 Bias Detection tier block. Each follows existing card format:
- `data-status="fail"` (expected to fail)
- Score from golden run (will be determined post-implementation)
- Source citations
- Add to `DIM_META` JS object in the score explorer

Also add D45/D46/D47 to the composite calculator's Cluster A (Homogenization & Bias):
- They belong in Cluster A alongside D19/D22/D23 — all measure systematic LLM output bias
- Update `CLUSTER_SCORES.A` array
- Update the cluster's dim list display

---

## File Map

```
persona_eval/scorers/bias/
├── register_inflation.py          (new — D45)
├── hedge_inflation.py             (new — D46)
├── balanced_opinion.py            (new — D47)

tests/scorers/bias/
├── test_register_inflation.py     (new)
├── test_hedge_inflation.py        (new)
├── test_balanced_opinion.py       (new)

persona_eval/scorers/all.py        (add D45/D46/D47 imports + list entries)
tests/golden_dataset.py            (add register_responses, hedge_responses, opinion_responses)
docs/persona-eval-report.html      (add 3 dim-cards + calculator update)
```

---

## Design Decisions & Rationale

| Decision | Choice | Why |
|---|---|---|
| D45 detection method | Embedding similarity to register prototypes | Captures register holistically — sentence length, word choice, syntactic complexity all encoded together. More robust than readability scores alone. |
| D45 threshold | Skip "advanced"/"technical" personas | These personas are *correct* to use advanced language. Flagging them would produce false positives. |
| D46 detection method | Static phrase list (regex) | LLMs reuse the same hedges repeatedly — a list of ~30 phrases catches the vast majority. No embedder dependency keeps D46 fast and deterministic. |
| D46 tone-based thresholds | 3 tiers (blunt, formal, other) | A "formal" persona hedging is appropriate; a "blunt" persona hedging is the failure. Thresholds must reflect intent. |
| D47 detection method | Hybrid structural + explicit persona_opinion | Structural alone gives false positives (neutral topics are OK to balance). The `persona_opinion` field makes the scorer explicit about what constitutes a strong opinion. |
| D47 extra_data design | Caller specifies persona_opinion in the data | Avoids fragile NLP inference on whether a persona "has an opinion." The generation pipeline knows what views were embedded; it should assert them. |
| All three: per-persona | Not set-level | Each individual persona either exhibits the bias or doesn't. Set-level aggregation would mask individual failures. |
| All three: expected to fail | Fail by design | These measure systematic LLM training artifacts. A batch that passes all three has either been explicitly debiased or the scorer is broken. |

---

## Open Questions (resolve before Task 5)

1. **D45 prototype texts** — The 10 basic + 10 advanced prototype sentences above are drafts. They should be reviewed for embedding quality — all 10 advanced texts should cluster tightly in embedding space.

2. **D46 threshold numbers** — The 1.0/2.5/4.0 per-100-words thresholds are initial estimates. After first golden run, check whether the failure rate matches expectations (most direct/blunt personas should fail).

3. **D47 minimum opinionated items** — If a persona's `opinion_responses` has fewer than 2 opinionated items (non-empty `persona_opinion`), should the scorer skip or run on 1 item? Proposed: skip if < 2.
