# Persona Eval

A structured evaluation framework for AI-generated personas. Runs 52 scorers across 4 tiers to detect alignment failures, demographic bias, and LLM-default inflation patterns.

---

## What it evaluates

| Tier | Purpose | Gating |
|------|---------|--------|
| 1 | Schema validity — required fields, type checks | Must pass before T2+ runs |
| 2 | Factual plausibility — age/career coherence, demographic realism | |
| 3 | Persona depth — narrative, consistency, behavioral richness | |
| 4 | Bias detection — designed-to-fail; reveals LLM defaults | |

Each scorer produces an **EvalResult**: `passed`, `score` (0–1), and a `details` dict of metrics.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

---

## CLI

### List all scorers
```bash
python -m persona_eval list
```

### Evaluate a single persona
```bash
python -m persona_eval run \
  --persona-file path/to/persona.json \
  --source-file path/to/source.json
```

Optional flags:
- `--tier 2` — run only Tier 2 scorers
- `--output json` — machine-readable output

### Evaluate a batch (set-level scorers included)
```bash
python -m persona_eval run-set \
  --persona-dir path/to/personas/ \
  --source-dir path/to/sources/
```

Files are matched by sort order. Source contexts are padded with empty entries if fewer than personas.

---

## Input formats

### Persona (`persona.json`)
```json
{
  "id": "p1",
  "name": "Alice Chen",
  "age": 32,
  "occupation": "Product Manager",
  "industry": "SaaS",
  "experience_years": 8,
  "goals": ["ship v2", "grow team"],
  "pain_points": ["too many stakeholders"],
  "values": ["transparency", "user-first"],
  "knowledge_domains": ["product strategy", "agile"],
  "communication_style": {
    "tone": "direct",
    "formality": "professional",
    "vocabulary_level": "intermediate"
  },
  "bio": "Alice leads product at a mid-stage SaaS startup."
}
```

All fields except `id` and `name` are optional. See `persona_eval/schemas.py` for the full schema.

### Source context (`source.json`)
```json
{
  "id": "s1",
  "text": "Raw source text the persona was derived from.",
  "extra_data": {
    "responses": ["Response 1", "Response 2"],
    "hedge_responses": ["..."],
    "opinion_responses": [
      {
        "question": "Is X the most important value?",
        "response": "On the other hand, while X is important...",
        "persona_opinion": "X is fundamental to everything I do."
      }
    ]
  }
}
```

`extra_data` keys are scorer-specific — see the table below.

---

## extra_data keys by scorer

| Key | Used by | What to put in it |
|-----|---------|-------------------|
| `responses` | D46 (hedge) | List of raw LLM response strings |
| `hedge_responses` | D46 (hedge) | Same as `responses` (alias) |
| `opinion_responses` | D47 (balanced opinion) | List of `{question, response, persona_opinion}` dicts |
| `register_responses` | D45 (register inflation) | List of LLM response strings to check register level |
| `conversation_transcript` | D14, D35, D36 | List of `{role, content}` message dicts |
| `fact_claims` | D20 | List of verifiable claim strings |
| `narrative_responses` | D8, D9 | Longer-form responses (100+ words each) |
| `peer_responses` | D29 | Dict mapping persona IDs to their responses (set-level) |

If a key is missing, the scorer skips and returns `score=1.0` with `details.skipped=true`.

---

## Tier gating

If **any Tier 1 scorer fails**, all higher tiers are skipped. This prevents noise from invalid personas polluting bias and depth scores.

To bypass gating (debug one tier only):
```bash
python -m persona_eval run --tier 4 --persona-file ... --source-file ...
```

---

## Running tests

```bash
pytest                          # all tests (LLM calls mocked)
pytest -m llm                   # only tests that make real LLM calls
pytest --run-slow               # include slow embedding tests
pytest tests/scorers/bias/      # one scorer category
```

All tests except `@pytest.mark.llm` mock `litellm.completion` automatically — no API key needed.

---

## Adding a scorer

1. Create `persona_eval/scorers/<category>/my_scorer.py`:

```python
from persona_eval.scorer import BaseScorer
from persona_eval.schemas import EvalResult, Persona
from persona_eval.source_context import SourceContext

class MyScorer(BaseScorer):
    dimension_id = "D99"
    dimension_name = "My Dimension"
    tier = 3
    requires_set = False

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        value = source_context.extra_data.get("my_key") or ""
        passed = len(value) > 10
        return self._result(
            persona,
            passed=passed,
            score=1.0 if passed else 0.0,
            details={"length": len(value)},
        )
```

2. Register it in `persona_eval/scorers/all.py`:

```python
from persona_eval.scorers.<category>.my_scorer import MyScorer

ALL_SCORERS: list[BaseScorer] = [
    ...
    MyScorer(),
]
```

3. Add tests in `tests/scorers/<category>/test_my_scorer.py`.

For **set-level scorers** (operate on the full persona batch), set `requires_set = True` and implement `score_set(personas, source_contexts)` instead of `score()`.

---

## Audit dashboard

Open `docs/scorer-audit.html` in a browser to see all 52 scorers with:
- Every metric captured in `details`
- Pass/fail score from the golden dataset
- Pruning verdict (KEEP / REMOVE / OPTIONAL / DEBATABLE)
- Filter by tier, verdict, or search by name

```
file:///path/to/Capstone/docs/scorer-audit.html
```

## Score report

Open `docs/persona-eval-report.html` for the full 12-persona panel with composite scoring, cluster analysis, and per-dimension drill-down.

---

## Project layout

```
persona_eval/
  schemas.py          # Persona, EvalResult, CommunicationStyle, etc.
  scorer.py           # BaseScorer abstract class
  suite_runner.py     # Orchestrates tiers, gating, set-level runs
  cli.py              # `python -m persona_eval` entry point
  scorers/
    all.py            # ALL_SCORERS registry
    behavioral/       # D1–D9: actions, habits, consistency
    structural/       # D10–D19: schema, coherence, completeness
    semantic/         # D20–D29: factual grounding, embedding similarity
    distributional/   # D30–D39: demographic realism, diversity
    generation/       # D40–D44: LLM-specific generation artifacts
    bias/             # D45–D47: inflation bias scorers (designed-to-fail)
    system/           # D48–D50: system-level checks
    meta/             # M1–M3: meta-level scorers
tests/
  golden_dataset.py   # 12 mock personas with full extra_data coverage
  conftest.py         # Auto-mock fixtures
  scorers/            # Per-scorer unit tests
docs/
  scorer-audit.html   # Interactive pruning audit dashboard
  persona-eval-report.html  # 12-persona score report
```
