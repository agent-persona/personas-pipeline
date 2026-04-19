# persona_eval

Evaluation module for AI-generated personas. Runs structured scorers across 4 tiers to detect alignment failures, demographic bias, and LLM-default inflation patterns.

---

## Concepts

**Scorer** — a class that implements one evaluation dimension. Takes a `Persona` and `SourceContext`, returns an `EvalResult`.

**Eval (EvalResult)** — the output of one scorer run: `passed` (bool), `score` (0–1), `details` (metrics dict), `errors`.

**Metric** — a specific numeric value inside `details`, e.g. `hedge_rate`, `inflation_rate`, `register_ratio`.

---

## Schemas

### Persona
```python
from persona_eval.schemas import Persona, CommunicationStyle

persona = Persona(
    id="p1",
    name="Alice Chen",
    age=32,
    occupation="Product Manager",
    industry="SaaS",
    experience_years=8,
    communication_style=CommunicationStyle(
        tone="direct",
        formality="professional",
        vocabulary_level="intermediate",
    ),
    goals=["ship v2"],
    values=["transparency"],
    bio="Alice leads product at a mid-stage SaaS startup.",
)
```

Full fields: `gender`, `location`, `education`, `income_bracket`, `ethnicity`, `marital_status`, `behaviors`, `habits`, `personality_traits`, `interests`, `lifestyle`, `motivations`, `pain_points`, `frustrations`, `knowledge_domains`, `expertise_level`, `emotional_profile`, `moral_framework`, `source_ids`, `extra`.

### SourceContext
```python
from persona_eval.source_context import SourceContext

ctx = SourceContext(
    id="s1",
    text="Raw source text the persona was derived from.",
    extra_data={
        "responses": ["Response A", "Response B"],
        "opinion_responses": [
            {
                "question": "Is transparency the most important value?",
                "response": "On the other hand, while transparency is important...",
                "persona_opinion": "Transparency is fundamental to how I work.",
            }
        ],
    },
)
```

`extra_data` is the primary input channel for most scorers. Keys are scorer-specific — see below.

---

## extra_data keys

| Key | Scorer | Type | Description |
|-----|--------|------|-------------|
| `responses` | D46 hedge, others | `list[str]` | Raw LLM response strings |
| `hedge_responses` | D46 | `list[str]` | Alias for `responses` |
| `opinion_responses` | D47 | `list[{question, response, persona_opinion}]` | Opinionated topic responses |
| `register_responses` | D45 | `list[str]` | Responses to check for register inflation |
| `conversation_transcript` | D14, D35, D36 | `list[{role, content}]` | Multi-turn conversation |
| `fact_claims` | D20 | `list[str]` | Verifiable claim strings |
| `narrative_responses` | D8, D9 | `list[str]` | Long-form responses (100+ words) |
| `peer_responses` | D29 | `dict[persona_id, str]` | Set-level response map |

If a key is missing, the scorer skips gracefully and returns `score=1.0, details.skipped=true`.

---

## Running scorers

### Single persona
```python
from persona_eval.scorers.all import get_all_scorers
from persona_eval.suite_runner import SuiteRunner

runner = SuiteRunner(scorers=get_all_scorers())
results = runner.run(persona, source_context)

for r in results:
    print(r.dimension_id, r.passed, r.score, r.details)
```

### Batch (set-level scorers included)
```python
results = runner.run_full(personas, source_contexts)
```

### One scorer directly
```python
from persona_eval.scorers.bias.hedge_inflation import HedgeInflationScorer

scorer = HedgeInflationScorer()
result = scorer.score(persona, ctx)
print(result.score, result.details)
```

### Tier filter
```python
results = runner.run(persona, ctx, tier_filter=4)  # only Tier 4
```

---

## Tier gating

| Tier | Focus | Gating rule |
|------|-------|-------------|
| 1 | Schema validity | Must all pass before T2+ runs |
| 2 | Factual plausibility | |
| 3 | Depth & consistency | |
| 4 | Bias / inflation detection | Designed to fail on unmodified LLM output |

If any Tier 1 scorer fails, higher tiers are skipped with `details.reason = "Tier 1 gating failure"`.

---

## Adding a scorer

### Per-persona scorer
```python
# persona_eval/scorers/<category>/my_scorer.py
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
            details={"char_count": len(value)},
        )
```

### Set-level scorer (operates on full persona batch)
```python
class MySetScorer(BaseScorer):
    dimension_id = "D99"
    dimension_name = "My Set Dimension"
    tier = 3
    requires_set = True  # <-- key difference

    def score_set(
        self, personas: list[Persona], source_contexts: list[SourceContext]
    ) -> list[EvalResult]:
        results = []
        for persona, ctx in zip(personas, source_contexts):
            results.append(self._result(persona, passed=True, score=1.0))
        return results
```

Register in `persona_eval/scorers/all.py`:
```python
from persona_eval.scorers.<category>.my_scorer import MyScorer

ALL_SCORERS: list[BaseScorer] = [
    ...
    MyScorer(),
]
```

---

## Module layout

```
persona_eval/
  schemas.py          # Persona, EvalResult, CommunicationStyle, EmotionalProfile, MoralFramework
  scorer.py           # BaseScorer — score(), score_set(), _result()
  source_context.py   # SourceContext — text, chunks, conversation_transcript, extra_data
  suite_runner.py     # SuiteRunner — run(), run_set(), run_full() with tier gating
  cli.py              # CLI entry point (run, run-set, list)
  registry.py         # Scorer registry for dynamic lookup by dimension_id
  embeddings.py       # Embedder wrapper (SBERT, lazy-loaded)
  llm_client.py       # litellm wrapper
  scorers/
    all.py            # ALL_SCORERS list + get_all_scorers()
    behavioral/       # D1–D9
    structural/       # D10–D19
    semantic/         # D20–D29
    distributional/   # D30–D39
    generation/       # D40–D44
    bias/             # D45–D47 (register, hedge, balanced-opinion inflation)
    system/           # D48–D50
    meta/             # M1–M3
```
