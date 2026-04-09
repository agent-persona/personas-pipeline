# evaluation

**Problem space 5 — Evaluation & judge methodology.**

> *"We can't trust any of the above experiments unless we trust the eval.
>  LLM-as-judge has known failure modes (self-preference, position bias,
>  sycophancy). We need to red-team the judge before using it as ground
>  truth."* — `PRD_LAB_RESEARCH.md`

This module is a **scaffold**. Every other problem space (1, 2, 3, 4, 6)
depends on stable metric and judge APIs to compare a variant run against a
control run. Researcher #5 owns filling out the bodies; the signatures are
frozen today so everyone else can import them right now without waiting.

## What lives here

```
evaluation/
├── evaluation/
│   ├── metrics.py       # Shared metrics (schema validity, groundedness, ...)
│   ├── judges.py        # LLMJudge + JudgeScore — LLM-as-judge scaffold
│   └── golden_set.py    # GoldenTenant + load_golden_set() — frozen tenants
└── pyproject.toml
```

The files import and re-export cleanly through `evaluation/__init__.py`:

```python
from evaluation import (
    schema_validity, groundedness_rate, distinctiveness, cost_per_persona,
    LLMJudge, JudgeScore,
    GoldenTenant, load_golden_set,
)
```

## The three pieces

### 1. `metrics.py` — deterministic, shared metrics

| Metric | Status | Used by |
|---|---|---|
| `schema_validity()` | Implemented | space 1, 2 (schema experiments) |
| `groundedness_rate()` | Implemented (expects `GroundednessReport` from synthesis) | space 3 |
| `distinctiveness()` | **TODO** — stub returns NaN | space 4, 6 |
| `cost_per_persona()` | Implemented | everyone |
| `turing_pass_rate()` | **TODO** | space 4, 5 |
| `drift()` | **TODO** | space 4 |
| `turns_to_break()` | **TODO** | space 4 |
| `judge_rubric_score()` | **TODO** — wraps `LLMJudge` | everyone |
| `human_correlation()` | **TODO** — the trust check | space 5 |

### 2. `judges.py` — LLM-as-judge

`LLMJudge` has three entry points and no real body yet:

- `score_persona(persona_dict) -> JudgeScore` — one persona, one rubric.
- `score_transcript(persona, transcript) -> JudgeScore` — twin chat eval.
- `pairwise(persona_a, persona_b) -> (winner, rationale)` — A/B preference.

Default rubric dimensions: `grounded`, `distinctive`, `coherent`,
`actionable`, `voice_fidelity`. Finalize in experiment 5.5 (rubric ablation).

### 3. `golden_set.py` — frozen tenants

Single stub tenant today (`tenant_acme_corp`, matching the mock crawler
fixture). Expands to 20 tenants per `PRD_LAB_RESEARCH.md` once researcher
#5 sources the sealed data.

Every experiment — in **every** problem space — should iterate
`load_golden_set()` rather than constructing tenants ad-hoc. That's the
contract that keeps control runs and variant runs on the same inputs.

## Experiments that own this module (space 5)

From `PRD_LAB_RESEARCH.md`:

1. **5.1 Judge ↔ human correlation.** 200 samples, LLM judge scores vs
   human labels, Spearman per rubric dimension.
2. **5.2 Cross-judge agreement.** Opus / Sonnet / GPT-class / Gemini-class
   judging the same outputs. Disagreement = low-trust dimension.
3. **5.3 Self-preference bias.** Sonnet judging Sonnet vs Opus outputs.
4. **5.4 Position & verbosity bias.** Pairwise A/B order swap; padded text.
5. **5.5 Rubric ablation.** Full rubric vs one-dimension-removed.
6. **5.6 Human Turing-style protocol.** Blind matching vs pairwise vs
   forced-choice. Pick the highest-agreement format.
7. **5.7 Time-to-detect.** How long a human needs to identify a known
   agent persona — the customer-facing "realism score."
8. **5.8 Adversarial detector.** Small detector model (AI vs human
   transcript) as inverted realism metric.
9. **5.9 Eval drift over model versions.** Re-run as new Claude models
   ship. Version-pin if it drifts.

## Dependencies on other modules

Evaluation *imports* from the other modules but is never imported *by* them
(one-way dependency — prevents a cycle):

```
synthesis.models.persona     → used by metrics.schema_validity
synthesis.engine.groundedness → duck-typed by metrics.groundedness_rate
```

No runtime dependency on `crawler/`, `segmentation/`, `twin/`, or
`orchestration/`. Judges score persona dicts and transcripts; the
surrounding pipeline is the producer, not a collaborator.

## Default-is-sacred rule

Everyone else is calling `LLMJudge.score_persona()` from their experiment
harness. When you change the default rubric, the default judge model, or
the default weighting:

1. **Version the change.** Add `LLMJudgeV2` (or a `rubric_version` flag)
   rather than mutating the defaults.
2. **Re-run every other space's control set.** A rubric change silently
   invalidates every prior "this variant beats control by X%" claim.
3. **Publish the correlation number.** Nobody should trust a new judge
   without its Spearman vs human labels in the results file.
