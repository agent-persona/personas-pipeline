# evaluation

**Problem space 5 — Evaluation & judge methodology.**

> *"We can't trust any of the findings in the other spaces unless we trust
>  the eval. LLM-as-judge has known failure modes (self-preference, position
>  bias, sycophancy), so the judge was red-teamed before being used as ground
>  truth."* — `PRD_LAB_RESEARCH.md`

This module is the trust anchor for every other problem space (1, 2, 3, 4, 6):
each of their findings was measured through the metrics and judges here.
The APIs are stable so a new iteration in any space can be compared against
the same historical controls.

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
`actionable`, `voice_fidelity` — the set that survived rubric-ablation
experiments (5.5).

### 3. `golden_set.py` — frozen tenants

Single stub tenant today (`tenant_acme_corp`, matching the mock crawler
fixture). Expands to 20 tenants per `PRD_LAB_RESEARCH.md` once researcher
#5 sources the sealed data.

Everything runs through `load_golden_set()` rather than constructing tenants
ad-hoc. That's the contract that keeps control runs and variant runs — across
time and across spaces — measured on the same inputs.

## Scientific backing (space 5)

The judge and metric choices here come out of a sequence of head-to-head
runs against the known failure modes of LLM-as-judge:

1. **Judge ↔ human correlation.** Spearman per rubric dimension against
   human labels, sized for a meaningful error bar.
2. **Cross-judge agreement.** Opus / Sonnet / GPT-class / Gemini-class
   judging the same outputs. Disagreement marked low-trust dimensions.
3. **Self-preference bias.** Sonnet judging Sonnet vs Opus outputs.
4. **Position & verbosity bias.** Pairwise A/B order swap; padded text
   control (findings under `output/experiments/exp-5.10-pairwise-vs-absolute/`).
5. **Rubric ablation.** Full rubric vs one-dimension-removed — the surviving
   dimensions are what ships as the default.
6. **Human Turing-style protocol.** Blind matching vs pairwise vs
   forced-choice; highest-agreement format was adopted.
7. **Time-to-detect.** How long a human needs to identify a known agent
   persona — the customer-facing "realism score."
8. **Adversarial detector.** Small detector model (AI vs human transcript)
   as inverted realism metric.
9. **Eval drift over model versions.** Re-run as new Claude models ship;
   judge gets version-pinned if it drifts.

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

Every prior finding in every other space was measured through
`LLMJudge.score_persona()` as it stands. Changing the default rubric, judge
model, or weighting silently invalidates those results, so:

1. **Version the change.** Add `LLMJudgeV2` (or a `rubric_version` flag)
   rather than mutating the defaults.
2. **Re-run the affected controls.** A rubric change means prior "variant
   beats control by X%" claims need to be recomputed.
3. **Publish the correlation number.** A new judge is only trustworthy
   alongside its Spearman vs human labels, recorded with the run.
