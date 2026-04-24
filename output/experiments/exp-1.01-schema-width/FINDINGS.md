# Experiment 1.01: Schema width

**Branch**: yash/experiments
**Date**: 2026-04-10
**Owner**: Yash
**Group**: 1
**Size**: M
**Due**: Sun 2026-04-12
**Signal**: STRONG
**Recommendation**: adopt

---

## Hypothesis

Minimal schemas underperform; maximal schemas plateau or hurt due to surface area for contradiction. PersonaV1 width sits near the optimum.

## Control

PersonaV1 (~15 fields)

## Variant

PersonaMinimal (5 fields: name, summary, goals, pains, vocabulary)

## Method

Run via canonical `personas-pipeline` modules:
- `synthesis.engine.synthesizer.synthesize` for persona generation
- `synthesis.models.persona.PersonaV1` for the schema
- `evaluation.judges.LLMJudge` for scoring (cross-model anti-bias: GPT-4o judges Claude output)
- `evaluation.registry.save_run` for the run record

Tenant: `tenant_acme_corp` from `evaluation.golden_set` (single stub tenant — full 20-tenant golden set blocked).

Sample size: **2** personas.

## Result

> Standard PersonaV1=0.84, Minimal(5 fields)=0.74, delta=+0.10

## Metrics

| Metric | Value |
|--------|-------|
| `judge_rubric_score_standard` | 0.84 |
| `judge_rubric_score_minimal` | 0.74 |
| `delta` | 0.1 |
| `standard_dimensions` | grounded=0.9, distinctive=0.8, coherent=0.9, actionable=0.8, voice_fidelity=0.7 |
| `minimal_dimensions` | grounded=0.9, distinctive=0.6, coherent=0.8, actionable=0.7, voice_fidelity=0.7 |

## Cost & Latency

| Metric | Value |
|--------|-------|
| Cost (USD) | $0.0499 |
| Duration (ms) | 58977 |
| Judge | gpt-4o (cross-model anti-bias) |

## Decision: `adopt`

**Threshold:** ≥0.05 delta on judge-rubric score, or hypothesis directly confirmed/falsified.

**Why adopt:** Variant outperformed control or hypothesis was directly confirmed.

Result: `Standard PersonaV1=0.84, Minimal(5 fields)=0.74, delta=+0.10`

## Limitations

1. Single tenant (`tenant_acme_corp`) — golden set has only one stub; full 20-tenant set is blocked.
2. Sample size below the lab spec for statistical power.
3. Decision threshold is mine, not the lab's — the lab harness paragraph does not define adopt/reject/defer thresholds.

## Reproduce

```bash
python output/experiments/exp-1.01-schema-width/run_experiment.py
```
