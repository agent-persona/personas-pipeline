# Experiment 1.09: Hierarchical vs flat schema

**Branch**: yash/experiments
**Date**: 2026-04-10
**Owner**: Yash
**Group**: 1
**Size**: S
**Due**: Fri 2026-04-10
**Signal**: WEAK
**Recommendation**: defer

---

## Hypothesis

Hierarchical schemas with semantic grouping (Demographics, Firmographics) produce higher coherence than flat schemas because the LLM can structure related fields together.

## Control

PersonaV1 hierarchical

## Variant

Flat schema (all fields top-level)

## Method

Run via canonical `personas-pipeline` modules:
- `synthesis.engine.synthesizer.synthesize` for persona generation
- `synthesis.models.persona.PersonaV1` for the schema
- `evaluation.judges.LLMJudge` for scoring (cross-model anti-bias: GPT-4o judges Claude output)
- `evaluation.registry.save_run` for the run record

Tenant: `tenant_acme_corp` from `evaluation.golden_set` (single stub tenant — full 20-tenant golden set blocked).

Sample size: **2** personas.

## Result

> Hierarchical=0.86 vs Flat=0.88, delta=-0.02

## Metrics

| Metric | Value |
|--------|-------|
| `judge_rubric_score_hierarchical` | 0.86 |
| `judge_rubric_score_flat` | 0.88 |
| `delta` | -0.02 |
| `hierarchical_dims` | grounded=0.9, distinctive=0.8, coherent=0.9, actionable=0.8, voice_fidelity=0.8 |
| `flat_dims` | grounded=0.9, distinctive=0.8, coherent=0.9, actionable=0.8, voice_fidelity=0.9 |

## Cost & Latency

| Metric | Value |
|--------|-------|
| Cost (USD) | $0.0348 |
| Duration (ms) | 44321 |
| Judge | gpt-4o (cross-model anti-bias) |

## Decision: `defer`

**Threshold:** ≥0.05 delta on judge-rubric score, or hypothesis directly confirmed/falsified.

**Why defer:** Inconclusive at this sample size — delta within ±0.05, synthesis ceiling effect, or pipeline failure.

Result: `Hierarchical=0.86 vs Flat=0.88, delta=-0.02`

## Limitations

1. Single tenant (`tenant_acme_corp`) — golden set has only one stub; full 20-tenant set is blocked.
2. Sample size below the lab spec for statistical power.
3. Decision threshold is mine, not the lab's — the lab harness paragraph does not define adopt/reject/defer thresholds.

## Reproduce

```bash
python output/experiments/exp-1.09-hierarchical-vs-flat/run_experiment.py
```
