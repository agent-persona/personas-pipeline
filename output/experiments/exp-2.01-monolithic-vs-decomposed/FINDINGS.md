# Experiment 2.01: Monolithic vs decomposed

**Branch**: yash/experiments
**Date**: 2026-04-10
**Owner**: Yash
**Group**: 2
**Size**: M
**Due**: Mon 2026-04-13
**Signal**: STRONG (negative)
**Recommendation**: reject

---

## Hypothesis

Decomposed synthesis (3 sub-calls: demographics, goals/pains, voice) beats monolithic because each call has tighter scope and clearer success criteria.

## Control

Single monolithic synthesize() call

## Variant

3-step decomposed synthesis via direct LLM calls

## Method

Run via canonical `personas-pipeline` modules:
- `synthesis.engine.synthesizer.synthesize` for persona generation
- `synthesis.models.persona.PersonaV1` for the schema
- `evaluation.judges.LLMJudge` for scoring (cross-model anti-bias: GPT-4o judges Claude output)
- `evaluation.registry.save_run` for the run record

Tenant: `tenant_acme_corp` from `evaluation.golden_set` (single stub tenant — full 20-tenant golden set blocked).

Sample size: **2** personas.

## Result

> Monolithic=0.84 vs Decomposed=0.00, delta=-0.84

## Metrics

| Metric | Value |
|--------|-------|
| `judge_rubric_score_monolithic` | 0.84 |
| `judge_rubric_score_decomposed` | 0.0 |
| `delta` | -0.84 |

## Cost & Latency

| Metric | Value |
|--------|-------|
| Cost (USD) | $0.0459 |
| Duration (ms) | 47695 |
| Judge | gpt-4o (cross-model anti-bias) |

## Decision: `reject`

**Threshold:** ≥0.05 delta on judge-rubric score, or hypothesis directly confirmed/falsified.

**Why reject:** Variant underperformed control or hypothesis was falsified.

Result: `Monolithic=0.84 vs Decomposed=0.00, delta=-0.84`

## Limitations

1. Single tenant (`tenant_acme_corp`) — golden set has only one stub; full 20-tenant set is blocked.
2. Sample size below the lab spec for statistical power.
3. Decision threshold is mine, not the lab's — the lab harness paragraph does not define adopt/reject/defer thresholds.

## Reproduce

```bash
python output/experiments/exp-2.01-monolithic-vs-decomposed/run_experiment.py
```
