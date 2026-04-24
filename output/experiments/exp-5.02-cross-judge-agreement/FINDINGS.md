# Experiment 5.02: Cross-judge agreement

**Branch**: yash/experiments
**Date**: 2026-04-10
**Owner**: Yash
**Group**: 5
**Size**: M
**Due**: Sat 2026-04-11
**Signal**: WEAK
**Recommendation**: defer

---

## Hypothesis

Different LLM judges (GPT-4o vs Claude Sonnet) agree on persona quality rankings with Spearman >= 0.8, indicating judge choice doesn't dominate the signal.

## Control

GPT-4o judge scores

## Variant

Claude Sonnet judge scores on same personas

## Method

Run via canonical `personas-pipeline` modules:
- `synthesis.engine.synthesizer.synthesize` for persona generation
- `synthesis.models.persona.PersonaV1` for the schema
- `evaluation.judges.LLMJudge` for scoring (cross-model anti-bias: GPT-4o judges Claude output)
- `evaluation.registry.save_run` for the run record

Tenant: `tenant_acme_corp` from `evaluation.golden_set` (single stub tenant — full 20-tenant golden set blocked).

Sample size: **5** personas.

## Result

> Cross-judge Spearman=0.100

## Metrics

| Metric | Value |
|--------|-------|
| `gpt_scores` | 0.84, 0.84, 0.84, 0.82, 0.84 |
| `claude_scores` | 0.86, 0.86, 0.56, 0.86, 0.86 |
| `spearman` | 0.1 |

## Cost & Latency

| Metric | Value |
|--------|-------|
| Cost (USD) | $0.2064 |
| Duration (ms) | 210117 |
| Judge | gpt-4o (cross-model anti-bias) |

## Decision: `defer`

**Threshold:** ≥0.05 delta on judge-rubric score, or hypothesis directly confirmed/falsified.

**Why defer:** Inconclusive at this sample size — delta within ±0.05, synthesis ceiling effect, or pipeline failure.

Result: `Cross-judge Spearman=0.100`

## Limitations

1. Single tenant (`tenant_acme_corp`) — golden set has only one stub; full 20-tenant set is blocked.
2. Sample size below the lab spec for statistical power.
3. Decision threshold is mine, not the lab's — the lab harness paragraph does not define adopt/reject/defer thresholds.

## Reproduce

```bash
python output/experiments/exp-5.02-cross-judge-agreement/run_experiment.py
```
