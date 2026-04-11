# Experiment 6.01: Distinctiveness floor

**Branch**: yash/experiments
**Date**: 2026-04-10
**Owner**: Yash
**Group**: 6
**Size**: M
**Due**: Mon 2026-04-13
**Signal**: WEAK
**Recommendation**: defer

---

## Hypothesis

Setting a minimum distinctiveness threshold (reject and regenerate if below 0.6) improves overall persona set quality.

## Control

Accept all generated personas

## Variant

Reject personas with distinctiveness dimension < 0.6

## Method

Run via canonical `personas-pipeline` modules:
- `synthesis.engine.synthesizer.synthesize` for persona generation
- `synthesis.models.persona.PersonaV1` for the schema
- `evaluation.judges.LLMJudge` for scoring (cross-model anti-bias: GPT-4o judges Claude output)
- `evaluation.registry.save_run` for the run record

Tenant: `tenant_acme_corp` from `evaluation.golden_set` (single stub tenant — full 20-tenant golden set blocked).

Sample size: **6** personas.

## Result

> Generated 6, rejected 0. Control mean=0.83, Filtered mean=0.83

## Metrics

| Metric | Value |
|--------|-------|
| `total_generated` | 6 |
| `rejected` | 0 |
| `control_mean` | 0.835 |
| `variant_mean` | 0.835 |

## Cost & Latency

| Metric | Value |
|--------|-------|
| Cost (USD) | $0.2330 |
| Duration (ms) | 247472 |
| Judge | gpt-4o (cross-model anti-bias) |

## Decision: `defer`

**Threshold:** ≥0.05 delta on judge-rubric score, or hypothesis directly confirmed/falsified.

**Why defer:** Inconclusive at this sample size — delta within ±0.05, synthesis ceiling effect, or pipeline failure.

Result: `Generated 6, rejected 0. Control mean=0.83, Filtered mean=0.83`

## Limitations

1. Single tenant (`tenant_acme_corp`) — golden set has only one stub; full 20-tenant set is blocked.
2. Sample size below the lab spec for statistical power.
3. Decision threshold is mine, not the lab's — the lab harness paragraph does not define adopt/reject/defer thresholds.

## Reproduce

```bash
python output/experiments/exp-6.01-distinctiveness-floor/run_experiment.py
```
