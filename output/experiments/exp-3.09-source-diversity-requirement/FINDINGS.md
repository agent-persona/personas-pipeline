# Experiment 3.09: Source diversity requirement

**Branch**: yash/experiments
**Date**: 2026-04-10
**Owner**: Yash
**Group**: 3
**Size**: S
**Due**: Tue 2026-04-14
**Signal**: WEAK
**Recommendation**: defer

---

## Hypothesis

Requiring evidence from >= 2 different source types (ga4 + hubspot) produces more grounded personas because cross-source corroboration reduces single-source bias.

## Control

No source diversity requirement (current)

## Variant

Enforce minimum 2 source types in source_evidence

## Method

Run via canonical `personas-pipeline` modules:
- `synthesis.engine.synthesizer.synthesize` for persona generation
- `synthesis.models.persona.PersonaV1` for the schema
- `evaluation.judges.LLMJudge` for scoring (cross-model anti-bias: GPT-4o judges Claude output)
- `evaluation.registry.save_run` for the run record

Tenant: `tenant_acme_corp` from `evaluation.golden_set` (single stub tenant — full 20-tenant golden set blocked).

Sample size: **1** personas.

## Result

> Sources: [] (0 types). Groundedness=0.90

## Metrics

| Metric | Value |
|--------|-------|
| `source_types_used` |  |
| `diversity_count` | 0 |
| `judge_rubric_score` | 0.88 |
| `groundedness` | 0.9 |

## Cost & Latency

| Metric | Value |
|--------|-------|
| Cost (USD) | $0.0261 |
| Duration (ms) | 30387 |
| Judge | gpt-4o (cross-model anti-bias) |

## Decision: `defer`

**Threshold:** ≥0.05 delta on judge-rubric score, or hypothesis directly confirmed/falsified.

**Why defer:** Inconclusive at this sample size — delta within ±0.05, synthesis ceiling effect, or pipeline failure.

Result: `Sources: [] (0 types). Groundedness=0.90`

## Limitations

1. Single tenant (`tenant_acme_corp`) — golden set has only one stub; full 20-tenant set is blocked.
2. Sample size below the lab spec for statistical power.
3. Decision threshold is mine, not the lab's — the lab harness paragraph does not define adopt/reject/defer thresholds.

## Reproduce

```bash
python output/experiments/exp-3.09-source-diversity-requirement/run_experiment.py
```
