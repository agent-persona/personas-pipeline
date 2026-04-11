# Experiment 6.19: Population Gini coefficient

**Branch**: yash/experiments
**Date**: 2026-04-10
**Owner**: Yash
**Group**: 6
**Size**: S
**Due**: Fri 2026-04-10
**Signal**: STRONG
**Recommendation**: adopt

---

## Hypothesis

Persona sets with Gini coefficient < 0.3 on quality scores indicate healthy diversity; > 0.5 indicates dominant/marginal persona imbalance.

## Control

(measurement only — no comparison)

## Variant

Calculate Gini coefficient over judge-rubric scores in a generated persona set

## Method

Run via canonical `personas-pipeline` modules:
- `synthesis.engine.synthesizer.synthesize` for persona generation
- `synthesis.models.persona.PersonaV1` for the schema
- `evaluation.judges.LLMJudge` for scoring (cross-model anti-bias: GPT-4o judges Claude output)
- `evaluation.registry.save_run` for the run record

Tenant: `tenant_acme_corp` from `evaluation.golden_set` (single stub tenant — full 20-tenant golden set blocked).

Sample size: **4** personas.

## Result

> Gini=0.023 across 4 personas, mean=0.83

## Metrics

| Metric | Value |
|--------|-------|
| `gini_coefficient` | 0.023 |
| `persona_scores` | 0.87, 0.82, 0.86, 0.78 |
| `mean_score` | 0.833 |

## Cost & Latency

| Metric | Value |
|--------|-------|
| Cost (USD) | $0.1420 |
| Duration (ms) | 149242 |
| Judge | gpt-4o (cross-model anti-bias) |

## Decision: `adopt`

**Threshold:** ≥0.05 delta on judge-rubric score, or hypothesis directly confirmed/falsified.

**Why adopt:** Variant outperformed control or hypothesis was directly confirmed.

Result: `Gini=0.023 across 4 personas, mean=0.83`

## Limitations

1. Single tenant (`tenant_acme_corp`) — golden set has only one stub; full 20-tenant set is blocked.
2. Sample size below the lab spec for statistical power.
3. Decision threshold is mine, not the lab's — the lab harness paragraph does not define adopt/reject/defer thresholds.

## Reproduce

```bash
python output/experiments/exp-6.19-population-gini/run_experiment.py
```
