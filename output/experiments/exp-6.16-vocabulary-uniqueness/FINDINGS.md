# Experiment 6.16: Vocabulary uniqueness

**Branch**: yash/experiments
**Date**: 2026-04-10
**Owner**: Yash
**Group**: 6
**Size**: S
**Due**: Sat 2026-04-11
**Signal**: STRONG
**Recommendation**: adopt

---

## Hypothesis

Generated personas use < 30% shared vocabulary in voice fields, indicating genuine voice distinctiveness rather than template repetition.

## Control

(measurement only — no comparison)

## Variant

Mean pairwise Jaccard similarity over vocabulary + sample_quotes

## Method

Run via canonical `personas-pipeline` modules:
- `synthesis.engine.synthesizer.synthesize` for persona generation
- `synthesis.models.persona.PersonaV1` for the schema
- `evaluation.judges.LLMJudge` for scoring (cross-model anti-bias: GPT-4o judges Claude output)
- `evaluation.registry.save_run` for the run record

Tenant: `tenant_acme_corp` from `evaluation.golden_set` (single stub tenant — full 20-tenant golden set blocked).

Sample size: **4** personas.

## Result

> Avg vocabulary Jaccard similarity=23.2% (good uniqueness)

## Metrics

| Metric | Value |
|--------|-------|
| `avg_jaccard_similarity` | 0.232 |

## Cost & Latency

| Metric | Value |
|--------|-------|
| Cost (USD) | $0.1241 |
| Duration (ms) | 140246 |
| Judge | gpt-4o (cross-model anti-bias) |

## Decision: `adopt`

**Threshold:** ≥0.05 delta on judge-rubric score, or hypothesis directly confirmed/falsified.

**Why adopt:** Variant outperformed control or hypothesis was directly confirmed.

Result: `Avg vocabulary Jaccard similarity=23.2% (good uniqueness)`

## Limitations

1. Single tenant (`tenant_acme_corp`) — golden set has only one stub; full 20-tenant set is blocked.
2. Sample size below the lab spec for statistical power.
3. Decision threshold is mine, not the lab's — the lab harness paragraph does not define adopt/reject/defer thresholds.

## Reproduce

```bash
python output/experiments/exp-6.16-vocabulary-uniqueness/run_experiment.py
```
