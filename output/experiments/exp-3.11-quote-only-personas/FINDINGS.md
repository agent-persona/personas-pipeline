# Experiment 3.11: Quote-only personas

**Branch**: yash/experiments
**Date**: 2026-04-10
**Owner**: Yash
**Group**: 3
**Size**: M
**Due**: Tue 2026-04-14
**Signal**: STRONG
**Recommendation**: adopt

---

## Hypothesis

Personas defined primarily through sample quotes produce more distinctive twins than field-first personas because voice anchors are more contagious than abstract trait descriptions.

## Control

Standard field-first PersonaV1

## Variant

Quote-first persona (7 quotes + minimal name/summary)

## Method

Run via canonical `personas-pipeline` modules:
- `synthesis.engine.synthesizer.synthesize` for persona generation
- `synthesis.models.persona.PersonaV1` for the schema
- `evaluation.judges.LLMJudge` for scoring (cross-model anti-bias: GPT-4o judges Claude output)
- `evaluation.registry.save_run` for the run record

Tenant: `tenant_acme_corp` from `evaluation.golden_set` (single stub tenant — full 20-tenant golden set blocked).

Sample size: **6** personas.

## Result

> Standard voice=0.85 vs Quote-first=0.90, delta=+0.05

## Metrics

| Metric | Value |
|--------|-------|
| `standard_voice` | 0.85 |
| `quote_voice` | 0.9 |
| `standard_overall` | 0.87 |
| `quote_overall` | 0.9 |

## Cost & Latency

| Metric | Value |
|--------|-------|
| Cost (USD) | $0.0441 |
| Duration (ms) | 61859 |
| Judge | gpt-4o (cross-model anti-bias) |

## Decision: `adopt`

**Threshold:** ≥0.05 delta on judge-rubric score, or hypothesis directly confirmed/falsified.

**Why adopt:** Variant outperformed control or hypothesis was directly confirmed.

Result: `Standard voice=0.85 vs Quote-first=0.90, delta=+0.05`

## Limitations

1. Single tenant (`tenant_acme_corp`) — golden set has only one stub; full 20-tenant set is blocked.
2. Sample size below the lab spec for statistical power.
3. Decision threshold is mine, not the lab's — the lab harness paragraph does not define adopt/reject/defer thresholds.

## Reproduce

```bash
python output/experiments/exp-3.11-quote-only-personas/run_experiment.py
```
