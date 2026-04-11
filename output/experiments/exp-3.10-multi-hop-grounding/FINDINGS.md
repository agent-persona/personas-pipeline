# Experiment 3.10: Multi-hop grounding

**Branch**: yash/experiments
**Date**: 2026-04-10
**Owner**: Yash
**Group**: 3
**Size**: M
**Due**: Tue 2026-04-14
**Signal**: WEAK
**Recommendation**: defer

---

## Hypothesis

Multi-hop evidence chains (record → insight → claim) improve groundedness over single-hop because they force the model to derive insights before claims.

## Control

Single-hop evidence (current synthesize)

## Variant

Multi-hop chain via direct LLM calls

## Method

Run via canonical `personas-pipeline` modules:
- `synthesis.engine.synthesizer.synthesize` for persona generation
- `synthesis.models.persona.PersonaV1` for the schema
- `evaluation.judges.LLMJudge` for scoring (cross-model anti-bias: GPT-4o judges Claude output)
- `evaluation.registry.save_run` for the run record

Tenant: `tenant_acme_corp` from `evaluation.golden_set` (single stub tenant — full 20-tenant golden set blocked).

Sample size: **2** personas.

## Result

> Single-hop grounded=0.90 vs Multi-hop=0.90, delta=+0.00

## Metrics

| Metric | Value |
|--------|-------|
| `single_hop_grounded` | 0.9 |
| `multi_hop_grounded` | 0.9 |
| `single_hop_overall` | 0.88 |
| `multi_hop_overall` | 0.76 |

## Cost & Latency

| Metric | Value |
|--------|-------|
| Cost (USD) | $0.0409 |
| Duration (ms) | 49621 |
| Judge | gpt-4o (cross-model anti-bias) |

## Decision: `defer`

**Threshold:** ≥0.05 delta on judge-rubric score, or hypothesis directly confirmed/falsified.

**Why defer:** Inconclusive at this sample size — delta within ±0.05, synthesis ceiling effect, or pipeline failure.

Result: `Single-hop grounded=0.90 vs Multi-hop=0.90, delta=+0.00`

## Limitations

1. Single tenant (`tenant_acme_corp`) — golden set has only one stub; full 20-tenant set is blocked.
2. Sample size below the lab spec for statistical power.
3. Decision threshold is mine, not the lab's — the lab harness paragraph does not define adopt/reject/defer thresholds.

## Reproduce

```bash
python output/experiments/exp-3.10-multi-hop-grounding/run_experiment.py
```
