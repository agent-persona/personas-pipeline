# Experiment 4.02: RAG over persona

**Branch**: yash/experiments
**Date**: 2026-04-10
**Owner**: Yash
**Group**: 4
**Size**: M
**Due**: Sun 2026-04-12
**Signal**: WEAK
**Recommendation**: defer

---

## Hypothesis

Per-turn field retrieval (RAG) produces better consistency than full persona in system prompt on long conversations because it reduces context dilution.

## Control

Full persona JSON in system prompt, 15 turns

## Variant

Per-turn field retrieval based on question content, 15 turns

## Method

Run via canonical `personas-pipeline` modules:
- `synthesis.engine.synthesizer.synthesize` for persona generation
- `synthesis.models.persona.PersonaV1` for the schema
- `evaluation.judges.LLMJudge` for scoring (cross-model anti-bias: GPT-4o judges Claude output)
- `evaluation.registry.save_run` for the run record

Tenant: `tenant_acme_corp` from `evaluation.golden_set` (single stub tenant — full 20-tenant golden set blocked).

Sample size: **30** personas.

## Result

> Full=0.91 vs RAG=0.87 on 15-turn conversation, delta=-0.04

## Metrics

| Metric | Value |
|--------|-------|
| `full_overall` | 0.91 |
| `rag_overall` | 0.87 |
| `turns` | 15 |

## Cost & Latency

| Metric | Value |
|--------|-------|
| Cost (USD) | $0.1386 |
| Duration (ms) | 162566 |
| Judge | gpt-4o (cross-model anti-bias) |

## Decision: `defer`

**Threshold:** ≥0.05 delta on judge-rubric score, or hypothesis directly confirmed/falsified.

**Why defer:** Inconclusive at this sample size — delta within ±0.05, synthesis ceiling effect, or pipeline failure.

Result: `Full=0.91 vs RAG=0.87 on 15-turn conversation, delta=-0.04`

## Limitations

1. Single tenant (`tenant_acme_corp`) — golden set has only one stub; full 20-tenant set is blocked.
2. Sample size below the lab spec for statistical power.
3. Decision threshold is mine, not the lab's — the lab harness paragraph does not define adopt/reject/defer thresholds.

## Reproduce

```bash
python output/experiments/exp-4.02-rag-over-persona/run_experiment.py
```
