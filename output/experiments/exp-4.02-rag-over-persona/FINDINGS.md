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

Per-turn field retrieval (RAG) produces better consistency than full persona in system prompt because it reduces context dilution on long conversations.

## Control

Full persona JSON in system prompt (default TwinChat)

## Variant

Per-turn relevant-field retrieval based on question content

## Method

Run via canonical `personas-pipeline` modules:
- `synthesis.engine.synthesizer.synthesize` for persona generation
- `synthesis.models.persona.PersonaV1` for the schema
- `evaluation.judges.LLMJudge` for scoring (cross-model anti-bias: GPT-4o judges Claude output)
- `evaluation.registry.save_run` for the run record

Tenant: `tenant_acme_corp` from `evaluation.golden_set` (single stub tenant — full 20-tenant golden set blocked).

Sample size: **10** personas.

## Result

> Full prompt=0.89 vs RAG=0.87, delta=-0.02

## Metrics

| Metric | Value |
|--------|-------|
| `full_prompt_overall` | 0.89 |
| `rag_overall` | 0.87 |

## Cost & Latency

| Metric | Value |
|--------|-------|
| Cost (USD) | $0.0556 |
| Duration (ms) | 71742 |
| Judge | gpt-4o (cross-model anti-bias) |

## Decision: `defer`

**Threshold:** ≥0.05 delta on judge-rubric score, or hypothesis directly confirmed/falsified.

**Why defer:** Inconclusive at this sample size — delta within ±0.05, synthesis ceiling effect, or pipeline failure.

Result: `Full prompt=0.89 vs RAG=0.87, delta=-0.02`

## Limitations

1. Single tenant (`tenant_acme_corp`) — golden set has only one stub; full 20-tenant set is blocked.
2. Sample size below the lab spec for statistical power.
3. Decision threshold is mine, not the lab's — the lab harness paragraph does not define adopt/reject/defer thresholds.

## Reproduce

```bash
python output/experiments/exp-4.02-rag-over-persona/run_experiment.py
```
