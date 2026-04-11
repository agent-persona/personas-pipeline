# Experiment 1.02: Structured vs narrative

**Branch**: yash/experiments
**Date**: 2026-04-10
**Owner**: Yash
**Group**: 1
**Size**: M
**Due**: Sun 2026-04-12
**Signal**: WEAK
**Recommendation**: defer

---

## Hypothesis

Structured JSON personas produce better twin conversations than narrative because the model grounds responses in specific fields rather than recalling prose.

## Control

Structured PersonaV1 JSON

## Variant

Free-form narrative persona (3 paragraphs)

## Method

Run via canonical `personas-pipeline` modules:
- `synthesis.engine.synthesizer.synthesize` for persona generation
- `synthesis.models.persona.PersonaV1` for the schema
- `evaluation.judges.LLMJudge` for scoring (cross-model anti-bias: GPT-4o judges Claude output)
- `evaluation.registry.save_run` for the run record

Tenant: `tenant_acme_corp` from `evaluation.golden_set` (single stub tenant — full 20-tenant golden set blocked).

Sample size: **6** personas.

## Result

> Structured=0.87 vs Narrative=0.88, delta=-0.01

## Metrics

| Metric | Value |
|--------|-------|
| `judge_rubric_score_structured` | 0.87 |
| `judge_rubric_score_narrative` | 0.88 |
| `structured_dims` | voice_consistency=0.85, personality_adherence=0.9, knowledge_boundaries=0.95, character_stability=0.85, distinctiveness= |
| `narrative_dims` | voice_consistency=0.85, personality_adherence=0.9, knowledge_boundaries=1.0, character_stability=0.85, distinctiveness=0 |

## Cost & Latency

| Metric | Value |
|--------|-------|
| Cost (USD) | $0.0463 |
| Duration (ms) | 70256 |
| Judge | gpt-4o (cross-model anti-bias) |

## Decision: `defer`

**Threshold:** ≥0.05 delta on judge-rubric score, or hypothesis directly confirmed/falsified.

**Why defer:** Inconclusive at this sample size — delta within ±0.05, synthesis ceiling effect, or pipeline failure.

Result: `Structured=0.87 vs Narrative=0.88, delta=-0.01`

## Limitations

1. Single tenant (`tenant_acme_corp`) — golden set has only one stub; full 20-tenant set is blocked.
2. Sample size below the lab spec for statistical power.
3. Decision threshold is mine, not the lab's — the lab harness paragraph does not define adopt/reject/defer thresholds.

## Reproduce

```bash
python output/experiments/exp-1.02-structured-vs-narrative/run_experiment.py
```
