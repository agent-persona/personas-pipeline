# Experiment 4.01: System prompt format

**Branch**: yash/experiments
**Date**: 2026-04-10
**Owner**: Yash
**Group**: 4
**Size**: S
**Due**: Fri 2026-04-10
**Signal**: WEAK
**Recommendation**: defer

---

## Hypothesis

Structured system prompts (IDENTITY/PERSONALITY/KNOWLEDGE/RULES sections) produce more consistent twin behavior than narrative prompts because labeled sections give cleaner boundaries.

## Control

build_persona_system_prompt() narrative format

## Variant

Structured format with IDENTITY/PERSONALITY/KNOWLEDGE/RULES sections

## Method

Run via canonical `personas-pipeline` modules:
- `synthesis.engine.synthesizer.synthesize` for persona generation
- `synthesis.models.persona.PersonaV1` for the schema
- `evaluation.judges.LLMJudge` for scoring (cross-model anti-bias: GPT-4o judges Claude output)
- `evaluation.registry.save_run` for the run record

Tenant: `tenant_acme_corp` from `evaluation.golden_set` (single stub tenant — full 20-tenant golden set blocked).

Sample size: **6** personas.

## Result

> Narrative=0.87 vs Structured=0.87, delta=+0.00

## Metrics

| Metric | Value |
|--------|-------|
| `narrative_overall` | 0.87 |
| `structured_overall` | 0.87 |
| `narrative_dims` | voice_consistency=0.85, personality_adherence=0.9, knowledge_boundaries=0.95, character_stability=0.85, distinctiveness= |
| `structured_dims` | voice_consistency=0.85, personality_adherence=0.9, knowledge_boundaries=0.95, character_stability=0.85, distinctiveness= |

## Cost & Latency

| Metric | Value |
|--------|-------|
| Cost (USD) | $0.0411 |
| Duration (ms) | 61575 |
| Judge | gpt-4o (cross-model anti-bias) |

## Decision: `defer`

**Threshold:** ≥0.05 delta on judge-rubric score, or hypothesis directly confirmed/falsified.

**Why defer:** Inconclusive at this sample size — delta within ±0.05, synthesis ceiling effect, or pipeline failure.

Result: `Narrative=0.87 vs Structured=0.87, delta=+0.00`

## Limitations

1. Single tenant (`tenant_acme_corp`) — golden set has only one stub; full 20-tenant set is blocked.
2. Sample size below the lab spec for statistical power.
3. Decision threshold is mine, not the lab's — the lab harness paragraph does not define adopt/reject/defer thresholds.

## Reproduce

```bash
python output/experiments/exp-4.01-system-prompt-format/run_experiment.py
```
