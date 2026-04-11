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

Structured system prompts (IDENTITY/PERSONALITY/KNOWLEDGE/RULES sections) produce more consistent twin behavior than narrative prompts on longer conversations where prompt influence decays.

## Control

build_persona_system_prompt() narrative format, 15 turns

## Variant

Structured format with labeled sections, 15 turns

## Method

Run via canonical `personas-pipeline` modules:
- `synthesis.engine.synthesizer.synthesize` for persona generation
- `synthesis.models.persona.PersonaV1` for the schema
- `evaluation.judges.LLMJudge` for scoring (cross-model anti-bias: GPT-4o judges Claude output)
- `evaluation.registry.save_run` for the run record

Tenant: `tenant_acme_corp` from `evaluation.golden_set` (single stub tenant — full 20-tenant golden set blocked).

Sample size: **30** personas.

## Result

> Narrative=0.89 vs Structured=0.89 on 15-turn conversation, delta=+0.00

## Metrics

| Metric | Value |
|--------|-------|
| `narrative_overall` | 0.89 |
| `structured_overall` | 0.89 |
| `narrative_dims` | voice_consistency=0.85, personality_adherence=0.9, knowledge_boundaries=0.95, character_stability=0.85, distinctiveness= |
| `structured_dims` | voice_consistency=0.9, personality_adherence=0.85, knowledge_boundaries=0.95, character_stability=0.9, distinctiveness=0 |
| `turns` | 15 |

## Cost & Latency

| Metric | Value |
|--------|-------|
| Cost (USD) | $0.1108 |
| Duration (ms) | 130025 |
| Judge | gpt-4o (cross-model anti-bias) |

## Decision: `defer`

**Threshold:** ≥0.05 delta on judge-rubric score, or hypothesis directly confirmed/falsified.

**Why defer:** Inconclusive at this sample size — delta within ±0.05, synthesis ceiling effect, or pipeline failure.

Result: `Narrative=0.89 vs Structured=0.89 on 15-turn conversation, delta=+0.00`

## Limitations

1. Single tenant (`tenant_acme_corp`) — golden set has only one stub; full 20-tenant set is blocked.
2. Sample size below the lab spec for statistical power.
3. Decision threshold is mine, not the lab's — the lab harness paragraph does not define adopt/reject/defer thresholds.

## Reproduce

```bash
python output/experiments/exp-4.01-system-prompt-format/run_experiment.py
```
