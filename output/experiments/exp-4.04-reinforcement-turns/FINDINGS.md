# Experiment 4.04: Reinforcement turns

**Branch**: yash/experiments
**Date**: 2026-04-10
**Owner**: Yash
**Group**: 4
**Size**: S
**Due**: Sun 2026-04-12
**Signal**: WEAK
**Recommendation**: defer

---

## Hypothesis

Periodic 'stay in character' reminders reduce drift on long conversations because they re-anchor the persona before the system prompt fades from effective attention.

## Control

No reinforcement reminders, 25 turns

## Variant

Reminder every 4 turns, 25 turns

## Method

Run via canonical `personas-pipeline` modules:
- `synthesis.engine.synthesizer.synthesize` for persona generation
- `synthesis.models.persona.PersonaV1` for the schema
- `evaluation.judges.LLMJudge` for scoring (cross-model anti-bias: GPT-4o judges Claude output)
- `evaluation.registry.save_run` for the run record

Tenant: `tenant_acme_corp` from `evaluation.golden_set` (single stub tenant — full 20-tenant golden set blocked).

Sample size: **50** personas.

## Result

> No reminders=0.89 vs With reminders=0.87 on 25-turn conversation, delta=-0.02

## Metrics

| Metric | Value |
|--------|-------|
| `no_reinforcement` | 0.89 |
| `with_reinforcement` | 0.87 |
| `turns` | 25 |

## Cost & Latency

| Metric | Value |
|--------|-------|
| Cost (USD) | $0.2047 |
| Duration (ms) | 194389 |
| Judge | gpt-4o (cross-model anti-bias) |

## Decision: `defer`

**Threshold:** ≥0.05 delta on judge-rubric score, or hypothesis directly confirmed/falsified.

**Why defer:** Inconclusive at this sample size — delta within ±0.05, synthesis ceiling effect, or pipeline failure.

Result: `No reminders=0.89 vs With reminders=0.87 on 25-turn conversation, delta=-0.02`

## Limitations

1. Single tenant (`tenant_acme_corp`) — golden set has only one stub; full 20-tenant set is blocked.
2. Sample size below the lab spec for statistical power.
3. Decision threshold is mine, not the lab's — the lab harness paragraph does not define adopt/reject/defer thresholds.

## Reproduce

```bash
python output/experiments/exp-4.04-reinforcement-turns/run_experiment.py
```
