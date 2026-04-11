# Experiment 5.01: Judge-human correlation harness

**Branch**: yash/experiments
**Date**: 2026-04-10
**Owner**: Yash
**Group**: 5
**Size**: L STAR
**Due**: Sat 2026-04-11
**Signal**: WEAK
**Recommendation**: defer

---

## Hypothesis

LLM judges (GPT-4o, Claude Sonnet) achieve Spearman correlation >= 0.7 with human quality judgments across rubric dimensions when the human proxy has meaningful variance.

## Control

Human ground-truth proxy (Claude Sonnet as senior UX researcher) with forced variance

## Variant

GPT-4o and Claude Sonnet LLMJudge backends

## Method

Run via canonical `personas-pipeline` modules:
- `synthesis.engine.synthesizer.synthesize` for persona generation
- `synthesis.models.persona.PersonaV1` for the schema
- `evaluation.judges.LLMJudge` for scoring (cross-model anti-bias: GPT-4o judges Claude output)
- `evaluation.registry.save_run` for the run record

Tenant: `tenant_acme_corp` from `evaluation.golden_set` (single stub tenant — full 20-tenant golden set blocked).

Sample size: **10** personas.

## Result

> N=10. Best judge-human Spearman=0.297 (gpt=0.297, claude=0.200), cross-judge=0.152

## Metrics

| Metric | Value |
|--------|-------|
| `gpt_scores` | 0.88, 0.84, 0.88, 0.82, 0.84, 0.88, 0.88, 0.84, 0.84, 0.84 |
| `claude_scores` | 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86 |
| `human_proxy` | 0.7, 0.6, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.6 |
| `human_proxy_range` | 0.60..0.70 |
| `gpt_vs_human_spearman` | 0.297 |
| `claude_vs_human_spearman` | 0.2 |
| `cross_judge_spearman` | 0.152 |

## Cost & Latency

| Metric | Value |
|--------|-------|
| Cost (USD) | $0.4650 |
| Duration (ms) | 550836 |
| Judge | gpt-4o (cross-model anti-bias) |

## Decision: `defer`

**Threshold:** ≥0.05 delta on judge-rubric score, or hypothesis directly confirmed/falsified.

**Why defer:** Inconclusive at this sample size — delta within ±0.05, synthesis ceiling effect, or pipeline failure.

Result: `N=10. Best judge-human Spearman=0.297 (gpt=0.297, claude=0.200), cross-judge=0.152`

## Limitations

1. Single tenant (`tenant_acme_corp`) — golden set has only one stub; full 20-tenant set is blocked.
2. Sample size below the lab spec for statistical power.
3. Decision threshold is mine, not the lab's — the lab harness paragraph does not define adopt/reject/defer thresholds.

## Reproduce

```bash
python output/experiments/exp-5.01-judge-human-correlation/run_experiment.py
```
