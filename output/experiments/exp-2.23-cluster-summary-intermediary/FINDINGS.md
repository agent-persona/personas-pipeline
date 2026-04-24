# Experiment 2.23: Cluster summary intermediary

**Branch**: yash/experiments
**Date**: 2026-04-10
**Owner**: Yash
**Group**: 2
**Size**: S
**Due**: Mon 2026-04-13
**Signal**: WEAK
**Recommendation**: defer

---

## Hypothesis

Adding a cluster summarization step before synthesis improves quality by distilling signal from noise in raw cluster data.

## Control

Direct synthesize() from raw cluster

## Variant

Summarize cluster first, then synthesize from summary

## Method

Run via canonical `personas-pipeline` modules:
- `synthesis.engine.synthesizer.synthesize` for persona generation
- `synthesis.models.persona.PersonaV1` for the schema
- `evaluation.judges.LLMJudge` for scoring (cross-model anti-bias: GPT-4o judges Claude output)
- `evaluation.registry.save_run` for the run record

Tenant: `tenant_acme_corp` from `evaluation.golden_set` (single stub tenant — full 20-tenant golden set blocked).

Sample size: **2** personas.

## Result

> Direct=0.86 vs Summary-first=0.87, delta=+0.01

## Metrics

| Metric | Value |
|--------|-------|
| `judge_rubric_score_direct` | 0.86 |
| `judge_rubric_score_summarized` | 0.87 |

## Cost & Latency

| Metric | Value |
|--------|-------|
| Cost (USD) | $0.0544 |
| Duration (ms) | 66770 |
| Judge | gpt-4o (cross-model anti-bias) |

## Decision: `defer`

**Threshold:** ≥0.05 delta on judge-rubric score, or hypothesis directly confirmed/falsified.

**Why defer:** Inconclusive at this sample size — delta within ±0.05, synthesis ceiling effect, or pipeline failure.

Result: `Direct=0.86 vs Summary-first=0.87, delta=+0.01`

## Limitations

1. Single tenant (`tenant_acme_corp`) — golden set has only one stub; full 20-tenant set is blocked.
2. Sample size below the lab spec for statistical power.
3. Decision threshold is mine, not the lab's — the lab harness paragraph does not define adopt/reject/defer thresholds.

## Reproduce

```bash
python output/experiments/exp-2.23-cluster-summary-intermediary/run_experiment.py
```
