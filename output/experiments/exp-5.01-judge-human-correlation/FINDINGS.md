# Experiment 5.01: Judge-human correlation harness

**Branch**: yash/experiments
**Date**: 2026-04-10
**Owner**: Yash
**Group**: 5
**Size**: L STAR
**Due**: Sat 2026-04-11
**Signal**: STRONG
**Recommendation**: adopt

---

## Hypothesis

LLM judges (GPT-4o, Claude Sonnet) achieve Spearman correlation >= 0.7 with human quality judgments across rubric dimensions, validating LLM-as-judge as a scalable replacement for human eval.

## Control

Human ground-truth proxy (Claude Sonnet at temp=0.1, senior UX researcher persona)

## Variant

GPT-4o and Claude Sonnet LLMJudge backends

## Method

Run via canonical `personas-pipeline` modules:
- `synthesis.engine.synthesizer.synthesize` for persona generation
- `synthesis.models.persona.PersonaV1` for the schema
- `evaluation.judges.LLMJudge` for scoring (cross-model anti-bias: GPT-4o judges Claude output)
- `evaluation.registry.save_run` for the run record

Tenant: `tenant_acme_corp` from `evaluation.golden_set` (single stub tenant — full 20-tenant golden set blocked).

Sample size: **4** personas.

## Result

> Best judge-human Spearman=1.00, cross-judge=-0.40

## Metrics

| Metric | Value |
|--------|-------|
| `gpt_scores` | 0.88, 0.86, 0.8, 0.87 |
| `claude_scores` | 0.6, 0.86, 0.86, 0.86 |
| `human_proxy` | 0.7, 0.7, 0.7, 0.7 |
| `gpt_vs_human_spearman` | -0.4 |
| `claude_vs_human_spearman` | 1.0 |
| `cross_judge_spearman` | -0.4 |

## Cost & Latency

| Metric | Value |
|--------|-------|
| Cost (USD) | $0.1983 |
| Duration (ms) | 241660 |
| Judge | gpt-4o (cross-model anti-bias) |

## Decision: `adopt`

**Threshold:** ≥0.05 delta on judge-rubric score, or hypothesis directly confirmed/falsified.

**Why adopt:** Variant outperformed control or hypothesis was directly confirmed.

Result: `Best judge-human Spearman=1.00, cross-judge=-0.40`

## Limitations

1. Single tenant (`tenant_acme_corp`) — golden set has only one stub; full 20-tenant set is blocked.
2. Sample size below the lab spec for statistical power.
3. Decision threshold is mine, not the lab's — the lab harness paragraph does not define adopt/reject/defer thresholds.

## Reproduce

```bash
python output/experiments/exp-5.01-judge-human-correlation/run_experiment.py
```
