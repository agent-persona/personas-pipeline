# Experiment 5.10: Pairwise vs Absolute Scoring

**Branch**: exp-5.10-pairwise-vs-absolute
**Base**: origin/exp-5.13 (real judge implementation)
**Date**: 2026-04-11
**Owner**: Yash
**Signal**: MODERATE (negative)
**Decision**: defer

---

## Hypothesis

Pairwise preference judging produces higher inter-judge agreement than absolute 1-5 scoring.

## Method

1. Generated 4 personas via canonical `synthesize()` on `tenant_acme_corp`
2. Scored all 4 with absolute mode using real exp-5.13 `LLMJudge` — two judge models (Sonnet, Haiku)
3. Ran full pairwise tournament (6 pairs × 2 orderings × 2 judges = 24 judge calls) using new `evals/pairwise_judging.py`
4. Position-bias debiasing: every pair compared in both (A,B) and (B,A) orderings; disagreements become TIE
5. Converted pairwise verdicts to rankings via win count and Bradley-Terry

## Results

### Inter-judge agreement

| Mode | Spearman (Sonnet vs Haiku) |
|------|---------------------------|
| Absolute (1-5) | **0.000** |
| Pairwise (win count) | **0.056** |
| Pairwise (Bradley-Terry) | **-0.105** |

Neither mode produces inter-judge agreement. Pairwise is marginally better on win count but BT actually inverts.

### Position bias

| Metric | Value |
|--------|-------|
| Disagreement rate | **43.3%** |
| Disagreements | 26 / 60 dimension-votes |

Nearly half of all pairwise dimension votes flipped when the presentation order changed.

### Absolute score distribution

| Judge | P1 | P2 | P3 | P4 | Range |
|-------|----|----|----|----|-------|
| Sonnet | 3.60 | 3.60 | 4.00 | 4.40 | 0.80 |
| Haiku | 4.00 | 4.00 | 4.00 | 4.00 | 0.00 |

Haiku assigns identical scores to all 4 personas — zero discriminative power.

### Within-model agreement (absolute vs pairwise ranking)

| Correlation | Value |
|-------------|-------|
| Abs vs win count (Sonnet) | 0.632 |
| Abs vs Bradley-Terry (Sonnet) | 0.632 |

The same judge's absolute and pairwise rankings moderately agree when the judge is Sonnet. But Haiku produces no variance in absolute mode, making this comparison degenerate for Haiku.

## Key Findings

1. **Pairwise does NOT improve inter-judge agreement.** Hypothesis falsified at this sample size.
2. **Position bias is severe (43%).** This is worse than the 30% threshold I expected. The debiasing strategy (run both orders, disagree → tie) works but converts nearly half of all signals into ties.
3. **Haiku lacks discriminative power in absolute mode** — all 4 personas scored exactly 4.00. This makes it useless as a second judge for agreement studies.
4. **Sonnet shows internal consistency** (0.63 Spearman between its own absolute and pairwise rankings) but disagrees with Haiku in both modes.

## Interpretation

This extends the 5.01 STAR finding (judge-human Spearman = 0.30). Even comparing judges to EACH OTHER (not to humans), agreement is near-zero. The eval methodology has a fundamental problem: the judge rubric is too coarse, the models are too similar in capability gap, and position bias dominates pairwise mode.

**For the team:** avoid relying on absolute numbers from LLM-as-judge. Use only for relative within-experiment comparisons (control vs variant, same judge, same run). Do NOT compare scores across experiments or across judges.

## Cost

| Component | Cost |
|-----------|------|
| 4 persona syntheses | $0.12 |
| 8 absolute judge calls | ~$0.06 |
| 24 pairwise judge calls | ~$0.18 |
| **Total** | **~$0.36** |

## Limitations

1. N=4 personas on single tenant — Spearman with N=4 has high variance
2. Only Anthropic models tested (no GPT-4o cross-vendor comparison — no API key on this branch)
3. Bradley-Terry with 6 pairs is underdetermined — need more data for stable estimates
4. Haiku's score uniformity limits what pairwise can demonstrate

## Reproduce

```bash
cd /path/to/personas-pipeline
python scripts/experiment_5_10.py
```
