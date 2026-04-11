# Experiment 5.10: Pairwise vs Absolute Judging

## Hypothesis
Pairwise preference judging produces higher inter-judge agreement than absolute 1-5 scoring.

## Method
1. Generated 4 personas from repeated golden-tenant synthesis.
2. Scored each persona with absolute judging using Haiku and Sonnet.
3. Ran bidirectional pairwise judging on every persona pair with the same two models.
4. Converted both modes into rank orderings and compared cross-model Spearman agreement.

## Absolute-Mode Agreement
- `grounded`: `-0.600`
- `distinctive`: `-0.200`
- `coherent`: `-0.200`
- `actionable`: `-0.800`
- `voice_fidelity`: `-0.200`
- `overall`: `-0.200`
- Mean agreement: `-0.367`

## Pairwise-Mode Agreement
- `grounded`: `1.000`
- `distinctive`: `0.800`
- `coherent`: `1.000`
- `actionable`: `0.400`
- `voice_fidelity`: `1.000`
- `overall`: `0.800`
- Mean agreement: `0.833`

## Distribution Tightness
- Absolute overall stddev: `0.331`
- Pairwise rank stddev: `1.118`

## Decision
Adopt. Cross-model agreement jumped from mean Spearman `-0.367` in absolute
mode to `0.833` in pairwise mode, so pairwise judging was materially more
stable on the same persona set.

## Caveat
Tiny sample: 1 tenant, 4 personas, 2 judge models.
