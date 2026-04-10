# Experiment 5.19: Continuous vs binary metrics

## Metadata
- **Branch**: exp-5.19-continuous-vs-binary-metrics
- **Date**: 2026-04-10
- **Problem Space**: 5

## Hypothesis
Binary metrics yield more robust signals than scales despite coarser resolution

## Results

### Judge Scores (by dimension)
| Dimension | persona_00 binary | persona_00 1-5 | persona_01 binary | persona_01 1-5 |
|---|---|---|---|---|
| grounded | 1 | 5 | 1 | 5 |
| distinctive | 1 | 5 | 1 | 4 |
| coherent | 1 | 5 | 1 | 5 |
| actionable | 1 | 5 | 1 | 5 |
| voice_fidelity | 1 | 4 | 1 | 4 |

### Reliability Analysis
| Scale | Discriminating dimensions | Coefficient of variation | Verdict |
|---|---|---|---|
| Binary (0/1) | 0/5 dimensions different | N/A | not robust |
| Continuous (1-5) | N/A | 0.0975 | robust |

### Target Metric: Reliability per scale type
- Binary discrimination rate: **0.00** (0% of dimensions where personas scored differently)
- Continuous pooled CV: **0.0975** (spread exists; captures quality gradient)

## Signal Strength: **MODERATE**
## Recommendation: **reject**

Binary scale produced zero discrimination between two high-quality personas — both scored 1 (pass) on every single dimension. The hypothesis that binary is more robust is **refuted** for this persona quality tier. Continuous scale captured meaningful nuance: voice_fidelity scored 4 vs 5 for persona_00, and distinctive scored 4 vs 5 for persona_01, enabling cross-persona comparison and quality gradients that binary simply cannot express.

The continuous CV of 0.0975 is modest (most dimensions are near-ceiling), but it is non-zero — binary CV is definitionally zero here. For pipelines producing uniformly high-quality personas (both passed every bar), binary metrics become a flat signal: useless for ranking, regression detection, or A/B evaluation of generation strategies. Recommend the continuous 1-5 scale as the default rubric, with binary reserved as a minimum-bar gate (pass/fail threshold) rather than a primary quality signal.

## Cost
- All runs: $0.00 (Claude Code as judge)
