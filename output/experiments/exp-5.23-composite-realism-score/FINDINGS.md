# Experiment 5.23: Composite realism score

## Metadata
- **Branch**: exp-5.23-composite-realism-score
- **Date**: 2026-04-10
- **Problem Space**: 5

## Hypothesis
Composite metric stability is achievable and correlates with downstream business outcomes

## Sub-metric Values (per persona)

| Sub-metric | persona_00 | persona_01 | Source |
|---|---|---|---|
| schema_validity | 100/100 | 100/100 | deterministic |
| groundedness | 100/100 | 100/100 | deterministic (JSON field = 1.0) |
| distinctiveness | 76.2/100 | 76.2/100 | exp-6.09 spread_score (set-level) |
| judge_rubric | 96/100 | 96/100 | Claude as judge (24/25) |
| turing_estimate | 88/100 | 90/100 | Claude as Turing rater |

**Judge rubric dimensions:** specificity of pain points, grounding in source data, internal coherence, distinctiveness from generic archetypes, actionability for product/marketing teams.

**Turing rater notes:** Alex scores 88 — highly convincing but slightly idealized. Maya scores 90 — the hourly billing math framing feels especially human. Both docked from 100 for being somewhat too clean.

## Composite Scores by Weighting Scheme

| Scheme | persona_00 | persona_01 | Mean |
|---|---|---|---|
| Equal weights | 92.04/100 | 92.44/100 | 92.24/100 |
| Groundedness-heavy | 94.03/100 | 94.33/100 | 94.18/100 |
| Distinctiveness-heavy | 88.08/100 | 88.38/100 | 88.23/100 |

### Target Metric: Score stability

- **Stability** (max - min composite): **5.95 points** (per persona, both identical)
- Interpretation: 5-15 → moderate stability; weighting choice shifts the headline by ~6 points

**Root cause of instability:** Distinctiveness (76.2) is the only sub-metric below ~88, so schemes that up-weight it pull the composite down while schemes that up-weight groundedness (100) pull it up. The 6-point spread is driven entirely by this one below-average sub-metric.

## Signal Strength: **MODERATE**
## Recommendation: **defer**

The composite score is mathematically well-behaved and produces a sensible 0-100 number. However, the 5.95-point stability gap (just above the STRONG threshold of < 5) means two analysts using different schemes report scores 6 points apart. Defer until: (1) validated with 10+ persona sets to see if stability degrades, (2) weighting scheme anchored to a specific business priority, (3) correlation with at least one downstream outcome established.

## Cost
- All runs: $0.00
