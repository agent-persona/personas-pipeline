# Experiment 2.09: Best-of-N

## Hypothesis
Best-of-N with diversity selection outperforms single-shot control at an acceptable cost multiplier.

## Method
1. Generated `N=5` candidates per cluster at `temperature=0.7`.
2. Scored each candidate with the judge helper copied from the validated judge branch pattern.
3. Selected a pure best-score persona and a composite best-diverse persona per cluster.
4. Compared each selected persona against the single-shot control on quality and cost.

- Provider: `openai`
- Synthesis model: `gpt-5-nano`
- Judge model: `gpt-5-nano`

## Cluster Outcomes
- `clust_a5ae79fa3526`: control `5.00`, best-score `5.00`, best-diverse `5.00`
- `clust_c7530cc05964`: control `4.00`, best-score `5.00`, best-diverse `5.00`

## Aggregate Metrics
- Mean control score: `4.50`
- Mean best-score score: `5.00`
- Mean best-diverse score: `5.00`
- Mean best-score gain: `+0.50`
- Mean best-diverse gain: `+0.50`
- Mean pool similarity: `0.967`

## Decision
Adopt with caution. Best-of-N improved mean judge score by `+0.50` at zero measured incremental cost in this fallback-heavy run, but most candidates converged to near-identical heuristic personas.

## Caveat
Small sample: 1 tenant, 2 clusters. Distinctiveness is heuristic and judge-based signal is coarse.
