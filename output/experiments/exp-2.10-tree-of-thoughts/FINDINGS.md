# Experiment 2.10: Tree-of-Thoughts

## Hypothesis
Generate -> score -> prune -> refine yields better personas than single-shot control at a reasonable cost multiplier.

## Method
1. Ran a single-shot control on each golden-tenant cluster.
2. Generated 3 stochastic candidates per cluster.
3. Judged and pruned the lowest-scoring candidate.
4. Refined the top candidate using judge feedback as extra context.

- Synthesis model: `claude-haiku-4-5-20251001`
- Judge model: `claude-sonnet-4-20250514`

## Cluster Outcomes
- `clust_bf23bcb3db00`: control `5.00` -> refined `5.00` (delta `+0.00`)
- `clust_e8079eedea27`: control `5.00` -> refined `5.00` (delta `+0.00`)

## Aggregate
- Mean control score: `5.00`
- Mean refined score: `5.00`
- Mean convergence delta: `+0.00`
- Mean cost multiplier vs control: `3.72x`

## Decision
Reject. Tree-of-thoughts produced no score lift at all (`5.00` to `5.00`)
while increasing synthesis cost by `3.72x` versus the single-shot control.

## Caveat
Small sample: 1 tenant, 2 clusters. Diversity comes from synthesis temperature, not extra data.
