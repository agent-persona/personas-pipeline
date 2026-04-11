# Experiment 2.22: Beam Search

## Hypothesis
Beam search over full-persona candidates yields quality lift at acceptable cost versus single-shot control.

## Method
1. Ran a single-shot control on each golden-tenant cluster.
2. Generated an initial beam of 3 candidate personas.
3. Scored them, refined top candidates across 2 rounds, and kept the best beam.
4. Compared final judge score and cost against the control.

- Provider: `anthropic->openai`
- Synthesis model: `claude-haiku-4-5-20251001`
- Judge model: `claude-sonnet-4-20250514`

## Cluster Outcomes
- `clust_894fb737005d`: control `5.00` -> final `5.00` (delta `+0.00`), cost `0.00x`
- `clust_f70f2de4823f`: control `4.00` -> final `5.00` (delta `+1.00`), cost `0.00x`

## Aggregate
- Mean control score: `4.50`
- Mean final score: `5.00`
- Mean quality delta: `+0.50`
- Mean cost multiplier: `0.00x`

## Decision
Adopt. Beam search raised mean quality enough to justify the added cost.

## Caveat
Small sample: 1 tenant, 2 clusters. This is full-persona beam search, not true partial-state search.
