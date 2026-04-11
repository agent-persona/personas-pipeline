# Experiment 6.23: Hierarchical Archetypes

## Hypothesis
A parent/child persona tree should improve navigability and coverage versus a flat persona list.

## Method
1. Generated a flat control persona per cluster from the golden tenant.
2. Generated a broad parent archetype per cluster, then two child variants per parent.
3. Scored every persona with the local judge helper and compared flat versus hierarchical sets.
4. Measured coverage, distinctiveness, and information density across the two representations.

- Provider: `anthropic->openai`
- Synthesis model: `claude-haiku-4-5-20251001`
- Judge model: `claude-sonnet-4-20250514`
- Clusters: `2`

## Group Metrics
- `clust_92787443519c`: flat judge `4.00`, hierarchy judge `3.55`, coverage delta `+0.00`
- `clust_17e3ebdce746`: flat judge `4.03`, hierarchy judge `3.90`, coverage delta `+0.01`

## Aggregate Metrics
- Flat mean judge: `4.00`
- Hierarchy mean judge: `3.61`
- Flat coverage: `13.34%`
- Hierarchy coverage: `13.78%`
- Within-parent distinctiveness: `0.021`
- Across-parent distinctiveness: `0.204`
- Flat information density: `42.0`
- Hierarchy information density: `38.0`

## Decision
Reject. The hierarchical tree did not improve coverage or distinctiveness enough to justify the extra personas.

## Caveat
Only one tenant and two natural clusters. Treat this as a small-sample structural comparison, not a general benchmark.
