# Experiment 6.11: Outlier Persona Forced

## Hypothesis
Explicit outlier slots should improve population coverage without collapsing coherence.

## Method
1. Synthesized baseline personas from the two natural clusters in the golden tenant.
2. Selected the lowest-similarity user from each cluster and merged them into a forced outlier slot.
3. Re-synthesized the outlier slot with an explicit atypical-user prompt hint in cluster metadata.
4. Compared baseline vs forced coverage using sample record coverage and behavior coverage.

- Provider: `anthropic->openai`
- Synthesis model: `claude-haiku-4-5-20251001`
- Judge model: `claude-haiku-4-5-20251001`

## Baseline Clusters
- `clust_e874723d6058`: judge `4.00`, records `12`, behaviors `8`
- `clust_264fb85237aa`: judge `4.00`, records `12`, behaviors `8`

## Outlier Slot
- Selection: `lowest-similarity user from each natural cluster`
- Users: `user_eng_d, user_des_d`
- Judge overall: `4.00`
- Coverage lift in records: `+0.158`
- Coverage lift in behaviors: `+0.091`

## Aggregate Metrics
- Baseline mean judge: `4.00`
- Forced mean judge: `4.00`
- Baseline record coverage: `63.2%`
- Forced record coverage: `78.9%`
- Baseline behavior coverage: `72.7%`
- Forced behavior coverage: `81.8%`

## Decision
Adopt. The forced outlier slot added coverage while keeping the persona coherent enough to remain usable.

## Caveat
Tiny tenant, 38 records, only two natural clusters. This is a coverage probe, not a general clustering policy.
