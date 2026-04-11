# Experiment 6.05: Stability Across Reruns

## Hypothesis
Stable source data should produce consistent persona archetypes across reruns.

## Method
1. Re-ran the golden tenant pipeline 5 times.
2. Added synthesis stochasticity with `temperature=0.7` while leaving clustering unchanged.
3. Matched each rerun back to run 1 via best overall persona similarity.
4. Measured cross-run similarity on names, summaries, goals, pains, vocabulary, and quotes.

- Model: `claude-haiku-4-5-20251001`
- Temperature: `0.7`
- Runs: `5`

## Baseline-Referenced Comparisons
- `run_2`: mean similarity `0.283`, archetype recurrence `0.0%`
- `run_3`: mean similarity `0.284`, archetype recurrence `0.0%`
- `run_4`: mean similarity `0.287`, archetype recurrence `0.0%`
- `run_5`: mean similarity `0.291`, archetype recurrence `0.0%`

## Aggregate Metrics
- Mean cross-run similarity: `0.286`
- Mean archetype recurrence: `0.0%`

## Decision
Reject. Mean cross-run similarity only reached `0.286` and archetype
recurrence stayed at `0.0%`, so the same tenant did not regenerate stable
persona archetypes under stochastic synthesis.

## Caveat
Only 1 tenant and 2 natural clusters. Variation here comes mostly from synthesis stochasticity, not segmentation.
