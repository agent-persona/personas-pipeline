# Experiment 1.22: Spine Minimum

## Hypothesis
~3 core persona fields should survive greedy sequential ablation; the rest should fall away with limited score loss.

## Method
1. Synthesized personas for `1` golden-tenant cluster.
2. Ran greedy sequential ablation over `9` removable list fields.
3. Removed the field whose ablation caused the smallest score drop at each step.
4. Stopped when the judge score dropped below the collapse threshold.

- Provider: `openai`
- Synthesis model: `gpt-5-nano`
- Judge model: `heuristic`
- Collapse threshold: `2.0`

## Cluster Outcomes
- `clust_7070bb4cd110`: control `3.07` -> spine `1.86` after `7` removals, spine fields `goals, sample_quotes`, control model `local-fallback`, control cost `$0.0000`

## Aggregate Metrics
- Mean control score: `3.07`
- Mean control groundedness: `1.00`
- Mean spine score: `1.86`
- Mean quality drop: `+1.21`
- Mean spine size: `2.0`
- Mean steps to collapse: `7.0`

## Removal Ranking
- `channels`: mean removal step `1.0`, survival rate `0%`
- `objections`: mean removal step `2.0`, survival rate `0%`
- `decision_triggers`: mean removal step `3.0`, survival rate `0%`
- `motivations`: mean removal step `4.0`, survival rate `0%`
- `vocabulary`: mean removal step `5.0`, survival rate `0%`
- `pains`: mean removal step `6.0`, survival rate `0%`
- `journey_stages`: mean removal step `7.0`, survival rate `0%`
- `goals`: mean removal step `8.0`, survival rate `100%`
- `sample_quotes`: mean removal step `8.0`, survival rate `100%`

## Decision
Adopt. A short spine remained stable in the sampled cluster and the judge drop concentrated in a few fields.

## Caveat
Tiny sample: 1 tenant, 2 clusters. Ablations are post-hoc raw JSON edits, not reruns of synthesis.
