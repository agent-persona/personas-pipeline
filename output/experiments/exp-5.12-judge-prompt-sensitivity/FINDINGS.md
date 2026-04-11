# Experiment 5.12: Judge Prompt Sensitivity

## Hypothesis
Minor prompt rewording produces meaningful score shifts, exposing judge fragility.

## Method
1. Generated `6` personas from repeated golden-tenant synthesis.
2. Scored each persona with `6` rubric prompt variants.
3. Computed per-persona overall variance and per-dimension coefficient of variation.

- Synthesis model: `claude-haiku-4-5-20251001`
- Judge model: `claude-sonnet-4-20250514`

## Per-Persona Overall Variance
- `run_1_cluster_1`: 0.139
- `run_1_cluster_2`: 0.000
- `run_2_cluster_1`: 0.250
- `run_2_cluster_2`: 0.000
- `run_3_cluster_1`: 0.222
- `run_3_cluster_2`: 0.000

## Per-Dimension Coefficient of Variation
- `grounded`: 0.098
- `distinctive`: 0.098
- `coherent`: 0.046
- `actionable`: 0.056
- `voice_fidelity`: 0.105

## Variant Mean Overall Scores
- `formal_baseline`: mean overall `5.00`
- `casual`: mean overall `4.83`
- `numbered_checklist`: mean overall `4.67`
- `negative_framing`: mean overall `5.00`
- `reordered`: mean overall `4.67`
- `terse`: mean overall `4.83`

## Sensitivity Readout
- Most sensitive dimension: `voice_fidelity`
- Least sensitive dimension: `coherent`

## Decision
Adopt. Prompt wording moved mean overall scores from `4.67` to `5.00`,
and `voice_fidelity` showed the highest coefficient of variation (`0.105`),
which is enough instability to justify version-pinning the judge prompt.

## Caveat
Small persona set from one stub tenant; score variance is directional, not definitive.
