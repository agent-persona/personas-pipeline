# Experiment 5.09: Eval drift over model versions

## Metadata
- **Branch**: exp-5.09-eval-drift-model-versions
- **Date**: 2026-04-10
- **Problem Space**: 5

## Hypothesis
Judges become measurably harsher or kinder with model generations; requires version pinning

## Changes Made
- `evals/version_drift.py`: Version-pinned eval harness that saves per-model-version score snapshots to `output/experiments/exp-5.09-eval-drift-model-versions/scores_v{model_version}.json`

## Results

### Baseline Scores (claude-sonnet-4-6)
| Persona | grounded | distinctive | coherent | actionable | voice_fidelity | mean |
|---|---|---|---|---|---|---|
| Alex the API-First DevOps Engineer | 5/5 | 5/5 | 5/5 | 5/5 | 4/5 | 4.80/5 |
| Maya the Freelance Brand Designer | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5.00/5 |
| **Overall mean** | | | | | | **4.90/5** |

### Schema validity: 1.0 (both personas pass all structural checks)
### Deterministic groundedness: 1.0 (both personas report full groundedness from synthesis pipeline)

### Target Metric: Score delta per version bump
**Baseline established**: mean judge score = **4.90/5** on claude-sonnet-4-6

Future runs will save new snapshot files (`scores_v{model}.json`) alongside this baseline.
A diff of snapshots reveals calibration drift. Threshold: any dimension delta > 0.3 warrants investigation.

## Signal Strength: **MODERATE** (harness built, baseline established — real drift signal requires future model releases)
## Recommendation: **adopt**
Version pinning framework is ready. Baseline locked at 4.90/5. Run again when claude-haiku-4-5 or
claude-opus-4-6 is available to measure calibration drift. The harness is zero-cost (Claude Code as
judge) and takes under 5 seconds to execute.

## Cost
- All runs: $0.00 (Claude Code as judge; no API calls made)
