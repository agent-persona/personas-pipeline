# Merge Decisions — Experiment Branch Evaluation

Evaluated 51 runtime-changing branches against main baseline.

## Summary

| Verdict | Count | Meaning |
|---------|-------|---------|
| MERGE | 4 | Clear improvement, safe to merge |
| REVIEW | 7 | Mixed signals, human read-through needed |
| NEUTRAL | 19 | No meaningful change; safe to merge if orthogonal |
| REJECT | 10 | Quality regression |
| REJECT-RELIABILITY | 10 | Reduced success rate |
| REJECT-BROKEN | 1 | Zero successful personas — branch is broken |
| NO-RESULTS | 0 | Benchmark didn't run — infra issue |

## Legend
- **ΔG**: change in mean groundedness vs main (higher is better)
- **ΔC%**: % change in total cost vs main (negative is better)
- **ΔS**: change in success rate vs main (higher is better)
- **ΔJ**: change in mean judge score vs main (higher is better)

## Ranked verdicts

| Branch | Verdict | ΔJ | ΔG | ΔC% | ΔS | Notes |
|--------|---------|----|----|-----|----|-------|
| `exp-1.11-negative-space` | **MERGE** | +0.26 | -0.006 | -4.1% | -0.05 | rt-files=3 |
| `exp-2.07-order-of-fields` | **MERGE** | +0.21 | -0.006 | -3.9% | -0.02 | rt-files=3 |
| `exp-4.14-latency-vs-realism` | **MERGE** | +0.18 | -0.014 | -5.8% | -0.05 | rt-files=1 |
| `exp-6.04` | **MERGE** | +0.12 | -0.006 | -8.7% | +0.00 | rt-files=2 |
| `exp-1.20` | **REVIEW** | +0.11 | -0.006 | -1.3% | -0.05 | rt-files=2 |
| `exp-5.10-pairwise-vs-absolute-yash` | **REVIEW** | +0.09 | -0.009 | -2.8% | -0.02 | rt-files=1 |
| `exp-1.03-vocab-anchoring` | **REVIEW** | +0.09 | -0.005 | -1.6% | -0.05 | rt-files=5 |
| `exp-2.16-prompt-compression` | **REVIEW** | +0.09 | -0.005 | +1.8% | -0.05 | rt-files=2 |
| `exp-2.06-temperature-sweep` | **REVIEW** | +0.08 | -0.010 | -6.1% | -0.02 | rt-files=1 |
| `exp-5.12-judge-prompt-sensitivity` | **REVIEW** | +0.06 | -0.007 | -5.6% | -0.02 | rt-files=1 |
| `exp-2.18-prompt-prefix-caching` | **REVIEW** | +0.01 | -0.015 | -10.4% | -0.05 | rt-files=2 |
| `exp-1.19-schema-artifact-format` | **NEUTRAL** | +0.04 | -0.007 | -7.7% | -0.02 | rt-files=2 |
| `exp-2.09-best-of-n` | **NEUTRAL** | +0.04 | -0.012 | -6.4% | -0.05 | rt-files=1 |
| `exp-b1-semantic-groundedness-proxy` | **NEUTRAL** | +0.03 | -0.004 | -5.8% | -0.02 | rt-files=1 |
| `exp-3.19-source-weighting-recency` | **NEUTRAL** | +0.03 | -0.008 | -9.6% | -0.02 | rt-files=1 |
| `exp-1.01-schema-width` | **NEUTRAL** | +0.02 | -0.009 | +0.6% | -0.02 | rt-files=4 |
| `exp-4.07` | **NEUTRAL** | +0.01 | -0.005 | -7.7% | -0.05 | rt-files=1 |
| `exp-3.18-pii-stripped` | **NEUTRAL** | -0.00 | -0.006 | -2.5% | -0.05 | rt-files=1 |
| `exp-2.14-constitutional-persona` | **NEUTRAL** | -0.01 | -0.008 | +1.4% | +0.00 | rt-files=1 |
| `exp-2.05-few-shot-exemplars` | **NEUTRAL** | -0.01 | -0.004 | +0.1% | +0.00 | rt-files=2 |
| `exp-1.16-persona-to-persona-references` | **NEUTRAL** | -0.01 | -0.004 | +0.6% | +0.00 | rt-files=2 |
| `exp-5.10-pairwise-vs-absolute` | **NEUTRAL** | -0.01 | -0.010 | -5.3% | -0.05 | rt-files=1 |
| `exp-5.05-rubric-ablation` | **NEUTRAL** | -0.01 | -0.016 | -4.1% | -0.05 | rt-files=1 |
| `exp-4.10-personality-strength-dial` | **NEUTRAL** | -0.01 | -0.008 | -5.7% | -0.02 | rt-files=2 |
| `exp-2.10-tree-of-thoughts` | **NEUTRAL** | -0.02 | -0.007 | -9.9% | -0.02 | rt-files=1 |
| `exp-1.15-edge-case-behavior-fields` | **NEUTRAL** | -0.02 | +0.000 | +1.0% | -0.05 | rt-files=4 |
| `exp-1.17-length-budgets-per-field` | **NEUTRAL** | -0.03 | -0.005 | -0.1% | +0.00 | rt-files=2 |
| `exp-3.20-confidence-weighted-corroboration` | **NEUTRAL** | -0.03 | -0.009 | -0.8% | -0.05 | rt-files=2 |
| `exp-4.06` | **NEUTRAL** | -0.04 | -0.007 | -7.1% | -0.05 | rt-files=2 |
| `exp-2.22-beam-search` | **NEUTRAL** | -0.04 | -0.001 | -5.6% | -0.02 | rt-files=1 |
| `exp-4.11` | **REJECT-RELIABILITY** | +0.18 | -0.005 | -0.7% | -0.10 | rt-files=1 |
| `exp-6.05-stability-across-reruns` | **REJECT-RELIABILITY** | +0.15 | -0.009 | -2.5% | -0.10 | rt-files=1 |
| `exp-4.21-curiosity-behavior` | **REJECT-RELIABILITY** | +0.09 | -0.010 | -0.8% | -0.10 | rt-files=1 |
| `exp-5.04-position-verbosity-bias` | **REJECT-RELIABILITY** | +0.08 | -0.006 | -3.2% | -0.07 | rt-files=1 |
| `exp-2.12-self-consistency-voting` | **REJECT-RELIABILITY** | -0.00 | -0.003 | -0.1% | -0.07 | rt-files=1 |
| `exp-3.06` | **REJECT-RELIABILITY** | -0.01 | -0.003 | +4.4% | -0.10 | rt-files=4 |
| `exp-2.08-synthetic-warmstart` | **REJECT-RELIABILITY** | -0.02 | -0.011 | -1.0% | -0.07 | rt-files=2 |
| `exp-5.14` | **REJECT-RELIABILITY** | -0.04 | -0.009 | -7.5% | -0.07 | rt-files=1 |
| `exp-5.13` | **REJECT-RELIABILITY** | -0.06 | -0.013 | -4.4% | -0.10 | rt-files=1 |
| `exp-4.13-length-matching` | **REJECT-RELIABILITY** | -0.11 | -0.007 | -3.3% | -0.07 | rt-files=1 |
| `exp-3.03-retrieval-augmented-synthesis` | **REJECT** | +0.11 | -0.011 | +1.7% | -0.14 | rt-files=2 |
| `exp-2.17-system-vs-user-prompt` | **REJECT** | +0.11 | -0.004 | -2.5% | -0.12 | rt-files=2 |
| `exp-3.17-evidence-ablation` | **REJECT** | +0.11 | -0.006 | -6.1% | -0.12 | rt-files=1 |
| `exp-4.23-persona-wake-words` | **REJECT** | +0.09 | -0.017 | -6.7% | -0.12 | rt-files=1 |
| `exp-5.11-reference-based-vs-reference-free` | **REJECT** | +0.06 | -0.004 | +1.1% | -0.12 | rt-files=1 |
| `exp-4.15-cold-start-warmup` | **REJECT** | +0.05 | -0.010 | +6.5% | -0.12 | rt-files=1 |
| `exp-1.14-belief-value-separation` | **REJECT** | +0.04 | -0.011 | +45.2% | -0.40 | rt-files=3 |
| `exp-3.14-negative-evidence` | **REJECT** | +0.04 | -0.003 | +6.5% | -0.12 | rt-files=1 |
| `exp-1.24-stylometric-anchors` | **REJECT** | +0.03 | -0.010 | +4.2% | -0.14 | rt-files=4 |
| `exp-1.23-internalized-contradictions` | **REJECT** | -0.17 | -0.012 | +16.3% | -0.14 | rt-files=3 |
| `exp-3.13-temporal-grounding` | **REJECT-BROKEN** | — | -0.999 | +70.5% | -1.00 | rt-files=2 |

## Raw metrics per branch

### `exp-1.11-negative-space` — MERGE

- Personas: 40/42 (main: 42/42)
- Groundedness: 0.993 (main: 0.999)
- Judge score: 4.50 (main: 4.24)
- Mean attempts: 1.70 (main: 2.00)
- Total cost: $1.4851 (main: $1.5492)

### `exp-2.07-order-of-fields` — MERGE

- Personas: 41/42 (main: 42/42)
- Groundedness: 0.993 (main: 0.999)
- Judge score: 4.45 (main: 4.24)
- Mean attempts: 1.85 (main: 2.00)
- Total cost: $1.4887 (main: $1.5492)

### `exp-4.14-latency-vs-realism` — MERGE

- Personas: 40/42 (main: 42/42)
- Groundedness: 0.985 (main: 0.999)
- Judge score: 4.42 (main: 4.24)
- Mean attempts: 1.80 (main: 2.00)
- Total cost: $1.4590 (main: $1.5492)

### `exp-6.04` — MERGE

- Personas: 42/42 (main: 42/42)
- Groundedness: 0.993 (main: 0.999)
- Judge score: 4.36 (main: 4.24)
- Mean attempts: 1.81 (main: 2.00)
- Total cost: $1.4151 (main: $1.5492)

### `exp-1.20` — REVIEW

- Personas: 40/42 (main: 42/42)
- Groundedness: 0.993 (main: 0.999)
- Judge score: 4.35 (main: 4.24)
- Mean attempts: 1.88 (main: 2.00)
- Total cost: $1.5296 (main: $1.5492)

### `exp-5.10-pairwise-vs-absolute-yash` — REVIEW

- Personas: 41/42 (main: 42/42)
- Groundedness: 0.990 (main: 0.999)
- Judge score: 4.34 (main: 4.24)
- Mean attempts: 1.88 (main: 2.00)
- Total cost: $1.5055 (main: $1.5492)

### `exp-1.03-vocab-anchoring` — REVIEW

- Personas: 40/42 (main: 42/42)
- Groundedness: 0.994 (main: 0.999)
- Judge score: 4.33 (main: 4.24)
- Mean attempts: 1.90 (main: 2.00)
- Total cost: $1.5238 (main: $1.5492)

### `exp-2.16-prompt-compression` — REVIEW

- Personas: 40/42 (main: 42/42)
- Groundedness: 0.994 (main: 0.999)
- Judge score: 4.33 (main: 4.24)
- Mean attempts: 1.95 (main: 2.00)
- Total cost: $1.5764 (main: $1.5492)

### `exp-2.06-temperature-sweep` — REVIEW

- Personas: 41/42 (main: 42/42)
- Groundedness: 0.989 (main: 0.999)
- Judge score: 4.32 (main: 4.24)
- Mean attempts: 1.80 (main: 2.00)
- Total cost: $1.4550 (main: $1.5492)

### `exp-5.12-judge-prompt-sensitivity` — REVIEW

- Personas: 41/42 (main: 42/42)
- Groundedness: 0.992 (main: 0.999)
- Judge score: 4.31 (main: 4.24)
- Mean attempts: 1.80 (main: 2.00)
- Total cost: $1.4624 (main: $1.5492)

### `exp-2.18-prompt-prefix-caching` — REVIEW

- Personas: 40/42 (main: 42/42)
- Groundedness: 0.984 (main: 0.999)
- Judge score: 4.25 (main: 4.24)
- Mean attempts: 1.68 (main: 2.00)
- Total cost: $1.3879 (main: $1.5492)

### `exp-1.19-schema-artifact-format` — NEUTRAL

- Personas: 41/42 (main: 42/42)
- Groundedness: 0.992 (main: 0.999)
- Judge score: 4.28 (main: 4.24)
- Mean attempts: 1.78 (main: 2.00)
- Total cost: $1.4297 (main: $1.5492)

### `exp-2.09-best-of-n` — NEUTRAL

- Personas: 40/42 (main: 42/42)
- Groundedness: 0.987 (main: 0.999)
- Judge score: 4.28 (main: 4.24)
- Mean attempts: 1.77 (main: 2.00)
- Total cost: $1.4500 (main: $1.5492)

### `exp-b1-semantic-groundedness-proxy` — NEUTRAL

- Personas: 41/42 (main: 42/42)
- Groundedness: 0.995 (main: 0.999)
- Judge score: 4.28 (main: 4.24)
- Mean attempts: 1.83 (main: 2.00)
- Total cost: $1.4595 (main: $1.5492)

### `exp-3.19-source-weighting-recency` — NEUTRAL

- Personas: 41/42 (main: 42/42)
- Groundedness: 0.991 (main: 0.999)
- Judge score: 4.27 (main: 4.24)
- Mean attempts: 1.78 (main: 2.00)
- Total cost: $1.4003 (main: $1.5492)

### `exp-1.01-schema-width` — NEUTRAL

- Personas: 41/42 (main: 42/42)
- Groundedness: 0.990 (main: 0.999)
- Judge score: 4.27 (main: 4.24)
- Mean attempts: 1.90 (main: 2.00)
- Total cost: $1.5579 (main: $1.5492)

### `exp-4.07` — NEUTRAL

- Personas: 40/42 (main: 42/42)
- Groundedness: 0.993 (main: 0.999)
- Judge score: 4.25 (main: 4.24)
- Mean attempts: 1.75 (main: 2.00)
- Total cost: $1.4306 (main: $1.5492)

### `exp-3.18-pii-stripped` — NEUTRAL

- Personas: 40/42 (main: 42/42)
- Groundedness: 0.992 (main: 0.999)
- Judge score: 4.24 (main: 4.24)
- Mean attempts: 1.85 (main: 2.00)
- Total cost: $1.5108 (main: $1.5492)

### `exp-2.14-constitutional-persona` — NEUTRAL

- Personas: 42/42 (main: 42/42)
- Groundedness: 0.991 (main: 0.999)
- Judge score: 4.24 (main: 4.24)
- Mean attempts: 2.00 (main: 2.00)
- Total cost: $1.5705 (main: $1.5492)

### `exp-2.05-few-shot-exemplars` — NEUTRAL

- Personas: 42/42 (main: 42/42)
- Groundedness: 0.995 (main: 0.999)
- Judge score: 4.24 (main: 4.24)
- Mean attempts: 1.98 (main: 2.00)
- Total cost: $1.5509 (main: $1.5492)

### `exp-1.16-persona-to-persona-references` — NEUTRAL

- Personas: 42/42 (main: 42/42)
- Groundedness: 0.995 (main: 0.999)
- Judge score: 4.24 (main: 4.24)
- Mean attempts: 1.98 (main: 2.00)
- Total cost: $1.5583 (main: $1.5492)

### `exp-5.10-pairwise-vs-absolute` — NEUTRAL

- Personas: 40/42 (main: 42/42)
- Groundedness: 0.988 (main: 0.999)
- Judge score: 4.23 (main: 4.24)
- Mean attempts: 1.85 (main: 2.00)
- Total cost: $1.4678 (main: $1.5492)

### `exp-5.05-rubric-ablation` — NEUTRAL

- Personas: 40/42 (main: 42/42)
- Groundedness: 0.983 (main: 0.999)
- Judge score: 4.23 (main: 4.24)
- Mean attempts: 1.82 (main: 2.00)
- Total cost: $1.4852 (main: $1.5492)

### `exp-4.10-personality-strength-dial` — NEUTRAL

- Personas: 41/42 (main: 42/42)
- Groundedness: 0.990 (main: 0.999)
- Judge score: 4.23 (main: 4.24)
- Mean attempts: 1.76 (main: 2.00)
- Total cost: $1.4608 (main: $1.5492)

### `exp-2.10-tree-of-thoughts` — NEUTRAL

- Personas: 41/42 (main: 42/42)
- Groundedness: 0.992 (main: 0.999)
- Judge score: 4.23 (main: 4.24)
- Mean attempts: 1.73 (main: 2.00)
- Total cost: $1.3961 (main: $1.5492)

### `exp-1.15-edge-case-behavior-fields` — NEUTRAL

- Personas: 40/42 (main: 42/42)
- Groundedness: 0.999 (main: 0.999)
- Judge score: 4.22 (main: 4.24)
- Mean attempts: 1.88 (main: 2.00)
- Total cost: $1.5648 (main: $1.5492)

### `exp-1.17-length-budgets-per-field` — NEUTRAL

- Personas: 42/42 (main: 42/42)
- Groundedness: 0.994 (main: 0.999)
- Judge score: 4.21 (main: 4.24)
- Mean attempts: 1.95 (main: 2.00)
- Total cost: $1.5473 (main: $1.5492)

### `exp-3.20-confidence-weighted-corroboration` — NEUTRAL

- Personas: 40/42 (main: 42/42)
- Groundedness: 0.990 (main: 0.999)
- Judge score: 4.21 (main: 4.24)
- Mean attempts: 1.93 (main: 2.00)
- Total cost: $1.5370 (main: $1.5492)

### `exp-4.06` — NEUTRAL

- Personas: 40/42 (main: 42/42)
- Groundedness: 0.992 (main: 0.999)
- Judge score: 4.21 (main: 4.24)
- Mean attempts: 1.77 (main: 2.00)
- Total cost: $1.4388 (main: $1.5492)

### `exp-2.22-beam-search` — NEUTRAL

- Personas: 41/42 (main: 42/42)
- Groundedness: 0.998 (main: 0.999)
- Judge score: 4.21 (main: 4.24)
- Mean attempts: 1.85 (main: 2.00)
- Total cost: $1.4629 (main: $1.5492)

### `exp-4.11` — REJECT-RELIABILITY

- Personas: 38/42 (main: 42/42)
- Groundedness: 0.994 (main: 0.999)
- Judge score: 4.43 (main: 4.24)
- Mean attempts: 1.84 (main: 2.00)
- Total cost: $1.5376 (main: $1.5492)

### `exp-6.05-stability-across-reruns` — REJECT-RELIABILITY

- Personas: 38/42 (main: 42/42)
- Groundedness: 0.990 (main: 0.999)
- Judge score: 4.39 (main: 4.24)
- Mean attempts: 1.79 (main: 2.00)
- Total cost: $1.5110 (main: $1.5492)

### `exp-4.21-curiosity-behavior` — REJECT-RELIABILITY

- Personas: 38/42 (main: 42/42)
- Groundedness: 0.989 (main: 0.999)
- Judge score: 4.34 (main: 4.24)
- Mean attempts: 1.87 (main: 2.00)
- Total cost: $1.5373 (main: $1.5492)

### `exp-5.04-position-verbosity-bias` — REJECT-RELIABILITY

- Personas: 39/42 (main: 42/42)
- Groundedness: 0.993 (main: 0.999)
- Judge score: 4.32 (main: 4.24)
- Mean attempts: 1.82 (main: 2.00)
- Total cost: $1.4993 (main: $1.5492)

### `exp-2.12-self-consistency-voting` — REJECT-RELIABILITY

- Personas: 39/42 (main: 42/42)
- Groundedness: 0.996 (main: 0.999)
- Judge score: 4.24 (main: 4.24)
- Mean attempts: 1.90 (main: 2.00)
- Total cost: $1.5479 (main: $1.5492)

### `exp-3.06` — REJECT-RELIABILITY

- Personas: 38/42 (main: 42/42)
- Groundedness: 0.996 (main: 0.999)
- Judge score: 4.23 (main: 4.24)
- Mean attempts: 2.00 (main: 2.00)
- Total cost: $1.6176 (main: $1.5492)

### `exp-2.08-synthetic-warmstart` — REJECT-RELIABILITY

- Personas: 39/42 (main: 42/42)
- Groundedness: 0.988 (main: 0.999)
- Judge score: 4.22 (main: 4.24)
- Mean attempts: 1.90 (main: 2.00)
- Total cost: $1.5338 (main: $1.5492)

### `exp-5.14` — REJECT-RELIABILITY

- Personas: 39/42 (main: 42/42)
- Groundedness: 0.990 (main: 0.999)
- Judge score: 4.21 (main: 4.24)
- Mean attempts: 1.72 (main: 2.00)
- Total cost: $1.4330 (main: $1.5492)

### `exp-5.13` — REJECT-RELIABILITY

- Personas: 38/42 (main: 42/42)
- Groundedness: 0.986 (main: 0.999)
- Judge score: 4.18 (main: 4.24)
- Mean attempts: 1.76 (main: 2.00)
- Total cost: $1.4814 (main: $1.5492)

### `exp-4.13-length-matching` — REJECT-RELIABILITY

- Personas: 39/42 (main: 42/42)
- Groundedness: 0.992 (main: 0.999)
- Judge score: 4.14 (main: 4.24)
- Mean attempts: 1.82 (main: 2.00)
- Total cost: $1.4979 (main: $1.5492)

### `exp-3.03-retrieval-augmented-synthesis` — REJECT

- Personas: 36/42 (main: 42/42)
- Groundedness: 0.988 (main: 0.999)
- Judge score: 4.35 (main: 4.24)
- Mean attempts: 1.83 (main: 2.00)
- Total cost: $1.5751 (main: $1.5492)

### `exp-2.17-system-vs-user-prompt` — REJECT

- Personas: 37/42 (main: 42/42)
- Groundedness: 0.994 (main: 0.999)
- Judge score: 4.35 (main: 4.24)
- Mean attempts: 1.81 (main: 2.00)
- Total cost: $1.5103 (main: $1.5492)

### `exp-3.17-evidence-ablation` — REJECT

- Personas: 37/42 (main: 42/42)
- Groundedness: 0.993 (main: 0.999)
- Judge score: 4.35 (main: 4.24)
- Mean attempts: 1.73 (main: 2.00)
- Total cost: $1.4545 (main: $1.5492)

### `exp-4.23-persona-wake-words` — REJECT

- Personas: 37/42 (main: 42/42)
- Groundedness: 0.982 (main: 0.999)
- Judge score: 4.34 (main: 4.24)
- Mean attempts: 1.68 (main: 2.00)
- Total cost: $1.4461 (main: $1.5492)

### `exp-5.11-reference-based-vs-reference-free` — REJECT

- Personas: 37/42 (main: 42/42)
- Groundedness: 0.995 (main: 0.999)
- Judge score: 4.31 (main: 4.24)
- Mean attempts: 1.84 (main: 2.00)
- Total cost: $1.5656 (main: $1.5492)

### `exp-4.15-cold-start-warmup` — REJECT

- Personas: 37/42 (main: 42/42)
- Groundedness: 0.989 (main: 0.999)
- Judge score: 4.29 (main: 4.24)
- Mean attempts: 2.00 (main: 2.00)
- Total cost: $1.6491 (main: $1.5492)

### `exp-1.14-belief-value-separation` — REJECT

- Personas: 25/42 (main: 42/42)
- Groundedness: 0.988 (main: 0.999)
- Judge score: 4.29 (main: 4.24)
- Mean attempts: 1.96 (main: 2.00)
- Total cost: $2.2499 (main: $1.5492)

### `exp-3.14-negative-evidence` — REJECT

- Personas: 37/42 (main: 42/42)
- Groundedness: 0.996 (main: 0.999)
- Judge score: 4.29 (main: 4.24)
- Mean attempts: 1.97 (main: 2.00)
- Total cost: $1.6500 (main: $1.5492)

### `exp-1.24-stylometric-anchors` — REJECT

- Personas: 36/42 (main: 42/42)
- Groundedness: 0.989 (main: 0.999)
- Judge score: 4.27 (main: 4.24)
- Mean attempts: 1.92 (main: 2.00)
- Total cost: $1.6143 (main: $1.5492)

### `exp-1.23-internalized-contradictions` — REJECT

- Personas: 36/42 (main: 42/42)
- Groundedness: 0.986 (main: 0.999)
- Judge score: 4.08 (main: 4.24)
- Mean attempts: 1.97 (main: 2.00)
- Total cost: $1.8021 (main: $1.5492)

### `exp-3.13-temporal-grounding` — REJECT-BROKEN

- Personas: 0/42 (main: 42/42)
- Groundedness: 0.000 (main: 0.999)
- Mean attempts: 0.00 (main: 2.00)
- Total cost: $2.6418 (main: $1.5492)
