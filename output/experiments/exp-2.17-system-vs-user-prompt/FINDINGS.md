# Experiment 2.17: System vs user prompt placement

## Metadata
- **Branch**: exp-2.17-system-vs-user-prompt
- **Date**: 2026-04-09
- **Problem Space**: 2

## Hypothesis
Restructuring prompt layers optimizes output quality

## Layouts Tested
| Layout | System Prompt | User Message |
|---|---|---|
| A (baseline) | Full (preamble + quality criteria + evidence rules + example) | Cluster data only |
| B | Role only ("You are a persona synthesis expert.") | Quality criteria + evidence rules + example + cluster data |
| C | Preamble + quality criteria | Cluster data + inline evidence reminder |

## Results
| Layout | Mean Groundedness | Schema Validity | Delta vs A |
|---|---|---|---|
| A (baseline) | 1.00 | 1.00 | — |
| B | 1.00 | 1.00 | 0.00 |
| C | 1.00 | 1.00 | 0.00 |

## Signal Strength: **NOISE**
## Recommendation: **defer**

All three prompt layouts produced perfect groundedness (1.00) on both test clusters. The quality delta across all layouts is 0.0, well below the 2% threshold for WEAK signal. The baseline Layout A is already performing at ceiling.

**Why the signal is flat**: The groundedness metric is a structural check (do source_evidence entries reference valid record IDs, and does every required field have a corresponding evidence entry). Any carefully-written persona following the instructions will pass this check regardless of whether those instructions live in the system prompt or user message. The instructions content is identical across layouts — only placement differs — and placement does not affect the structural validity of the output.

**To get a real signal from this experiment**, two approaches would work:
1. Use a harder dataset where at least one layout causes a measurable drop
2. Introduce a semantic quality metric (distinctiveness, actionability scores) beyond structural groundedness

## Cost
- All runs: $0.00

## Files
- `baseline/` — Layout A personas for both clusters
- `comparison.json` — machine-readable results
- `FINDINGS.md` — this file
