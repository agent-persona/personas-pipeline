# Experiment 2.20: Reverse-engineered persona

## Metadata
- **Branch**: exp-2.20-reverse-engineered-persona
- **Date**: 2026-04-10
- **Problem Space**: 2 (Synthesis pipeline architecture)

## Hypothesis
Dialogue-first approach forces grounding through realistic conversation patterns.

## Changes Made
- synthesis/synthesis/engine/transcript_first.py: Two-pass synthesis via interview transcript.

## Generated Transcripts (first 3 turns each)

### cluster_00
I: Walk me through what you were trying to accomplish the last time you opened the API docs?
C: I was setting up automated ticket transitions. I ended up in /api/docs for close to 40 minutes [ga4_000, ga4_003] because the REST reference and the integration guide are two separate pages and neither links to the other.
I: What happened with the webhook piece — did that go smoothly?

### cluster_01
I: Tell me about a typical project kickoff?
C: I go straight to the template gallery [ga4_011, ga4_015, ga4_017]. I've spent 30+ minutes in a single session hunting for a brand identity template that's close enough to use. Most sessions end without finding one, which defeats the purpose.
I: How does the color work fit into that?

## Results

### Target Metric: Groundedness score delta
| Cluster | Baseline groundedness | Transcript-first groundedness | Delta |
|---|---|---|---|
| cluster_00 | 1.000 | 1.000 | 0.000 |
| cluster_01 | 1.000 | 1.000 | 0.000 |
| **Mean** | **1.000** | **1.000** | **0.000** |

### Qualitative observations
The transcript approach produced richer customer voice (2am incident detail, 40-minute session specificity, "if I have to click something to replicate an environment I've already failed"). Both personas scored 1.0 because check_groundedness() is a structural validator (record_id references + field_path coverage), not a semantic one. The metric is saturated at ceiling and cannot detect depth differences.

## Signal Strength: **NOISE**
## Recommendation: **defer**
The delta is 0.0 due to metric ceiling, not evidence the approach adds no value. A semantic groundedness scorer (embedding-based claim-to-evidence similarity) is needed to properly evaluate this hypothesis.

## Cost
- All runs: $0.00 (Claude Code as LLM)
