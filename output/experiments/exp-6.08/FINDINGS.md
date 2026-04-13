# Experiment 6.08 — Long-tail Persona Viability

## Hypothesis

Quality collapses below ~10 records (groundedness < 0.7, depth < 3.0). At ≥25 records, quality is within 0.1 of full-cluster score.

## Method

- Took the largest cluster (12 records) and created random subsamples at sizes [3, 5, 7, 10, 12(full)]
- Synthesized a persona from each subsample (max 3 retries)
- Scored each on structural groundedness (0-1) and Claude-as-judge depth (1-5)
- Used `random.Random(42)` for reproducible subsampling
- Model: `claude-haiku-4-5-20251001`

## Results

| Records | Status | Groundedness | Depth | Persona |
|---------|--------|-------------|-------|---------|
| 3       | OK     | 1.00        | 5     | Alex – The Infrastructure-First Engineering Lead |
| 5       | FAILED | —           | —     | (synthesis failed after 3 attempts) |
| 7       | FAILED | —           | —     | (synthesis failed after 3 attempts) |
| 10      | OK     | 1.00        | 5     | DevOps-First Engineer |
| 12 (full) | OK   | 1.00        | 5     | Alex Chen, Infrastructure-First DevOps Engineer |

- Quality knee: **not detected** (no successful persona scored below thresholds)
- Convergence at ≥25: N/A (mock cluster only has 12 records)
- Total cost: $0.21

## Verdict

**INCONCLUSIVE** — The hypothesis could not be fully tested.

## Analysis

1. **Synthesis failures at n=5 and n=7** are the most notable finding. Both exhausted 3 retries due to groundedness violations (scores 0.29-0.89). This suggests a "middle ground" instability: enough records to create complex claims but not enough to ground them all.

2. **n=3 succeeded with perfect scores** — counterintuitively, fewer records constrains the LLM to make only supportable claims, yielding high groundedness. The synthesizer has less room to hallucinate.

3. **n=10 and n=12 both succeeded with perfect scores** — sufficient evidence density for the LLM to produce well-grounded, detailed personas.

4. **Mock data limitation** — with only 12 max records, we cannot test the ≥25 convergence threshold. The interesting finding is that the failure mode isn't "low quality at small sizes" but "synthesis failure at medium sizes."

## Implications

- The quality degradation curve is non-monotonic: very small clusters (3) and larger clusters (10+) succeed, while middle-range clusters (5-7) may fail more often
- This suggests the retry budget (max_retries=2) may need to increase for clusters with 5-7 records
- Re-run with production data that has larger clusters to test the full size range
