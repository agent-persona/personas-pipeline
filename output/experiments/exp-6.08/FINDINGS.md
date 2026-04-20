# exp-6.08 — Long-tail Persona Viability

**Branch:** `exp-6.08-long-tail-viability`
**Guide:** Guide 6 — Population Distinctiveness & Coverage
**Date:** 2026-04-12
**Status:** INCONCLUSIVE — Synthesis failures at n=5 and n=7 prevent full curve; no quality degradation in successful runs

## Hypothesis

Quality collapses below ~10 records (groundedness < 0.7, depth < 3.0). At >=25 records, quality is within 0.1 of full-cluster score.

## Control (shared baseline)

Default pipeline run (`scripts/run_baseline_control.py`):
- 2 personas from 2 clusters (12 records each)
- schema_validity: 1.00, groundedness_rate: 1.00, cost_per_persona: $0.0209
- Personas: "Alex, the Infrastructure-First Engineering Lead", "Carla the Client-Focused Freelancer"

The full cluster (n=12) serves as both the shared baseline quality reference and the experiment's own control. Subsampled clusters at sizes [3, 5, 7, 10] are the treatment conditions. Metrics from `evaluation/metrics.py`: schema_validity, groundedness_rate.

## Method

Took the largest cluster (12 records) and created random subsamples at sizes [3, 5, 7, 10, 12(full)]. Synthesized a persona from each subsample (max 3 retries). Scored each on structural groundedness (0-1) and Claude-as-judge depth (1-5). Used `random.Random(42)` for reproducible subsampling. Model: `claude-haiku-4-5-20251001`.

## Results

### Quantitative

| Records | Status | Groundedness | Depth | Persona |
|---------|--------|-------------|-------|---------|
| 3       | OK     | 1.00        | 5     | Alex -- The Infrastructure-First Engineering Lead |
| 5       | FAILED | --          | --    | (synthesis failed after 3 attempts) |
| 7       | FAILED | --          | --    | (synthesis failed after 3 attempts) |
| 10      | OK     | 1.00        | 5     | DevOps-First Engineer |
| 12 (full) | OK   | 1.00        | 5     | Alex Chen, Infrastructure-First DevOps Engineer |

### Key findings

1. **No quality degradation detected in successful runs.** All personas that passed synthesis scored groundedness=1.00 and depth=5, regardless of cluster size.
2. **Synthesis failures at n=5 and n=7** are the most notable finding. Both exhausted 3 retries due to groundedness violations (scores 0.29-0.89). This suggests a "middle ground" instability: enough records to create complex claims but not enough to ground them all.
3. **n=3 succeeded with perfect scores.** Counterintuitively, fewer records constrain the LLM to make only supportable claims, yielding high groundedness.
4. **Quality curve is non-monotonic.** Very small clusters (3) and larger clusters (10+) succeed, while middle-range clusters (5-7) fail more often.
5. **Cannot test >=25 convergence threshold.** Mock data has only 12 max records.

## Recommendation

INCONCLUSIVE — The hypothesis cannot be confirmed or rejected. The failure mode is synthesis instability at medium sizes, not gradual quality degradation.

**Action items:**
1. Increase retry budget (max_retries) for clusters with 5-7 records
2. Re-run with production data that has larger clusters to test the full size range
3. Investigate the groundedness violation pattern at n=5-7 to improve synthesis prompts

## Cost

- Total API cost: ~$0.21
- Model: `claude-haiku-4-5-20251001`

## Artifacts

- `summary.json` — per-size status, groundedness, and depth scores
- `personas.json` — successfully synthesized personas
