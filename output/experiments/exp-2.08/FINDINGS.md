# exp-2.08 — Synthetic-Data Warmstart

**Branch:** `exp-2.08-synthetic-warmstart`
**Guide:** Guide 2 — Synthesis Pipeline Architecture
**Date:** 2026-04-12
**Status:** WEAK — warmstart did not improve sparse-tenant quality; marginal groundedness degradation observed.

## Hypothesis

Prepending 3 synthetic prior personas (from unrelated industries) as context in the system prompt will improve sparse-tenant persona quality by ≥0.5 judge points on depth, without degrading dense-tenant groundedness below 0.9 (on a 1-5 judge scale normalized to 0-1).

## Method

Four conditions in a 2×2 matrix:

|  | Baseline (standard prompt) | Treatment (warmstart prompt) |
|---|---|---|
| **Sparse** (5 records) | sparse_baseline | sparse_treatment |
| **Dense** (full cluster) | dense_baseline | dense_treatment |

Sparse cluster is created by subsampling the first 5 records from the largest cluster. Treatment prepends 3 hand-crafted synthetic prior personas (ops lead, growth marketer, engineering manager) from unrelated B2B SaaS verticals.

Claude-as-judge (Haiku 4.5) rates each synthesized persona on:
- **Groundedness** (1-5): claim traceability to source records
- **Depth** (1-5): specificity and actionability of goals/pains/vocabulary

Backend: Haiku 4.5. 4 synthesis calls + 4 judge calls.

## Results

### Quantitative

| Condition | Status | Groundedness (judge) | Depth (judge) |
|---|---|---|---|
| sparse_baseline | ok | **4** | **4** |
| sparse_treatment | ok | **3** | **4** |
| dense_baseline | ok | **4** | **4** |
| dense_treatment | ok | **4** | **4** |

| Delta | Value | Hypothesis threshold |
|---|---|---|
| Δ sparse depth | **0.0** | needed ≥+0.5 |
| Δ sparse groundedness | **−1.0** | should not degrade |
| Δ dense groundedness | **0.0** | should stay ≥0.9 (normalized) |
| Δ dense depth | **0.0** | should not degrade |

### Interpretation

**Hypothesis not supported.** The warmstart treatment:

1. **Did NOT improve sparse-tenant depth** (both scored 4/5). The model produced comparably specific personas regardless of whether synthetic priors were present.
2. **Degraded sparse-tenant groundedness** by 1 point (4→3). The synthetic priors may have encouraged the model to generate claims inspired by the prior archetypes rather than strictly from the sparse source records.
3. **Had no effect on dense-tenant quality.** With sufficient data, the warmstart context was effectively ignored — the model had enough signal from the cluster itself.

### Why the hypothesis failed

1. **Haiku 4.5 is already good at sparse synthesis.** Even with only 5 records, the model produces depth-4 personas. The ceiling effect means there's little room for warmstart to improve.
2. **Priors introduce a hallucination vector.** The "Ops Lead" and "Growth Marketer" priors, while from different industries, share structural patterns (ROI-focused goals, tool evaluation pains) that could leak into the synthesized persona — explaining the groundedness drop.
3. **n=1 per condition.** Judge ratings are integer-valued 1-5 with high variance. A 1-point difference could easily be noise.

## Recommendation

**DEFER** warmstart. The mechanism is sound (priors load correctly, prompt renders them, synthesis completes), but the expected benefit didn't materialize. The groundedness regression on sparse data is concerning.

**Re-test when:**
- Using ≥3 trials per condition so judge variance is measurable
- Testing on genuinely ultra-sparse data (1-2 records) where baseline quality is low enough to show improvement
- Using a more distant prior set (non-SaaS archetypes) to reduce hallucination leakage

## Cost

- Total API cost: ~$0.30 (4 synthesis calls + 4 judge calls on Haiku 4.5)
- Model: `claude-haiku-4-5-20251001`

## Artifacts

- `sparse_baseline.json`, `sparse_treatment.json` — per-condition results with judge scores
- `dense_baseline.json`, `dense_treatment.json` — per-condition results with judge scores
- `summary.json` — full metric dump with deltas
