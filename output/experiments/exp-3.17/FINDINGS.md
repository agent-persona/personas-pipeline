# exp-3.17 — Evidence Ablation

**Branch:** `exp-3.17-evidence-ablation`
**Guide:** Guide 3 — Evidence & Grounding
**Date:** 2026-04-12
**Status:** STRONG — judge groundedness drops steeply with ablation (4→1), but structural groundedness paradoxically increases. Key insight: the two metrics measure fundamentally different things.

## Hypothesis

Removing the top 10% most informative records causes ≥0.3 structural groundedness drop and ≥1.0 depth drop.

## Method

Take the largest cluster (12 records from `tenant_acme_corp`). Rank records by informativeness (payload key count × text length). Create 5 ablated variants by removing the top 0%, 10%, 25%, 50%, 75% most informative records.

Synthesize a persona from each. Score on:
1. **Structural groundedness** (0-1): built-in `check_groundedness()` — validates record IDs and evidence coverage
2. **Judge groundedness** (1-5): Claude-as-judge rates claim traceability
3. **Judge depth** (1-5): Claude-as-judge rates specificity/actionability

Backend: Haiku 4.5. 5 synthesis calls + 4 judge calls (one synthesis failed).

## Results

### Quantitative

| Ablation % | Records | Status | Structural G | Judge G | Judge Depth |
|---|---|---|---|---|---|
| 0% | 12 | **FAILED** | — | — | — |
| 10% | 11 | ok | 0.95 | **4** | **5** |
| 25% | 9 | ok | 0.90 | **4** | **4** |
| 50% | 6 | ok | 1.00 | **2** | **4** |
| 75% | 3 | ok | 1.00 | **1** | **4** |

### Key findings

1. **0% ablation (full cluster) failed synthesis entirely.** The full 12-record cluster caused 3 consecutive groundedness failures. The most informative records contain complex payloads that generate too many claims for the model to properly evidence.

2. **Structural groundedness INCREASES with ablation** (0.95→1.00). This is because fewer records mean fewer claims, and fewer claims mean easier evidence coverage. The structural checker only verifies that stated evidence maps to real record IDs — it doesn't verify claim quality.

3. **Judge groundedness DROPS steeply** (4→1). The judge recognizes that with only 3 records, the persona is making claims that can't possibly be grounded in such thin data. This is the real signal.

4. **Judge depth is remarkably stable** (5→4→4→4). The model produces comparably specific personas regardless of data volume — it just makes them up when data is sparse. This confirms depth ≠ groundedness.

### The structural/judge divergence

This is the most important finding of the experiment. The built-in `check_groundedness()` measures **formal evidence validity** (are record IDs real? do all fields have evidence entries?). The judge measures **semantic groundedness** (are the claims actually supported by the data?). These diverge dramatically under ablation:

- At 75% ablation with 3 records, the model produces a fully valid evidence chain (structural score 1.0) pointing to those 3 records — but the claims are largely fabricated from the model's priors, not from the data. The judge correctly scores this as 1/5.

## Recommendation

**STRONG** result — not on the original hypothesis (which is untestable because 0% failed), but on the structural/judge divergence.

**Action items:**
1. The structural groundedness checker needs augmentation. It catches invalid record IDs but not semantic fabrication. Consider adding a lightweight LLM verification step.
2. The 0% synthesis failure suggests the groundedness threshold (0.90) is too aggressive for information-rich clusters. Consider adaptive thresholds based on cluster complexity.
3. Depth scores should never be used alone as a quality signal — they reward plausible-sounding output regardless of grounding.

## Cost

- Total API cost: ~$0.25 (5 synthesis attempts + 4 judge calls on Haiku 4.5)
- Model: `claude-haiku-4-5-20251001`

## Artifacts

- `ablated_personas.json` — 5 ablation-level results with personas and judge scores
- `summary.json` — ablation curve data
