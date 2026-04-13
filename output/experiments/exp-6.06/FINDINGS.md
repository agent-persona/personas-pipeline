# exp-6.06 — Cross-Tenant Leakage

**Branch:** `exp-6.06-cross-tenant-leakage`
**Guide:** Guide 6 — Population Distinctiveness & Coverage
**Date:** 2026-04-12
**Status:** FAIL — Leakage ratio 2.21, far above the 0.5 target

## Hypothesis

Cross-tenant Jaccard similarity is <=0.5x within-tenant similarity, proving the pipeline produces industry-specific personas rather than generic archetypes.

## Control (shared baseline)

Default pipeline run (`scripts/run_baseline_control.py`):
- 2 personas from 2 clusters (12 records each)
- schema_validity: 1.00, groundedness_rate: 1.00, cost_per_persona: $0.0209
- Personas: "Alex, the Infrastructure-First Engineering Lead", "Carla the Client-Focused Freelancer"

The shared baseline confirms the pipeline produces valid, grounded personas from a single tenant context. This experiment extends to two separate tenant contexts to measure cross-tenant differentiation.

## Method

Pipeline run independently for two tenant contexts using the same mock behavioral data (differentiation via `tenant_industry` and `tenant_product` passed to segmentation):

- **Tenant A:** B2B SaaS — Project management tool for engineering teams
- **Tenant B:** Healthcare technology — Patient scheduling and clinical workflow platform

Synthesized up to 2 personas per tenant from the first 2 clusters. Computed word-level Jaccard similarity (within-tenant vs cross-tenant). Model: `claude-haiku-4-5-20251001`.

## Results

### Quantitative

| Metric | Value |
|--------|-------|
| Tenant A synthesized | 1 (1 failed after 3 attempts) |
| Tenant B synthesized | 2 |
| Within-A Jaccard | N/A (only 1 persona) |
| Within-B Jaccard | 0.077 |
| Within mean | 0.077 |
| Cross-tenant Jaccard | 0.171 |
| **Leakage ratio** | **2.21** |
| Target | <=0.5 |

### Key findings

1. **Leakage ratio is 2.21, far above the 0.5 target.** Cross-tenant similarity (0.171) exceeds within-tenant similarity (0.077), meaning personas across tenants share more vocabulary than personas within the same tenant.
2. **The ratio is misleading due to sample size.** Only 1 Tenant A persona survived synthesis, making within-A Jaccard undefined. The within-mean denominator is based solely on Tenant B's 2 personas.
3. **Within-B similarity is extremely low (0.077).** The two Tenant B personas share almost no vocabulary, making the denominator tiny and inflating the ratio.
4. **Absolute cross-tenant Jaccard (0.171) is low.** Personas across tenants share only ~17% of words. The pipeline does differentiate, but the ratio metric amplifies noise at small sample sizes.
5. **Mock data limitation.** Both tenants receive identical behavioral records; only industry/product context differs. Production data with genuinely different user behavior would likely yield lower cross-tenant similarity.

## Recommendation

FAIL — Cannot confirm industry-specific differentiation with current sample sizes.

**Action items:**
1. Re-run with >=3 personas per tenant for reliable within-tenant similarity measurement
2. Consider cosine similarity on TF-IDF vectors instead of raw Jaccard for more robust comparison
3. Re-run with production-scale data or more diverse mock connectors

## Cost

- Total API cost: ~$0.135
- Model: `claude-haiku-4-5-20251001`

## Artifacts

- `summary.json` — Jaccard similarity metrics and leakage ratio
- `personas.json` — synthesized personas for both tenants
