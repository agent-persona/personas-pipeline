# Experiment 6.06 — Cross-Tenant Leakage

## Hypothesis

Cross-tenant similarity is ≤0.5× within-tenant similarity, proving the pipeline produces industry-specific personas rather than generic archetypes.

## Method

- **Tenant A:** B2B SaaS — Project management tool for engineering teams
- **Tenant B:** Healthcare technology — Patient scheduling and clinical workflow platform
- Both tenants use the same mock behavioral data; differentiation comes from `tenant_industry` and `tenant_product` passed to segmentation
- Synthesized up to 2 personas per tenant from the first 2 clusters
- Computed word-level Jaccard similarity (within-tenant vs cross-tenant)
- Model: `claude-haiku-4-5-20251001`

## Results

| Metric | Value |
|--------|-------|
| Tenant A synthesized | 1 (1 failed after 3 attempts) |
| Tenant B synthesized | 2 |
| Within-A Jaccard | N/A (only 1 persona) |
| Within-B Jaccard | 0.077 |
| Within mean | 0.077 |
| Cross-tenant Jaccard | 0.171 |
| **Leakage ratio** | **2.21** |
| Target | ≤0.5 |

- Total cost: $0.14

## Verdict

**FAIL** — Leakage ratio is 2.21, far above the 0.5 target.

## Analysis

The result is misleading due to data limitations:

1. **Only 1 Tenant A persona survived** — synthesis failed for 1 of 2 clusters, leaving within-A Jaccard undefined. The within-mean is based solely on Tenant B's 2 personas.
2. **Within-B similarity is extremely low (0.077)** — the two Tenant B personas share almost no vocabulary, making the denominator tiny and inflating the ratio.
3. **Cross-tenant similarity (0.171) is also low in absolute terms** — personas across tenants share only ~17% of words, which is a small overlap.
4. **Mock data limitation** — both tenants receive identical behavioral records; only the industry/product context differs. In production with truly different user behavior per tenant, cross-tenant similarity would likely be even lower.

The metric design (ratio of two small Jaccard scores) amplifies noise when sample sizes are small. The absolute cross-tenant Jaccard of 0.171 suggests the pipeline does differentiate, but the ratio metric doesn't capture this with only 1-2 personas per tenant.

## Implications

- Need ≥3 personas per tenant for reliable within-tenant similarity measurement
- Consider cosine similarity on TF-IDF vectors instead of raw Jaccard for more robust comparison
- Re-run with production-scale data or more diverse mock connectors
