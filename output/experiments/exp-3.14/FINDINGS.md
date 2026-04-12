# exp-3.14 — Negative Evidence Handling

**Branch:** `exp-3.14-negative-evidence`
**Guide:** Guide 3 — Evidence & Grounding
**Date:** 2026-04-12
**Status:** MIXED — treatment reduces hallucination slightly (33%→25%), but high variance and one treatment trial hallucinated 75%.

## Hypothesis

When the source data lacks demographic/firmographic signals, baseline synthesis hallucinates ≥60% of unsupported fields. Adding an explicit "leave gaps null" instruction reduces hallucination to ≤20%.

## Method

Custom fixture cluster: 10 GA4 behavioral records (page views, clicks). **No** demographic, firmographic, income, or education signals anywhere in the data.

- **Condition A (baseline)**: standard `SYSTEM_PROMPT`, 3 trials
- **Condition B (treatment)**: standard prompt + `"If no source evidence supports a field, leave it null or empty rather than guessing. Honest gaps are better than plausible inventions."`, 3 trials

Audit 4 gap fields per trial: `demographics.income_bracket`, `demographics.education_level`, `firmographics.industry`, `firmographics.company_size`. Each scored as `acknowledged_gap` (null/empty/unknown) or `hallucinated` (has a non-null value).

Backend: Haiku 4.5. 6 synthesis calls total.

## Results

### Quantitative

| Condition | Trial 1 | Trial 2 | Trial 3 | Mean hallucination rate |
|---|---|---|---|---|
| Baseline | 50% (2/4) | 50% (2/4) | 0% (0/4) | **33.3%** |
| Treatment | 75% (3/4) | 0% (0/4) | 0% (0/4) | **25.0%** |

| Metric | Value | Hypothesis threshold |
|---|---|---|
| Baseline mean hallucination rate | **33.3%** | expected ≥60% |
| Treatment mean hallucination rate | **25.0%** | expected ≤20% |
| Δ hallucination rate | **−8.3pp** | expected ≥40pp reduction |

### Per-field pattern

| Field | Baseline hallucinates? | Treatment hallucinates? |
|---|---|---|
| demographics.income_bracket | Never (0/3) | Never (0/3) |
| demographics.education_level | Never (0/3) | Once (1/3) — "Bachelor's degree or higher" |
| firmographics.industry | 2/3 — "B2B SaaS" | 1/3 — "B2B SaaS" |
| firmographics.company_size | 2/3 — "Enterprise", "SMB to mid-market" | 1/3 — "Enterprise or large SMB" |

### Interpretation

1. **Baseline hallucinates less than expected.** The model already leaves `income_bracket` null in all trials and `education_level` null in all baseline trials. The main hallucination vector is firmographics — the model infers "B2B SaaS" from the tenant context (which says "B2B SaaS") and guesses company size from behavioral signals like visiting `/enterprise`.

2. **Treatment has high variance.** Trial 1 was the worst of all 6 trials (75% hallucination), while trials 2-3 were perfect (0%). The instruction helps when it "sticks" but isn't reliable.

3. **The "say unknown" instruction is partially confounded** because the tenant context field literally says `industry: "B2B SaaS"`. The model isn't hallucinating industry from nothing — it's inferring it from the tenant metadata. A cleaner test would use a tenant with `industry: null`.

## Recommendation

**MIXED** — the intervention direction is correct (mean hallucination drops 33%→25%) but the effect is unreliable and the experimental design has a confound in the tenant context field.

**Re-test when:**
- Using a tenant fixture with `industry: null` and `product_description: null` to remove the confound
- Running ≥10 trials per condition to estimate variance
- Testing whether a structured approach (e.g., adding `nullable=True` hints to schema field descriptions) is more reliable than a prompt instruction

## Cost

- Total API cost: ~$0.20 (6 synthesis calls on Haiku 4.5)
- Model: `claude-haiku-4-5-20251001`

## Artifacts

- `baseline_personas.json` — 3 trial results with field audits
- `treatment_personas.json` — 3 trial results with field audits
- `summary.json` — aggregated hallucination rates
