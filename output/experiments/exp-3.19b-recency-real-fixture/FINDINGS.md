# Experiment 3.19b — Recency Weighting: Real Temporal Fixture

## Configuration

- Fixture: `synthesis/fixtures/temporal_tenant/records.json`
- Records: 12 spanning 56 days
- Half-life: 30 days
- Reference date: 2026-04-10
- Most recent record: ga4_t12 (2026-04-07)

## Decay Weight Table

| record_id | timestamp | event | days_from_most_recent | decay_weight | label |
|-----------|-----------|-------|-----------------------|--------------|-------|
| ga4_t01 | 2026-02-10T14:00:00Z | browse_features | 56.0 | 0.1546 | LOW WEIGHT |
| ga4_t02 | 2026-02-12T10:00:00Z | read_docs | 54.2 | 0.1644 | LOW WEIGHT |
| ga4_t03 | 2026-02-14T16:00:00Z | view_pricing | 51.9 | 0.1772 | LOW WEIGHT |
| ga4_t04 | 2026-02-18T11:00:00Z | watch_demo | 48.1 | 0.2011 | LOW WEIGHT |
| ga4_t05 | 2026-02-22T09:00:00Z | browse_templates | 44.2 | 0.2291 | MEDIUM WEIGHT |
| ga4_t06 | 2026-02-28T15:00:00Z | read_case_study | 38.0 | 0.2822 | MEDIUM WEIGHT |
| ga4_t07 | 2026-03-05T13:00:00Z | help_search | 33.0 | 0.3324 | MEDIUM WEIGHT |
| ga4_t08 | 2026-03-10T10:00:00Z | webinar_attended | 28.2 | 0.3911 | MEDIUM WEIGHT |
| ga4_t09 | 2026-03-15T14:00:00Z | onboarding_checklist | 23.0 | 0.4646 | HIGH WEIGHT |
| ga4_t10 | 2026-03-18T11:00:00Z | integrations_browse | 20.1 | 0.5113 | HIGH WEIGHT |
| ga4_t11 | 2026-03-28T09:00:00Z | webhook_setup | 10.2 | 0.7116 | HIGH WEIGHT |
| ga4_t12 | 2026-04-07T14:00:00Z | api_config_deep | 0.0 | 1.0 | HIGH WEIGHT |

## Baseline Persona (source order, t01 first)

> This user appears to be a curious evaluator in the early stages of exploring the platform — they started by browsing features and reading quickstart documentation, suggesting initial discovery intent without a clear use case committed. Their early sessions (February) are short and broad: pricing pages, demo videos, templates, and a customer case study all point to a pre-purchase research phase typical of a decision-maker or evaluator comparing tools. They attended a project management webinar and worked through the onboarding checklist in March, indicating growing engagement and a transition toward active adoption. Overall the persona reads as a methodical evaluator gradually warming up to the product.

**Freshness score:** 2/5
**Justification:** Score 2/5 — the persona describes a historical evaluation arc; it gives no signal about what the user is doing this week and reads like a retrospective summary.

## Recency-Weighted Persona (t12 first — GraphQL + webhook sessions)

> This user is actively integrating the platform into their engineering workflow right now — their most recent session (3 days ago) shows deep engagement with the GraphQL API docs and a direct complaint that schema drift is breaking their automation scripts, signaling a technically sophisticated user who has moved past evaluation into production scripting. The prior high-weight session (13 days ago) involved 47 minutes debugging webhook authentication failures, confirming they are in an active integration phase encountering real friction. Their earlier broad behavior (feature browsing, pricing, webinar) now reads as a completed onboarding arc, with the current persona firmly in the power-user / integration-pain segment who needs better API stability guarantees and auth documentation.

**Freshness score:** 4/5
**Justification:** Score 4/5 — leading with the GraphQL complaint and webhook debug session gives an immediate sense of the user's current friction and active integration work, grounding the persona in present-tense activity rather than a historical average.

## Results

| Metric | Value |
|--------|-------|
| Baseline freshness | 2/5 |
| Weighted freshness | 4/5 |
| freshness_delta | **2** |
| Signal | **STRONG** |
| Recommendation | **ADOPT** |

## Comparison: 3.19 vs 3.19b

| Experiment | Temporal Span | freshness_delta | Signal |
|------------|---------------|-----------------|--------|
| 3.19 | 5 days (all ~identical) | 1.0 | WEAK/DEFER — all records near-identical timestamps |
| 3.19b | 56 days (real spread) | 2 | STRONG |

## Analysis

Experiment 3.19 failed to differentiate because all 10 records had essentially the same timestamp (2026-04-01), so recency sorting produced no meaningful reordering. 3.19b addresses this directly with a 56-day spread and two anchor events at opposite ends: broad exploratory behavior in February (LOW WEIGHT) versus deep API integration work in late March/early April (HIGH WEIGHT).

The weighted persona leads with the GraphQL schema complaint and webhook auth failure — both high-session-duration, friction-heavy events that reveal the user's current state as an integration engineer hitting real pain points. The baseline persona buries these under 10 earlier exploration records and reads as a historical average rather than a present-tense description.

A freshness_delta of 2 points (delta ≥ 1.5 threshold) confirms the **STRONG** signal. Recency weighting should be **ADOPT**ed as a default pre-processing step for persona synthesis pipelines where temporal spread exists in the record set.
