# Experiment 3.19: Source Weighting by Recency — Findings

## Setup

- **Cluster**: clust_1adb81b417c0 (cluster_00.json)
- **Tenant**: tenant_acme_corp, B2B SaaS, project management for engineering teams
- **Records in prompt**: 12 sample records (11 GA4, 1 HubSpot, 1 Intercom — from 19 total)
- **Decay half-life**: 30 days

---

## Record Timestamps

| Record ID      | Source    | Timestamp               | Decay Weight |
|----------------|-----------|-------------------------|-------------|
| intercom_000   | intercom  | 2026-04-02T10:00:00Z    | 1.0000      |
| ga4_000        | ga4       | 2026-04-01T12:00:00Z    | 0.9699      |
| ga4_001        | ga4       | 2026-04-01T12:00:00Z    | 0.9699      |
| ga4_002        | ga4       | 2026-04-01T12:00:00Z    | 0.9699      |
| ga4_003        | ga4       | 2026-04-01T12:00:00Z    | 0.9699      |
| ga4_004        | ga4       | 2026-04-01T12:00:00Z    | 0.9699      |
| ga4_005        | ga4       | 2026-04-01T12:00:00Z    | 0.9699      |
| ga4_006        | ga4       | 2026-04-01T12:00:00Z    | 0.9699      |
| ga4_007        | ga4       | 2026-04-01T12:00:00Z    | 0.9699      |
| ga4_008        | ga4       | 2026-04-01T12:00:00Z    | 0.9699      |
| ga4_009        | ga4       | 2026-04-01T12:00:00Z    | 0.9699      |
| hubspot_000    | hubspot   | 2026-03-28T09:00:00Z    | 0.8453      |

**Total span**: 5 days (2026-03-28 to 2026-04-02).

**Critical observation**: All 10 GA4 records share an identical timestamp (2026-04-01T12:00:00Z), likely a synthetic/batch-assigned timestamp. The only real temporal differentiation is:
- intercom_000 is 1 day newer than the GA4 batch
- hubspot_000 is 4 days older than the GA4 batch

This means HIGH/MEDIUM/LOW labels within the GA4 batch are assigned arbitrarily by list position, not by genuine recency signal.

---

## Baseline Synthesis (no weighting)

Records rendered in source order. The HubSpot record (hubspot_000, demographic anchor: Senior DevOps Engineer, fintech, 50-200 employees) appears second in the list. The Intercom GraphQL complaint (intercom_000, most specific and most recent signal) appears 8th of 12 records.

**Persona summary (baseline)**:
Jordan Chen, Senior DevOps Engineer at a 50-200 employee fintech company. Engaged in API setup (three sessions, longest 2340s), webhook configuration, GitHub integration, Terraform setup, and custom dashboards. Also triggered a team invite, suggesting active onboarding of colleagues. Recently flagged GraphQL schema quality as a pain point. The demographic background (fintech, DevOps) provides the identity anchor; behavioral signals fill out the picture. Goals: automate CI/CD pipelines, integrate with GitHub repos. Pains: GraphQL rough edges (verbatim complaint).

**Freshness score (baseline)**: 2/5
Rationale: Reads as a cumulative behavioral profile. The identity anchor is a CRM record from 5 days ago. The GraphQL complaint (freshest and most specific signal) is buried mid-list and weighted equally with older behavioral data. No sense of "what this person is doing right now."

---

## Recency-Weighted Synthesis (weighting on, half_life=30d)

Records sorted by decay weight. intercom_000 promotes to position 1 (HIGH WEIGHT, weight=1.0). hubspot_000 demotes to position 12 (LOW WEIGHT, weight=0.8453).

**Persona summary (weighted)**:
Jordan (role inferred from behavioral depth, not CRM record). Lead signal: direct API critique — "Your REST API is solid but the GraphQL endpoint has some rough edges." Dominant behavioral thread: api_setup is recurring and deep (sessions 1240-2340s), suggesting active instrumentation not exploration. Webhook and GitHub integrations active. Terraform setup confirms IaC-first mindset. The HubSpot firmographic data (Senior DevOps, fintech, 50-200) is treated as LOW WEIGHT background context.

Key difference from baseline: framing shifts from "who is this person professionally" (baseline leads with hubspot demographic) to "what are they doing and complaining about right now" (weighted leads with intercom feedback). The GraphQL pain point is the primary trait rather than a mid-list item.

**Freshness score (weighted)**: 3/5
Rationale: The most recent behavioral signal (GraphQL API critique) now dominates. The persona feels like it describes active, current engagement. Score capped at 3 because most records share identical timestamps, so recency ordering is largely artificial beyond intercom/hubspot repositioning.

---

## Freshness Scores

| Variant   | Score | Notes |
|-----------|-------|-------|
| Baseline  | 2/5   | Demographic anchor leads; freshest signal buried mid-list |
| Weighted  | 3/5   | GraphQL complaint promoted to position 1; hubspot deprioritized |
| Delta     | +1.0  |       |

---

## Signal Assessment

**Signal**: WEAK

freshness_delta = 1.0 (nominally STRONG threshold), but confounded by temporal uniformity. Only 2 of 12 records have meaningfully distinct timestamps. The 10 GA4 records share identical timestamps and receive arbitrarily assigned HIGH/MEDIUM/LOW labels by list position. The measured delta reflects promoting one high-signal record (intercom_000 with a direct quote) to top position — not a genuine recency-weighting effect across a temporally spread dataset.

To test the hypothesis properly, records must span weeks or months with real variance. With a 5-day spread and 10/12 records at the same timestamp, the feature is effectively untestable on this data.

---

## Did Weighting Change Trait Emphasis?

Yes, but narrowly:
- intercom_000 promotion: GraphQL critique moved from position 8 to position 1. This shifted framing from "DevOps professional who uses the tool" to "opinionated API consumer actively evaluating schema quality."
- hubspot_000 demotion: Demographic anchor moved to last position. Weighted persona infers role from behavioral depth rather than leading with CRM record.
- GA4 batch reordering: No meaningful effect. Identical timestamps, identical weights, same behavior types regardless of position.

---

## Recommendation

**Defer** — implementation is correct and direction is sound. Cannot be meaningfully evaluated on synthetic data with clustered timestamps. Valid test requires: (1) records spanning 30-60+ days, (2) at least 30% of records with meaningfully different timestamps. Pair with real GA4 export or generate synthetic data with deliberate temporal spread before adopting.

---

## Implementation

Changes to `synthesis/synthesis/engine/prompt_builder.py`:
- Added `_parse_timestamp()` for ISO 8601 parsing
- Added `sort_records_by_recency(records, decay_half_life_days=30)` returning `(record, weight, label)` tuples
- Updated `build_user_message()` with `use_recency_weighting: bool = False` and `decay_half_life_days: float = 30` kwargs
- When enabled, records sorted descending by decay weight, prefixed with [HIGH WEIGHT] / [MEDIUM WEIGHT] / [LOW WEIGHT] (30/40/30 split)
- Default behavior unchanged (backward compatible)
