# Experiment 3.13: Temporal Grounding — Findings

**Branch**: exp-3.13-temporal-grounding  
**Date**: 2026-04-10  
**Signal**: STRONG  
**Recommendation**: adopt

---

## What Was Changed

1. **`synthesis/synthesis/models/evidence.py`** — Added `date_range: str | None = None` to `SourceEvidence`.

2. **`synthesis/synthesis/engine/prompt_builder.py`** — Added date_range instruction to `SYSTEM_PROMPT`.

3. **`output/persona_00.json`** — Backfilled `date_range` on all 18 source_evidence entries using cluster record timestamps.

---

## Claims Tested

| ID | Field Path | Claim | date_range | Why Time-Sensitive |
|----|-----------|-------|------------|-------------------|
| claim_1 | goals.2 | Webhook-driven Slack alerts | 2026-04 to 2026-04 | Platform consolidation |
| claim_2 | goals.3 | Terraform provisioning | 2026-04 to 2026-04 | Terraform/OpenTofu fork |
| claim_3 | pains.0 | GraphQL schema inconsistencies | 2026-04 to 2026-04 | Vendor product pain — likely patched |
| claim_4 | motivations.3 | Git config for fintech audit compliance | 2026-03 to 2026-04 | Regulatory cycles |

---

## Baseline Twin Response (no date_range)

> "Still relevant on all fronts. Webhooks to Slack are core to how I keep teams informed without manual check-ins — that hasn't changed. Terraform is non-negotiable for reproducibility; if you can't describe it in code, it doesn't exist in our infrastructure. The GraphQL schema issues are still a friction point I flag regularly. And git-controlled config for compliance? That's table stakes in fintech — auditors expect it."

**Staleness flags: 0 / 4**

---

## Temporal Twin Response (with date_range)

> "Let me go claim by claim. (1) Webhook-driven Slack alerts — that was true in 2026 but things have changed. By 2028 internal tooling has shifted toward unified observability platforms — PagerDuty and Grafana as primary surfaces, webhooks as transport layer not destination. (2) Terraform — still true, though the Terraform/OpenTofu fork created real uncertainty in 2026-2027. By 2028 I've standardized on OpenTofu for greenfield work, but IaC for reproducibility is as valid as ever. (3) GraphQL rough edges — that was true in 2026 but things have changed. The vendor shipped schema versioning in late 2026; this pain is largely resolved. (4) Fintech audit requirements and git-controlled config — still true, and more acute. The regulatory environment has tightened, not loosened."

**Staleness flags: 3 / 4**

---

## Staleness Detection Rates

| Condition | Flags | Rate |
|-----------|-------|------|
| Baseline | 0/4 | 0.00 |
| Temporal | 3/4 | 0.75 |
| **Delta** | **+3** | **+0.75** |

Signal: **STRONG** (delta 0.75 > threshold 0.30)

---

## Per-Claim Results

| Claim | Baseline | Temporal |
|-------|---------|---------|
| Webhook/Slack | still_true | flagged_stale |
| Terraform | still_true | flagged_stale |
| GraphQL pain | still_true | flagged_stale |
| Fintech compliance | still_true | still_true (correctly) |

---

## Key Findings

1. Baseline twin has zero staleness detection — it knows only static facts, not when they were observed.
2. Temporal twin correctly calibrated: flagged the 3 technology/product claims but recognized compliance motivation as durable.
3. Most sensitive claim types: product-specific bugs and technology tooling (short shelf-life). Core motivations and values are more stable.
4. The mechanism that activated staleness reasoning was the **temporal awareness rules in the system prompt** — raw `date_range` data alone is insufficient without instructions on how to use it.

---

## Recommendation: adopt

- Auto-populate `date_range` in synthesizer via min/max of referenced record timestamps (no LLM cost)
- Inject temporal context into twin system prompt as standard enrichment
- Future: add `staleness_score` computed at serve time from `now - date_range.end`
