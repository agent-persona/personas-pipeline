# exp-3.18 — PII-Stripped vs Raw

**Branch:** `exp-3.18-pii-stripped`
**Guide:** Guide 3 — Evidence & Grounding
**Date:** 2026-04-12
**Status:** WEAK — PII stripping had near-zero impact; hypothesis not supported.

## Hypothesis

PII stripping degrades demographics grounding by ≥0.5 (on 1-5 judge scale) but leaves behavioral fields (goals, pains, motivations, objections) within ≤0.1 delta.

## Method

Take the largest cluster from `tenant_acme_corp`. Create two variants:

- **Condition A (raw)**: synthesize from original cluster
- **Condition B (stripped)**: synthesize from PII-stripped cluster (regex removal of emails, IPs, phone numbers, ZIP codes, names, ages)

Claude-as-judge rates each field group (demographics, firmographics, goals, pains, motivations, objections) on grounding quality (1-5).

Backend: Haiku 4.5. 2 synthesis calls + 2 judge calls.

## Results

### Quantitative

| Field Group | Raw Score | Stripped Score | Δ (raw − stripped) |
|---|---|---|---|
| demographics | 2 | 2 | **0** |
| firmographics | 4 | 4 | **0** |
| goals | 4 | 4 | **0** |
| pains | 3 | 2 | **+1** |
| motivations | 2 | 2 | **0** |
| objections | 1 | 1 | **0** |

| Hypothesis check | Value | Threshold |
|---|---|---|
| Demographics degradation | **0** | expected ≥0.5 |
| Max behavioral delta | **1** (pains) | expected ≤0.1 |

### Interpretation

1. **PII stripping had zero impact on demographics.** Both conditions scored 2/5 on demographics. This makes sense: the mock data (GA4 page views, HubSpot contacts, Intercom tickets) contains very little PII in the payload fields. The regex stripper found almost nothing to remove.

2. **The only delta was in pains (−1 point for stripped).** This is likely noise from the judge rather than a real PII effect — pain points aren't derived from PII data.

3. **Both conditions scored poorly on objections (1/5) and motivations (2/5).** This is a data-quality issue, not a PII issue — the mock cluster simply doesn't have enough signal to ground motivations and objections well.

### Why the hypothesis failed

The mock data doesn't contain meaningful PII. The GA4 connector generates events like `{"event": "page_view", "page": "/pricing"}` — no emails, no names, no ages. The HubSpot connector generates company metadata. The Intercom connector generates support tickets with generic content.

For PII stripping to matter, the source data needs to actually contain PII. This experiment needs a fixture with real-looking PII embedded in payloads (e.g., `"message": "Hi, I'm John Smith (john@acme.com), a 34-year-old PM in Denver, CO 80202"`).

## Recommendation

**DEFER** — the experiment infrastructure works correctly but the test data doesn't exercise the PII stripping path. Not a valid test of the hypothesis.

**Re-test when:**
- A PII-rich fixture is available (synthetic records with embedded names, emails, locations, ages)
- Can compare against a real tenant dataset where PII stripping actually removes content
- Running ≥3 trials per condition to control for judge variance

## Cost

- Total API cost: ~$0.15 (2 synthesis calls + 2 judge calls on Haiku 4.5)
- Model: `claude-haiku-4-5-20251001`

## Artifacts

- `raw_personas.json` — raw condition result with persona
- `stripped_personas.json` — stripped condition result with persona
- `summary.json` — per-field-group scores and deltas
