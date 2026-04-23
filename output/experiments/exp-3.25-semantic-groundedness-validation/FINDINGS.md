# Experiment 3.25 — Semantic Groundedness Validation

**Date:** 2026-04-10
**Hypothesis:** The structural checker passes >90% of claims that the semantic proxy flags as weakly grounded — confirming the proxy adds real signal.

---

## Summary

The hypothesis is **confirmed**. The structural checker scores both personas at **1.0** (perfect) while the semantic proxy flags 11 of 34 claims (32.4%) as weakly grounded (token overlap < 0.1). Every diverging claim passed structural and failed semantic — a **100% divergence ratio among weak claims**. The semantic proxy adds genuine signal the structural checker cannot provide.

---

## Structural Scores

| Persona | Structural Score | Violations |
|---------|-----------------|------------|
| Alex (DevOps) | 1.0000 | 0 |
| Maya (Designer) | 1.0000 | 0 |

Both personas hit the structural ceiling. The checker validates that record_ids exist and field_paths are covered — nothing more. It cannot distinguish between a claim that says "api_setup" (matched by event name) and a claim that elaborates rich motivational language with zero vocabulary overlap to the underlying record payloads.

---

## Semantic Scores and Weak Pairs

| Persona | Semantic Score | Claims Analyzed | Weak (overlap < 0.1) | Divergence Rate |
|---------|---------------|-----------------|----------------------|-----------------|
| Alex (DevOps) | 0.1330 | 17 | 5 | 29.4% |
| Maya (Designer) | 0.1537 | 17 | 6 | 35.3% |
| **Aggregate** | **0.1434** | **34** | **11** | **32.4%** |

---

## Per-Claim Divergence Table

### Alex (DevOps)

| field_path | Structural | Semantic | Overlap | Diverges? |
|------------|-----------|----------|---------|-----------|
| goals.0 | PASS | FAIL | 0.0000 | YES |
| goals.1 | PASS | PASS | 0.1000 | - |
| goals.2 | PASS | FAIL | 0.0000 | YES |
| goals.3 | PASS | PASS | 0.1429 | - |
| goals.4 | PASS | PASS | 0.1111 | - |
| pains.0 | PASS | PASS | 0.5556 | - |
| pains.1 | PASS | PASS | 0.1250 | - |
| pains.2 | PASS | FAIL | 0.0833 | YES |
| pains.3 | PASS | PASS | 0.2500 | - |
| pains.4 | PASS | PASS | 0.2000 | - |
| motivations.0 | PASS | FAIL | 0.0000 | YES |
| motivations.1 | PASS | FAIL | 0.0000 | YES |
| motivations.2 | PASS | PASS | 0.1250 | - |
| motivations.3 | PASS | PASS | 0.1250 | - |
| objections.0 | PASS | PASS | 0.2000 | - |
| objections.1 | PASS | PASS | 0.1429 | - |
| objections.2 | PASS | PASS | 0.1000 | - |

### Maya (Designer)

| field_path | Structural | Semantic | Overlap | Diverges? |
|------------|-----------|----------|---------|-----------|
| goals.0 | PASS | PASS | 0.2222 | - |
| goals.1 | PASS | PASS | 0.3000 | - |
| goals.2 | PASS | FAIL | 0.0909 | YES |
| goals.3 | PASS | PASS | 0.1667 | - |
| goals.4 | PASS | PASS | 0.1000 | - |
| pains.0 | PASS | PASS | 0.2667 | - |
| pains.1 | PASS | PASS | 0.1333 | - |
| pains.2 | PASS | PASS | 0.2000 | - |
| pains.3 | PASS | PASS | 0.2000 | - |
| pains.4 | PASS | FAIL | 0.0909 | YES |
| motivations.0 | PASS | PASS | 0.1000 | - |
| motivations.1 | PASS | FAIL | 0.0833 | YES |
| motivations.2 | PASS | FAIL | 0.0000 | YES |
| motivations.3 | PASS | FAIL | 0.0000 | YES |
| objections.0 | PASS | PASS | 0.4000 | - |
| objections.1 | PASS | FAIL | 0.0769 | YES |
| objections.2 | PASS | PASS | 0.1818 | - |

---

## Divergence Rate

- **Overall divergence rate: 32.4%** (11 of 34 claims pass structural, fail semantic)
- Among the 11 claims flagged as weak by the semantic proxy, **100% also passed structural** — the structural checker missed every single one.
- The structural checker false-positive rate (claims it passes that are semantically empty): **100% of weak claims**

---

## Agreement Rate

- **Overall agreement rate: 67.6%** (23 of 34 claims where both metrics agree)
- Agreement is driven entirely by the "both pass" case — there are **zero claims that both fail**, confirming the structural checker never fails when record IDs are valid.

---

## Most Revealing Weak Pairs

### Alex (DevOps)

**goals.0** — "Automate all project state transitions via API so no engineer has to manually update tickets"
- Cited: `ga4_000`, `ga4_003`, `ga4_007` — all payloads: `{"event": "api_setup", "session_duration": N}`
- Overlap: **0.0** — claim vocabulary (automation, state transitions, tickets, engineers) has zero match with event-name payloads.

**motivations.0** — "Reducing toil — every manual step is a future incident waiting to happen"
- Cited: `ga4_000`, `ga4_003`, `ga4_007` (api_setup sessions)
- Overlap: **0.0** — "toil", "manual step", "incident" are entirely LLM-inferred from a behavioral event count.

**motivations.1** — "Proving that infrastructure-as-code extends to the project management layer, not just compute"
- Cited: `ga4_008` (`terraform_setup` event, session_duration: 870)
- Overlap: **0.0** — The record confirms a terraform_setup event happened. The IaC philosophy claim is synthesized narrative.

### Maya (Designer)

**motivations.2** — "Winning repeat business and referrals by making the review process feel effortless for non-designer clients"
- Cited: `ga4_014` (client_share, 280s), `ga4_016` (comment_threading, 920s)
- Overlap: **0.0** — "repeat business", "referrals", "effortless" — none appear in either record payload.

**motivations.3** — "Building a scalable solo practice where systems do the administrative work so she can focus on creative output"
- Cited: `ga4_019` (brand_kit_creation), `intercom_004` (white-label message)
- Overlap: **0.0** — "scalable", "practice", "administrative", "creative output" are not present in either record.

**objections.1** — "The template library isn't deep enough in brand identity categories to save me meaningful time on real projects"
- Cited: `ga4_011`, `ga4_015`, `ga4_017` (all template_browsing events)
- Overlap: **0.077** — The word "template" is shared, but the specific objection about depth and brand identity is synthesized.

---

## Root Cause

GA4 behavioral records carry payloads of the form `{"event": "<event_name>", "session_duration": <N>}`. These are behavior signals, not content signals. The LLM uses them as structural evidence for rich interpretive claims. Structurally valid (record ID exists, field_path covered). Semantically empty (claim vocabulary unrelated to two-field payload).

Intercom records are the exception: free-text payloads produce high overlap because the LLM quotes or paraphrases them. Confirmed by pains.0 (Alex) scoring 0.5556 — cited against `intercom_000` whose payload contains the verbatim phrase "GraphQL endpoint has some rough edges."

---

## Signal Assessment

**Signal: MODERATE** (divergence rate 32.4%, below the >50% STRONG threshold)

The more informative statistic: among claims the semantic proxy flags as weak, **the structural checker misses 100% of them**. The divergence rate is bounded by the fact that GA4 event names (webhook_config, github_integration, asset_export) do share tokens with claim text at the border of the 0.1 threshold. The 11 truly diverging claims are where the LLM abstracted beyond the event vocabulary entirely.

---

## Recommendation

**ADOPT the semantic proxy as a supplemental metric alongside the structural checker.**

| Metric | What it catches | What it misses |
|--------|----------------|----------------|
| Structural | Hallucinated record IDs, uncovered field paths | Semantically empty citations |
| Semantic proxy | LLM elaboration beyond record content | Missing field_path entries, invalid IDs |

**Suggested thresholds for production:**
- semantic_score < 0.08: auto-reject, trigger regeneration
- semantic_score 0.08-0.15: flag for human review
- semantic_score > 0.15: accept

**Next steps:**
1. Enrich GA4 payloads with page titles, search queries, or exit reasons to give the proxy more surface area
2. Run the proxy at synthesis time (post-generation, pre-emit) — weak semantic score triggers regeneration with stricter evidence constraints
3. Lower overlap threshold to 0.08 to catch near-zero cases without over-flagging borderline behavioral records
