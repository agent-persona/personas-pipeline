# B3: Re-run exp-3.22 with Semantic Groundedness Proxy
**Experiment:** B3-rerun-3.22-semantic  
**Date:** 2026-04-10  
**Eval script:** `evals/semantic_rerun_3_22.py`

---

## Context: What exp-3.22 Found

Experiment 3.22 tested a domain-specific grounding rule: a keyword classifier flags "sensitive" claims (financial, legal) and requires N+1 source evidence (>=2 unique record IDs). The structural checker found:

| Cluster | Flagged claims | N+1 satisfied | Flagging rate | Verdict |
|---------|---------------|---------------|---------------|---------|
| cluster_00 (Alex DevOps) | 3 | 2/3 (67%) | 7.9% | MODERATE |
| cluster_01 (Maya Designer) | 5 | 0/5 (0%) | 13.5% | DEFER |
| Combined | 8 | 2/8 (25%) | ~10.7% | MODERATE/DEFER |

Low N+1 satisfaction was explained structurally: decision_triggers and sample_quotes have no field_path entries in source_evidence, so no backing record IDs are ever registered. This made the N+1 check a near-certain fail for any flagged claim in those fields.

**Open question from 3.22:** The structural checker can count sources, but cannot answer whether flagged claims are actually less well-grounded semantically. B3 answers this.

---

## Semantic Proxy Results

The semantic proxy computes token overlap between claim text and the payloads of backing records. Claims scoring < 0.10 are "weak pairs." This runs only on goals, pains, motivations, objections — the four fields with source_evidence entries. decision_triggers, sample_quotes, and vocabulary are excluded (no source evidence registered for them in either persona).

### cluster_00 (Alex DevOps)

| Metric | Value |
|--------|-------|
| Semantic score (all claims) | 0.133 |
| Claims scored | 17 / 17 (coverage 1.0) |
| Weak pairs (overlap < 0.10) | 5 / 17 |
| Avg overlap — flagged claims | 0.1125 |
| Avg overlap — non-flagged claims | 0.1357 |
| Overlap gap (non-flagged minus flagged) | +0.0232 |
| Signal | **WEAK** |

Flagged claims (in semantic fields):
- motivations.3 — "audit requirements ... version-controlled in git" -> overlap 0.125
- objections.2 — "compliance-critical events" -> overlap 0.10

Weakest non-flagged claims: goals.0 (0.0), goals.2 (0.0), motivations.0 (0.0), motivations.1 (0.0) — all from GA4 event records with sparse payloads (event, session_duration only), which produce no useful token overlap with abstract claim language regardless of domain category.

### cluster_01 (Maya Designer)

| Metric | Value |
|--------|-------|
| Semantic score (all claims) | 0.1537 |
| Claims scored | 17 / 17 (coverage 1.0) |
| Weak pairs (overlap < 0.10) | 6 / 17 |
| Avg overlap — flagged claims | 0.1828 |
| Avg overlap — non-flagged claims | 0.1474 |
| Overlap gap (non-flagged minus flagged) | **-0.0354** |
| Signal | **INVERSE** |

Flagged claims (in semantic fields):
- pains.0 — "paying her rates" -> overlap 0.2667 (backed by intercom_004, rich message payload)
- motivations.0 — "bills hourly ... hourly rate" -> overlap 0.10 (intercom_004)
- objections.2 — "subscription cost ... hourly rate" -> overlap 0.1818 (intercom_004)

The flagged financial claims are actually **above** the non-flagged average because they are all backed by intercom_004 — the one record with a natural-language message payload. The non-flagged claims that drag the average down are abstract motivations (motivations.2 = 0.0, motivations.3 = 0.0) backed by sparse GA4 event records.

---

## Does the Keyword Classifier Correlate with Semantic Weakness?

**No — not in these two clusters.**

| Cluster | Direction | Magnitude | Verdict |
|---------|-----------|-----------|---------|
| cluster_00 (Alex DevOps) | Flagged slightly weaker | -0.023 gap | WEAK — below the 0.10 STRONG threshold |
| cluster_01 (Maya Designer) | Flagged actually stronger | +0.035 advantage | INVERSE — opposite of the hypothesis |
| Combined | No consistent direction | — | NOT CORRELATED |

The STRONG threshold for a meaningful correlation is >=0.10 overlap gap in favor of flagged-claims being weaker. Neither cluster reaches it. One cluster runs in the opposite direction.

### Why the classifier fails to predict semantic weakness

1. **The classifier triggers on vocabulary, not evidence quality.** "Pays her rates" (pains.0, Maya) is financial vocabulary — but it is backed by the richest record in the cluster (an intercom message that repeats the exact phrasing). The keyword is present because the claim is well-grounded in that record, not because it is weakly supported.

2. **The main driver of semantic weakness is record payload sparseness.** The genuinely weak pairs (overlap ~0.0) are claims backed by GA4 event records containing only {"event": "api_setup", "session_duration": 2340}. Abstract synthesis claims have no lexical anchor in these payloads regardless of their domain category. This is a structural limitation of GA4 event data, not a domain-sensitivity problem.

3. **Financial/legal keywords often signal direct customer quotes.** Intercom messages — the most semantically rich records — are the most likely to contain financial and legal language because those are the topics users write about explicitly. The classifier therefore tends to flag claims with above-average semantic grounding.

---

## Is the N+1 Rule Targeting the Right Claims?

**No.** The N+1 failures in exp-3.22 were caused by two factors:

1. decision_triggers and sample_quotes have no source_evidence entries — structural gap, not evidence weakness
2. Claims flagged in semantic fields (pains.0, motivations.0, objections.2 for Maya) already have backing — one record each — but that record is highly informative (intercom). The semantic score is not weak; only the source count is low

The N+1 rule would force the pipeline to seek additional sources for claims that are already semantically grounded, while doing nothing for the genuinely weak claims (zero-overlap goals/motivations backed by sparse GA4 events that happen to use non-sensitive vocabulary).

---

## Updated Recommendation

**Do not adopt the domain keyword classifier as currently designed.** The correlation between keyword flags and actual semantic weakness is absent or inverted. Shipping N+1 enforcement on keyword-matched claims would add noise without improving grounding quality.

**Revised path forward:**

1. **Replace keyword-based flagging with semantic-score-based flagging.** Flag claims with overlap < 0.10 directly — these are the genuinely under-evidenced claims regardless of topic. This is a stronger, more precise signal.

2. **Fix the structural gap separately.** decision_triggers and sample_quotes should either receive source_evidence entries or be excluded from N+1 enforcement until they do. The current design penalizes them by default.

3. **Weight source quality, not just count.** A single intercom message may ground a financial claim better than three GA4 event records. N+1 source count is a weak proxy for evidence quality; semantic overlap is a better one.

4. **Re-evaluate on richer record payloads.** The semantic proxy scores are uniformly low (0.10-0.18) because GA4 payloads are event names + durations. The proxy will become more discriminating when record payloads include search queries, support transcripts, or survey responses.

---

## Signal Summary

| | exp-3.22 (structural) | B3 (semantic) |
|---|---|---|
| Flagging rate | ~10.7% | Same — classifier unchanged |
| N+1 satisfaction | 25% | N/A |
| Correlation: flagged -> weak grounding | Not testable | **NOT CONFIRMED** |
| Signal | MODERATE/DEFER | **WEAK / INVERSE** |
| Recommendation | Defer adoption | **Revise classifier — do not adopt as-is** |
