# Findings: Track B2 — Exp-2.20 Semantic Groundedness Re-Run

## Context

Experiment 2.20 tested **transcript-first persona synthesis** (interview-style):
the pipeline first synthesizes a simulated transcript from cluster records, then
derives the persona from that transcript. Both the baseline and transcript-first
personas scored **1.0 on structural groundedness** — the metric was at ceiling
and produced zero signal (`NOISE`, delta=0.0).

This re-run applies the **semantic groundedness proxy** to measure token-level
overlap between persona claims and the cited source records.

---

## Original Exp-2.20 Result (Structural Groundedness)

| Method | Structural Score | Delta |
|---|---|---|
| Baseline | 1.0 | — |
| Transcript-first | 1.0 | **0.0 (NOISE)** |

The structural metric only checks whether `source_evidence` entries are present
and cite valid record IDs. Both personas cite all fields, so the score is always
1.0 regardless of claim quality.

---

## New Semantic Scores

### Cluster 00 — `clust_1adb81b417c0`

| Method | semantic_score | claim_count | weak_count | coverage |
|---|---|---|---|---|
| Baseline | 0.133 | 17 | 5 | 1.0 |
| Transcript-first | 0.0966 | 17 | 6 | 1.0 |

**semantic_score_delta: -0.0364**

### Cluster 01 — `clust_bc52ee85eb83`

| Method | semantic_score | claim_count | weak_count | coverage |
|---|---|---|---|---|
| Baseline | 0.1537 | 17 | 6 | 1.0 |
| Transcript-first | 0.1697 | 17 | 4 | 1.0 |

**semantic_score_delta: +0.0160**

---

## Aggregate Summary

| Metric | Value |
|---|---|
| avg_baseline_semantic_score | 0.1434 |
| avg_transcript_first_semantic_score | 0.1331 |
| avg_semantic_score_delta | -0.0103 |
| Signal | **NOISE** |
| Recommendation | **REJECT** |

---

## Weak-Pair Analysis

Weak pairs are claim-evidence pairs with token overlap < 0.10 (claim vocabulary
almost entirely absent from cited records).

**Cluster 00 — Baseline weak pairs:** 5
**Cluster 00 — Transcript-first weak pairs:** 6

**Cluster 01 — Baseline weak pairs:** 6
**Cluster 01 — Transcript-first weak pairs:** 4

---

## Qualitative Observation

The semantic proxy measures whether the words used in persona claims appear in
the payload of the cited source records. A higher score means claims stay closer
to the vocabulary of the underlying data rather than drifting into paraphrase or
hallucination.

Transcript-first synthesis routes all information through a simulated interview
transcript before persona generation. This introduces an additional abstraction
layer: the model first rewrites record payloads as natural interview speech, then
generates claims from that speech. The effect on token overlap depends on whether
the transcript preserves or transforms the raw vocabulary.

- If transcript-first **preserves** source vocabulary → higher semantic overlap
- If transcript-first **paraphrases** heavily → lower overlap even when factually grounded

The delta observed here (-0.0103 average) reflects this tradeoff.

---

## Conclusion

**Signal: NOISE**

No meaningful semantic advantage detected for transcript-first synthesis. The intermediate transcript step appears to transform vocabulary enough to reduce or neutralize token overlap with cited records.

**Recommendation: REJECT**
