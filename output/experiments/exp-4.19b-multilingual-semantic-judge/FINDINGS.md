# Experiment 4.19b — Multilingual Coherence: Semantic Judge

**Status:** Complete  
**Date:** 2026-04-10  
**Depends on:** exp-4.19 (generated responses, no new API calls)

---

## Motivation

Exp-4.19 used substring matching to score persona trait preservation across English, Spanish, and Mandarin. This had a structural flaw: substring matching requires the exact English keyword to appear in the response. Spanish responses use loanwords that sometimes differ slightly from the target string. Mandarin uses proper domain translations (e.g., 幂等 for "idempotent", 模式漂移 for "schema drift") that correctly convey the concept but produce zero substring matches. Both cases lead to undercounting.

This experiment re-scores the same six responses using the LLM as a semantic judge on a 1–5 rubric.

---

## Judge Scores

| Persona | English | Spanish | Mandarin | Delta (non-EN avg − EN) |
|---------|---------|---------|---------|------------------------|
| Alex (DevOps) | 5 | 5 | 5 | 0.00 |
| Maya (Designer) | 5 | 5 | 4 | −0.50 |
| **Overall avg** | **5.0** | **5.0** | **4.5** | |

---

## Justifications

### Alex the API-First DevOps Engineer

**English (5/5):** Every identity marker is present: GraphQL schema drift, silent-failing automation scripts, the 2am failure detail, IaC layer built on idempotent API calls, webhook payload versioning, compliance-critical pipeline — this is unambiguously Alex.

**Spanish (5/5):** All technical vocabulary is preserved as direct loanwords (GraphQL, IaC, idempotentes, payloads, webhook, pipeline, versionado) or accurate translations (cumplimiento normativo for compliance); only the '2am' specificity is softened, but every professional obsession reads intact.

**Mandarin (5/5):** Uses correct domain translations — 模式漂移 (schema drift), 幂等 (idempotent — the proper Chinese term), IaC层, webhook, pipeline, 合规关键 — and even preserves the '凌晨两点' (2am) anecdote; full character fidelity in Mandarin.

### Maya the Freelance Brand Designer

**English (5/5):** Full identity: logo-on-share-view frustration, deliverable link, brand kit and positioning undermined, premium-rate client perception anxiety, and the specific ask for a white-label client portal — every Maya obsession is here.

**Spanish (5/5):** Key vocabulary preserved as loanwords (brand kit, client portal, white-label) with accurate translations elsewhere (posicionamiento profesional, enlace de entrega); the brand-anxiety motivation and premium-rate self-consciousness come through fully.

**Mandarin (4/5):** White-label and brand kit survive as English terms, but 'client portal' becomes 客户門户 (losing the specific English label) and 'deliverable' disappears into the generic 交付链接; core brand-anxiety identity is intact but two signature vocabulary items are softened.

---

## Metric Comparison: Substring vs. Semantic

| Persona | Language | Substring Score | Semantic Score (1–5) | Agreement |
|---------|----------|----------------|----------------------|-----------|
| Alex | English | 0.50 | 5 | Substring undercounts |
| Alex | Spanish | 0.50 | 5 | Substring undercounts |
| Alex | Mandarin | 0.375 | 5 | **Strong disagreement** — Mandarin 幂等 correct but unmatched |
| Maya | English | 0.333 | 5 | Substring undercounts |
| Maya | Spanish | 0.333 | 5 | Substring undercounts |
| Maya | Mandarin | 0.222 | 4 | **Strong disagreement** — semantic still high, substring near-zero |

**Where they agree:** Both metrics show Mandarin scoring lower than Spanish (substring delta −0.0625 for Alex, −0.0555 for Maya). The direction of the difference is consistent.

**Where they disagree:** The magnitude is completely different. Substring matching makes all scores look mediocre (0.22–0.50), implying significant persona degradation in every language. Semantic judging shows that all responses are high-fidelity (4–5), and the Mandarin "drop" is minor (−0.50 from control), not a collapse.

---

## Spanish Loanwords vs. Mandarin Translation

**Spanish:** Technical English terms are absorbed directly into the Spanish text as loanwords — `payloads`, `webhook`, `pipeline`, `brand kit`, `client portal`, `white-label`. This means Spanish responses score identically on both metrics for vocabulary that was borrowed. Spanish achieves near-parity with English under both metrics.

**Mandarin:** Mandarin uses domain-standard Chinese translations where they exist (幂等 = idempotent, 模式漂移 = schema drift, 版本控制 = versioning) or preserves English terms that have no established Mandarin equivalent (GraphQL, IaC, webhook, white-label, brand kit). The substring matcher misses the translated terms entirely, generating false negatives. The semantic judge correctly identifies these as full-fidelity translations. The only genuine softening in Mandarin is Maya's "client portal" → 客户门户 and loss of "deliverable" as a term of art.

**Conclusion:** The hypothesis that Spanish loanwords were undercounted was correct. The hypothesis that Mandarin was undercounted was also correct — and more severely so, because Mandarin uses correct translations rather than borrowing. The substring metric was penalizing linguistic correctness.

---

## Signal Assessment

| Criterion | Value | Pass? |
|-----------|-------|-------|
| Spanish avg >= 4 | 5.0 | Yes |
| Mandarin avg <= 3.5 | 4.5 | No |
| Signal | — | **WEAK** |

The predefined STRONG signal condition required Mandarin <= 3.5. Mandarin scored 4.5 on average because the responses are genuinely high-fidelity — the model translated persona vocabulary correctly. The WEAK signal here is not a negative result; it means the original hypothesis was too pessimistic about Mandarin quality. The real finding is that substring match created a false alarm about Mandarin degradation.

---

## Recommendation

**ADOPT semantic judge over substring match for multilingual evals.**

Substring matching is unsuitable for multilingual persona scoring because it conflates "did not use the English keyword" with "did not preserve the concept." This is a measurement error, not a model quality problem.

**Characterize Spanish and Mandarin differently in documentation:**
- **Spanish:** Preserves persona vocabulary via loanword integration. Near-parity with English. Substring match is a reasonable proxy because loanwords match literally.
- **Mandarin:** Preserves persona vocabulary via domain-standard translations. Semantic judge is required. Substring match systematically undercounts fidelity and should not be used for Mandarin evaluation.

The semantic judge adds nuance substring match cannot capture: Maya's Mandarin response has a genuine minor softening (two terms translated out of their English form) that a substring matcher would inflate into a much larger-looking gap. The 4 vs. 5 distinction is real and meaningful; the 0.22 vs. 0.50 substring distinction is mostly noise.
