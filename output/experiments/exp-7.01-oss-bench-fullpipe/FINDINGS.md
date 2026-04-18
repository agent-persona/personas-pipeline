# exp-7.01 — full pipeline vs joongishin/persona-generation-workflow port

**Tenant:** `tenant_acme_corp`  
**Synthesis model:** `claude-haiku-4-5-20251001` (both sides, matched)  
**Judge model:** `claude-haiku-4-5-20251001`  

## Cluster inputs (identical on both sides)

- `clust_7dc89b8e9f89` size=4
- `clust_914dfd0f6510` size=4

## Judged persona quality

Scores are integer 1-5, averaged across clusters. Higher is better.

| Dimension | ours (personas-pipeline) | pgw-port (LLM-summarizing++) |
|---|---:|---:|
| specificity | 4.00 | 4.00 |
| plausibility | 5.00 | 5.00 |
| actionability | 5.00 | 5.00 |
| evidence_bind | 4.50 | 4.50 |

## Cost per persona

- ours (with groundedness retries): **$0.0392** per persona
- pgw-port (single-pass): **$0.0019** per persona

## Schema fidelity (no LLM — direct JSON inspection)

| Metric | ours (avg) | pgw-port (avg) |
|---|---:|---:|
| populated fields | 10.0 | 7.0 |
| `source_evidence` rows | 21.5 | 0.0 |
| sample quotes | 5.0 | 0.0 |
| goals / plans | 5.0 | 4.0 |
| pains | 5.0 | 0.0 |

## Interpretation

- **Narrative judge scores are identical across both methods.** The LLM judge — reading persona text alone — can't distinguish between a grounded persona backed by `source_evidence` record IDs and a single-pass summarization that qualitatively references behaviors. This is an honest negative for our synthesis pipeline *on this particular rubric*.
- **Schema fidelity tells the real story.** Our pipeline ships `source_evidence` rows (e.g., 23 per persona in the shipped output) that bind every claim to specific record IDs. The pgw-port produces none of that structure. A reader auditing a claim ("why does this persona prioritize webhooks?") can click through to the exact records in our output; in the pgw output they cannot.
- **Cost gap (~20x):** ours runs a retry/check loop that enforces groundedness and schema validity; pgw-port is a single Anthropic call. The cost buys auditability and schema richness, not narrative quality as judged by an LLM reading only the persona text.
- **Implication for the narrative judge:** this benchmark surfaces that a generic LLM-as-judge on persona text underestimates our differentiators. A stricter judge (one that sees the records AND the persona, and specifically verifies `source_evidence` record IDs) would detect the gap. That's effectively what our `evaluation/groundedness.py` already does — and it scores ours at 1.0 and pgw-port at 0.0 by construction.