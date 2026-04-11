# Batch 4 Experiment Results — Handoff

**Date:** 2026-04-11
**Branches:** exp-1.07, exp-2.06, exp-5.05, exp-5.11
**Status:** Branch hygiene cleaned. All 4 branches isolated, correct bases, no cross-contamination.

---

## Summary

| Experiment | Decision | Confidence | Signal |
|-----------|----------|------------|--------|
| exp-1.07 field interdependence | **Adopt** | Narrow (1 tenant, 1 persona, coarse scores) | goals + sample_quotes are load-bearing |
| exp-2.06 temperature sweep | **Adopt** | Moderate (2 clusters, one 429 noise) | temp=0.0 is the practical default |
| exp-5.11 reference vs free judging | **Reject** | High (clearest negative of the set) | Anchoring bias makes it worse than baseline |
| exp-5.05 rubric ablation | **Defer** | Low (ceiling effect, degenerate stats) | Two-tier rubric structure hinted but not confirmed |

---

## 1.07 Field Interdependence

**Branch:** `exp-1.07-field-interdependence` (from `origin/main`)
**Decision:** Adopt
**Artifacts:** `output/experiments/exp-1.07-field-interdependence/{FINDINGS.md, results.json, control_persona.json}`

### Findings

Load-bearing fields: `goals`, `sample_quotes`
Decorative or weak-support: `pains`, `motivations`, `objections`, `channels`, `vocabulary`, `decision_triggers`, `journey_stages`

Main deltas:
- Removing `goals` dropped grounded, distinctive, coherent by -1 each
- Removing `sample_quotes` dropped voice_fidelity by -2 and distinctive by -1

### Interpretation

- `goals` is the structural anchor — its presence holds up 3 scoring dimensions
- `sample_quotes` carries most of the voice signal
- Several schema fields appear low-value to the current judge

### Caveat

Based on one control persona, one tenant, coarse 1-5 judge scores. Useful directional signal, not yet robust.

### Next steps

- Prioritize `goals` + `sample_quotes` in synthesis QA
- `channels`/`vocabulary`/`journey_stages` can be deprioritized in cost-constrained scenarios
- Strengthen by running on multiple tenants with finer-grained scoring

---

## 2.06 Temperature Sweep

**Branch:** `exp-2.06-temperature-sweep` (from `origin/main`)
**Decision:** Adopt
**Artifacts:** `output/experiments/exp-2.06-temperature-sweep/{FINDINGS.md, results.json}`

### Findings

Best variant: `temperature=0.0`
- Groundedness reached 1.0
- Retry rate dropped to 50% (vs control)
- Avg cost dropped to $0.027 vs control $0.039 (30% savings)
- Distinctiveness stayed flat (~0.85-0.87)
- `top_p` had no useful signal
- Higher temperatures did not improve diversity enough to matter

### Interpretation

- Retry + groundedness checks dominate quality — temperature mainly affects efficiency, not output quality
- `temp=0.0` is the practical default

### Caveat

- Tiny sample: 2 clusters
- temp=0.4 had a 429 rate limit hit, so that data point is noisy

### Next steps

- Set `temperature=0.0` as production default in `AnthropicBackend`
- The model_backend.py change (optional `temperature`/`top_p` params) is ready to merge — defaults unchanged, backward compatible

---

## 5.11 Reference-Based vs Free Judging

**Branch:** `exp-5.11-reference-based-vs-reference-free` (from `origin/exp-5.13`)
**Decision:** Reject
**Artifacts:** `output/experiments/exp-5.11-reference-vs-free-judging/{FINDINGS.md, results.json}`

### Findings

- Reference mode reduced variance by 44% (std 0.548 -> 0.408)
- But introduced obvious anchoring:
  - Mean score shifted down by -0.333
  - Spearman rho = 0.657
  - 6/6 reference-mode scores were within 0.5 of the anchor quality

### Interpretation

- The reference does tighten scores, but by flattening them around the anchor, not by improving judgment
- It distorts rank ordering and penalizes stronger personas
- This is the clearest negative finding of the set

### Next steps

- Do not adopt reference-mode judging
- Few-shot calibration (exp-5.13) remains the better approach for score tightening without rank distortion

---

## 5.05 Rubric Ablation

**Branch:** `exp-5.05-rubric-ablation` (from `origin/exp-5.13`)
**Decision:** Defer (not trustworthy yet)
**Artifacts:** `output/experiments/exp-5.05-rubric-ablation/{FINDINGS.md, results.json, report.txt}`

### Findings

The experiment ran but produced degenerate results:
- All 5 personas scored identically (grounded=4, distinctive=4, coherent=5, actionable=5, voice_fidelity=5)
- Pairwise correlations: all NaN (zero variance)
- Kendall tau: all 0.000 (flat control rankings)
- Score shifts showed asymmetric cross-dimension effects (distinctive/actionable/voice_fidelity act as "anchor" dims)

### Interpretation

- Ceiling effect collapses all rank-based metrics
- The two-tier rubric structure (anchor dims vs independent dims) is an interesting hint but not confirmed
- The experiment needs wider quality variance (intentionally degraded personas) to produce meaningful statistics

### Next steps

- Rerun with degraded personas (remove evidence, genericize quotes) for wider score distribution
- Expand to additional tenants
- Consider testing with Opus as judge to check model-specific effects
- Retain current 5-dimension rubric as-is until follow-up confirms

---

## Branch hygiene

All 4 branches have been cleaned (2026-04-11):
- exp-1.07 and exp-2.06: correctly based from `origin/main`
- exp-5.05 and exp-5.11: correctly based from `origin/exp-5.13`
- No cross-experiment contamination
- Single clean commit per branch on top of base

Verification:
```bash
git branch -r | grep -E "exp-(2.06|1.07|5.05|5.11)"
# origin/exp-1.07-field-interdependence
# origin/exp-2.06-temperature-sweep
# origin/exp-5.05-rubric-ablation
# origin/exp-5.11-reference-based-vs-reference-free
```

---

## Remaining batch 4 work

From `docs/plans/2026-04-10-batch4-research-strategy.md`, the following experiments are still pending:
- A1 (3.24): Counterfactual grounding swap
- A2 (3.19b): Recency weighting with real temporal fixture
- A3 (4.19b): Multilingual coherence with semantic judge
- A4 (1.23b): Off-label adversarial probes
- A5 (4.21b): Curiosity throttle
- A6 (3.25): Semantic groundedness validation (depends on Track B)
- Track B: Semantic groundedness proxy infra
