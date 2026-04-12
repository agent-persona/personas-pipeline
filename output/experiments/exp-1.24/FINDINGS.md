# exp-1.24 — Stylometric Anchors

**Branch:** `exp-1.24-stylometric-anchors`
**Guide:** Guide 1 — Schema Design
**Date:** 2026-04-12
**Status:** WEAK — drift is near-zero in both conditions; anchors did not measurably help.

## Hypothesis

Adding explicit stylometric fields (sentence length, formality register, hedge frequency, discourse markers, pronoun preference) to the persona schema will reduce voice drift by ≥30% across 20-turn twin conversations.

## Method

Two Pydantic schemas with identical base fields — only the treatment adds a `Stylometrics` sub-model:

- **Baseline** — `PersonaV1` (no stylometric fields)
- **Treatment** — `PersonaV1WithStylometrics` (adds `stylometrics: Stylometrics`)

The twin system prompt renders stylometric anchors as a `## How you write` section when present. Both conditions use the same 20 conversational prompts covering work challenges, tools, evaluation criteria, and forward-looking concerns.

Drift is measured as cosine similarity of a 5-dimensional vector (avg_sentence_length, hedge_rate, i_ratio, we_ratio, you_ratio) at each turn relative to turn 1. Lower slope = less drift.

Backend: Haiku 4.5. 2 clusters × 2 schemas × 20 turns = 80 conversation turns + 4 synthesis calls.

## Results

### Quantitative

| Metric | Baseline (PersonaV1) | Treatment (WithStylometrics) | Δ |
|---|---|---|---|
| n_personas_ok | 2/2 | 2/2 | 0 |
| Mean drift slope | −0.000049 | −0.000072 | −0.000023 (worse) |
| Mean final cosine sim | 0.9996 | 0.9999 | +0.0003 |
| Drift reduction % | — | — | **−46.9%** (treatment drifted MORE) |

### Interpretation

Both conditions show **near-zero drift** — cosine similarity to turn 1 stays above 0.999 through all 20 turns. The 5-dimensional stylometric vector barely moves across turns for either condition, meaning the twin already maintains remarkably consistent sentence length, hedge rates, and pronoun usage without explicit anchors.

The treatment actually had a marginally steeper (more negative) drift slope, though the absolute magnitude is negligible (−0.000072 vs −0.000049). The "−46.9% drift reduction" is misleading — it's the ratio of two near-zero numbers.

### Why the hypothesis failed

1. **Haiku 4.5 is already stylistically stable.** The model doesn't drift much over 20 turns when given a well-structured persona prompt. The `## How you talk` section with vocabulary and sample quotes already anchors voice effectively.
2. **The 5-dim vector may be too coarse.** Sentence length and pronoun ratios capture shallow stylistic features. If drift manifests in subtler ways (word choice specificity, metaphor density, register shifts), this vector won't detect it.
3. **20 turns may be too few.** The hypothesis expected meaningful drift that anchors would prevent. With 20 turns, the twin hasn't had enough conversation to accumulate significant style changes.

## Recommendation

**DEFER** stylometric anchors. The mechanism works (schema populates correctly, twin prompt renders anchors), but the problem it solves — voice drift over short conversations — is not actually observed with the current model and conversation length.

**Re-test when:**
- Using longer conversations (50+ turns)
- Comparing across model tiers (e.g., Haiku vs Sonnet vs Opus, where cheaper models may drift more)
- Using a richer stylometric vector (e.g., embedding-based style distance)

## Cost

- Total API cost: ~$0.40 (4 synthesis calls + 80 conversation turns on Haiku 4.5)
- Model: `claude-haiku-4-5-20251001`

## Artifacts

- `baseline_personas.json` — 2 personas from PersonaV1
- `treatment_personas.json` — 2 personas from PersonaV1WithStylometrics
- `conversations.json` — full 20-turn transcripts with per-turn stylometric vectors
- `summary.json` — drift metrics
