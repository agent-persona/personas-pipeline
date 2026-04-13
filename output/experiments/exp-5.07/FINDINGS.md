# exp-5.07 — Time-to-detect Curve

**Branch:** `exp-5.07-time-to-detect`
**Guide:** Guide 5 — Evaluation & Judge Methodology
**Date:** 2026-04-12
**Status:** WEAK (inverted) — Treatment detected FASTER than baseline, reversing the hypothesis; over-specificity is a detection signal

## Hypothesis

Full-cluster (treatment) personas evade detection >=2x longer than sparse-cluster (baseline) personas, because richer synthesis data produces more realistic, harder-to-detect conversation partners.

## Control (shared baseline)

Default pipeline run (`scripts/run_baseline_control.py`):
- 2 personas from 2 clusters (12 records each)
- schema_validity: 1.00, groundedness_rate: 1.00, cost_per_persona: $0.0209
- Personas: "Alex, the Infrastructure-First Engineering Lead", "Carla the Client-Focused Freelancer"

The shared baseline confirms the pipeline's default quality level. This experiment uses a sparse 5-record subsample as the experiment's baseline condition and the full 12-record cluster as treatment, measuring detection evasion in TwinChat conversations.

## Method

Two personas synthesized from the same largest cluster:

- **Baseline** — 5-record random subsample -> "Dev Infrastructure Lead"
- **Treatment** — full 12-record cluster -> "Marcus, the Infrastructure-First Platform Engineer"

Each persona ran a 15-turn product-research interview via TwinChat. At turns 1, 3, 5, 7, 10, 15, five independent LLM judges (temperature=0.8) judged "Is this responder a real person or an AI?" with confidence 1-5. Detection point: first turn where verdict="AI" and confidence >= 4.

Backend: Haiku 4.5. 2 synthesis + 30 conversation turns + 60 judge calls.

## Results

### Quantitative

| Metric | Baseline (sparse) | Treatment (full) | Delta |
|---|---|---|---|
| Mean TTD (turns) | **15.0** | **6.4** | -8.6 (worse) |
| Realism score | **0.93** | **0.84** | -0.09 |
| Detection rate | 40% (2/5) | **100%** (5/5) | +60% |
| TTD ratio | -- | -- | **0.43x** (inverted) |

### Key findings

1. **The hypothesis is inverted.** More data makes personas more detectable, not less. The sparse persona was harder to identify because it made fewer specific claims — less to scrutinize.
2. **Over-specificity is a detection signal.** The full-cluster persona ("Marcus") produced highly detailed technical claims (GraphQL schemas, webhook listeners, infrastructure patterns) that read as synthetic to judges. Real people in interviews tend to be less perfectly articulate.
3. **The sparse persona's vagueness is protective.** With fewer grounded claims, "Dev Infrastructure Lead" produced more natural-sounding general statements that judges could not distinguish from a real person until turn 15.
4. **Both start clean.** Neither persona is detected at turns 1-3. Detection diverges at turn 5 for the treatment.

## Recommendation

WEAK — The TTD metric itself works and produces clean signal, but the finding contradicts the hypothesis: data richness hurts rather than helps detection resistance.

**Action items:**
1. Consider adding "natural imperfection" to the twin runtime — hedging, incomplete recall, topic deflection — to counteract the over-specificity problem
2. The TTD metric is useful for evaluation even if this hypothesis fails; it captures something that Likert realism scores miss
3. Re-test with a persona schema that includes uncertainty markers (e.g., "things I'm not sure about")

## Cost

- Total API cost: ~$0.23 (2 synthesis + 30 conversation turns + 60 judge calls on Haiku 4.5)
- Model: `claude-haiku-4-5-20251001`

## Artifacts

- `summary.json` — baseline vs treatment metrics
- `conversations.json` — full 15-turn transcripts for both conditions
- `judgments.json` — per-turn judge verdicts with rationales
