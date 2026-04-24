# exp-5.18 — Crowd Worker vs Expert Reviewer

**Branch:** `exp-5.18-crowd-vs-expert`
**Guide:** Guide 5 — Evaluation & Judge Methodology
**Date:** 2026-04-12
**Status:** DEFER (degenerate kappa) — kappa=0.0 on 5/6 dimensions and -0.33 on distinctiveness; both arms produce near-identical scores on only 2 items

## Hypothesis

"Crowd" (general-purpose LLM judge) labels suffice (kappa >= 0.6 vs "experts") for schema validity and groundedness, but diverge (kappa < 0.4) on distinctiveness and voice authenticity.

## Control (shared baseline)

Default pipeline run (`scripts/run_baseline_control.py`):
- 2 personas from 2 clusters (12 records each)
- schema_validity: 1.00, groundedness_rate: 1.00, cost_per_persona: $0.0209
- Personas: "Alex, the Infrastructure-First Engineering Lead", "Carla the Client-Focused Freelancer"

The shared baseline personas are the items being rated. Baseline arm = crowd-style LLM judges (standard prompt), treatment arm = expert-framed LLM judges (senior UX researcher prompt). Metrics from `evaluation/metrics.py` provide the structural quality reference.

## Method

Two LLM judge "arms" rated the same 2 personas on 6 dimensions (1-5 Likert):

- **Baseline (crowd arm)** — 5 LLM judge calls per persona, standard evaluation prompt, temperature=0.8
- **Treatment (expert arm)** — 3 LLM judge calls per persona, expert-framed prompt (senior UX researcher with 10+ years building buyer personas), temperature=0.5

Per-dimension scores aggregated per arm (mean -> round to int). Cohen's kappa computed between the two aggregate label vectors.

Backend: Haiku 4.5. 2 synthesis + 16 rating calls.

## Results

### Quantitative

| Dimension | Cohen's kappa | Crowd suffices (kappa >= 0.6)? |
|---|---|---|
| schema_validity | 0.00 | No |
| groundedness | 0.00 | No |
| distinctiveness | **-0.33** | No |
| voice_authenticity | 0.00 | No |
| depth | 0.00 | No |
| actionability | 0.00 | No |

### Key findings

1. **kappa=0.0 is a statistical artifact, not genuine disagreement.** When both arms round to the same integer on both items, there is zero variance for kappa to measure. The metric is undefined/degenerate with only 2 items.
2. **Distinctiveness kappa=-0.33 reflects a single-point disagreement.** One arm scored 4, the other 5, on one of the 2 items. With 2 items, any disagreement produces extreme kappa values.
3. **The expert prompt did not make the LLM more critical.** Despite framing the judge as a "senior UX researcher," the expert arm produced scores nearly identical to the crowd arm. The same model cannot genuinely simulate the difference between a crowd worker and a domain expert.
4. **N=2 items is fatally underpowered.** Cohen's kappa needs substantial item variety — at minimum 10+ items with genuine quality variance — to produce meaningful cross-group agreement scores.

## Recommendation

DEFER — Cannot conclude anything about crowd vs expert agreement from 2 items. The methodology (dual-arm kappa computation) works, but the experiment needs real rater diversity and more items.

**Re-test when:**
- >=10 personas spanning low/medium/high quality tiers
- Actual Prolific crowd workers vs recruited domain experts (the same LLM cannot play both roles)
- Sufficient quality variance in the persona pool to produce non-degenerate kappa values

## Cost

- Total API cost: ~$0.065 (2 synthesis + 16 rating calls on Haiku 4.5)
- Model: `claude-haiku-4-5-20251001`

## Artifacts

- `summary.json` — per-dimension kappa values
- `ratings.json` — crowd and expert arm scores per persona
