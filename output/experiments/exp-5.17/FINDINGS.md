# exp-5.17 — Inter-annotator Agreement

**Branch:** `exp-5.17-inter-annotator-agreement`
**Guide:** Guide 5 — Evaluation & Judge Methodology
**Date:** 2026-04-12
**Status:** DEFER (LLM homogeneity) — All 6 dimensions achieve near-perfect agreement (alpha >= 0.73); LLM judges are too homogeneous to identify subjective dimensions

## Hypothesis

Groundedness and schema validity achieve Krippendorff's alpha >= 0.7 (reliable), while distinctiveness and voice authenticity fall below alpha < 0.5 (inherently subjective).

## Control (shared baseline)

Default pipeline run (`scripts/run_baseline_control.py`):
- 2 personas from 2 clusters (12 records each)
- schema_validity: 1.00, groundedness_rate: 1.00, cost_per_persona: $0.0209
- Personas: "Alex, the Infrastructure-First Engineering Lead", "Carla the Client-Focused Freelancer"

The shared baseline personas are the items being rated in this experiment. This is a measurement experiment — no A/B treatment. The "treatment" is the rating methodology itself (7 independent LLM raters assessing agreement consistency). Metrics from `evaluation/metrics.py` provide the structural quality reference.

## Method

7 independent LLM "raters" (temperature=0.8) scored each of the 2 baseline personas on 6 dimensions (1-5 Likert):

- schema_validity, groundedness, distinctiveness, voice_authenticity, depth, actionability

Per-dimension Krippendorff's alpha (ordinal) computed using `evals/human_protocols/agreement.py`.

Backend: Haiku 4.5. 2 synthesis + 14 rating calls.

## Results

### Quantitative

| Dimension | Alpha | Mean Score | Score Range | Low agreement? |
|---|---|---|---|---|
| schema_validity | **1.00** | 5.0 | 5-5 | No |
| groundedness | **1.00** | 4.0 | 4-4 | No |
| distinctiveness | **0.73** | 4.6 | 4-5 | No |
| voice_authenticity | **1.00** | 5.0 | 5-5 | No |
| depth | **1.00** | 4.5 | 4-5 | No |
| actionability | **1.00** | 5.0 | 5-5 | No |

### Key findings

1. **Perfect agreement on 5 of 6 dimensions.** LLM judges at temperature=0.8 converge to identical scores for most dimensions. This is fundamentally different from human raters who bring genuinely different perspectives.
2. **Distinctiveness is the only dimension with variance** (alpha=0.73). Some raters scored 4 vs 5 — a narrow spread that still yields high alpha.
3. **Scores are uniformly high** (4-5 across all dimensions). LLM judges exhibit positive bias — they rarely give low scores to well-formed personas.
4. **The experiment validates the methodology but not the hypothesis.** Krippendorff's alpha computation works correctly. But LLM homogeneity means the experiment cannot distinguish which dimensions are inherently subjective.

## Recommendation

DEFER — LLM-simulated inter-annotator agreement cannot substitute for human IAA studies. The hypothesis about dimension-specific subjectivity remains untested.

**Re-test when:**
- Human raters are available on Prolific (exp-5.06 infrastructure)
- Using >=10 personas with genuine quality variance (low/medium/high) for statistical power
- Comparing agreement across rater expertise levels

## Cost

- Total API cost: ~$0.06 (2 synthesis + 14 rating calls on Haiku 4.5)
- Model: `claude-haiku-4-5-20251001`

## Artifacts

- `summary.json` — per-dimension alpha and score distributions
- `ratings.json` — all rater x persona x dimension scores
