# exp-5.17 — Inter-annotator Agreement

**Branch:** `exp-5.17-inter-annotator-agreement`
**Guide:** Guide 5 — Evaluation & Judge Methodology
**Date:** 2026-04-12
**Status:** FAIL (hypothesis not confirmed) — All 6 dimensions achieve near-perfect agreement (α ≥ 0.73). LLM judges are too homogeneous to test the hypothesis that some dimensions are inherently subjective.

## Hypothesis

Groundedness and schema validity achieve Krippendorff's α ≥ 0.7 (reliable), while distinctiveness and voice authenticity fall below α < 0.5 (inherently subjective).

## Method

This experiment tests agreement consistency rather than baseline vs treatment. 7 independent LLM "raters" (temperature=0.8) scored each persona on 6 dimensions (1-5 Likert):

- **schema_validity**, **groundedness**, **distinctiveness**, **voice_authenticity**, **depth**, **actionability**

2 personas synthesized from 2 available clusters. Per-dimension Krippendorff's α (ordinal) computed using `evals/human_protocols/agreement.py`.

Backend: Haiku 4.5. 2 synthesis + 14 rating calls.

## Results

### Quantitative

| Dimension | α | Mean Score | Score Range | Low agreement? |
|---|---|---|---|---|
| schema_validity | **1.00** | 5.0 | 5-5 | No |
| groundedness | **1.00** | 4.0 | 4-4 | No |
| distinctiveness | **0.73** | 4.6 | 4-5 | No |
| voice_authenticity | **1.00** | 5.0 | 5-5 | No |
| depth | **1.00** | 4.5 | 4-5 | No |
| actionability | **1.00** | 5.0 | 5-5 | No |

### Key findings

1. **Perfect agreement on 5 of 6 dimensions.** LLM judges at temperature=0.8 converge to identical scores for most dimensions. This is fundamentally different from human raters who bring genuinely different perspectives.

2. **Distinctiveness is the only dimension with variance** (α=0.73). Even with temperature variation, most raters agreed, but some scored 4 vs 5 — a narrow spread that still yields high α.

3. **Scores are uniformly high** (4-5 across all dimensions). LLM judges exhibit positive bias — they rarely give low scores to well-formed personas.

4. **The experiment validates the methodology but not the hypothesis.** Krippendorff's α computation works correctly. The `agreement.py` module handles ordinal data. But LLM homogeneity means the experiment cannot distinguish which dimensions are inherently subjective — that requires actual human raters.

## Recommendation

**DEFER** — LLM-simulated inter-annotator agreement cannot substitute for human IAA studies. The hypothesis about dimension-specific subjectivity remains untested.

**Re-test when:**
- Human raters are available on Prolific (exp-5.06 infrastructure)
- Using ≥10 personas with genuine quality variance (low/medium/high) for statistical power
- Comparing agreement across rater expertise levels

## Cost

- Total API cost: ~$0.06 (2 synthesis + 14 rating calls on Haiku 4.5)
- Model: `claude-haiku-4-5-20251001`

## Artifacts

- `summary.json` — per-dimension α and score distributions
- `ratings.json` — all rater × persona × dimension scores
