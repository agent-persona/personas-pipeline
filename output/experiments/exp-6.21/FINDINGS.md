# exp-6.21 — Population-level Turing Test

**Branch:** `exp-6.21-population-turing-test`
**Guide:** Guide 6 — Population Distinctiveness & Coverage
**Date:** 2026-04-12
**Status:** FAIL — Both conditions score 0% pass rate; judges unanimously identify both sets as AI-generated due to set-level composition issues

## Hypothesis

A treatment persona set (full-cluster synthesis) passes the population Turing test >=70% of the time, while the baseline set (sparse 3-record clusters) passes <=40%.

## Control (shared baseline)

Default pipeline run (`scripts/run_baseline_control.py`):
- 2 personas from 2 clusters (12 records each)
- schema_validity: 1.00, groundedness_rate: 1.00, cost_per_persona: $0.0209
- Personas: "Alex, the Infrastructure-First Engineering Lead", "Carla the Client-Focused Freelancer"

The shared baseline confirms the pipeline produces valid, grounded individual personas. The Turing test evaluates whether these personas, presented as a SET, are perceived as representing a real customer base. Individual quality (per `evaluation/metrics.py`) does not guarantee set-level credibility.

## Method

Two persona sets synthesized from the same tenant data (`tenant_acme_corp`, B2B SaaS):

- **Baseline** — 2 personas from sparse clusters (3 records each via random subsample)
- **Treatment** — 2 personas from full clusters (12 records each)

Each set presented as a complete portfolio to 10 independent LLM judges (temperature=0.8). Judges answered:
1. "Does this set represent a real company's customer base?" (real/AI-generated/unsure)
2. "What percentage of a real customer base does this set cover?" (0-100)
3. "Rate overall authenticity" (1-5)

Backend: Haiku 4.5. 4 synthesis calls + 20 judge calls.

## Results

### Quantitative

| Metric | Baseline (sparse) | Treatment (full) | Delta |
|---|---|---|---|
| Turing pass rate | **0%** (0/10) | **0%** (0/10) | 0 |
| Mean perceived coverage | 16% | 25% | **+9%** |
| Mean authenticity | 2.0/5 | 2.0/5 | 0 |
| Verdicts: AI-generated | 10/10 | 10/10 | 0 |

### Key findings

1. **Both conditions fail unanimously.** All 20 judgments returned "AI-generated." Data richness (3 vs 12 records) has zero effect on set-level perception.
2. **Coverage is the only dimension that moves.** Treatment set perceived coverage (25%) exceeds baseline (16%), a +9pp delta. Judges noticed the full-cluster personas had somewhat more plausible detail, but this did not help overall authenticity.
3. **The failure is structural, not synthesis quality.** Judges consistently flag: only 2 personas for a B2B engineering product (should be 4-6+), one persona (design freelancer) misaligned with the stated product, and missing core roles (engineering managers, tech leads, QA, product managers).
4. **Individual quality does not equal set quality.** Well-scored individual personas still fail the population-level test when the portfolio composition is wrong.

## Recommendation

FAIL — The population Turing test is a valuable metric, but cannot pass with only 2 clusters from mock data. The failure is in segmentation coverage, not synthesis quality.

**Re-test when:**
- Segmentation produces >=5 diverse clusters aligned to the product's actual user base
- Using production data with genuine behavioral diversity across roles
- Comparing a baseline PersonaV1 set against a PersonaV1 + batch-3 schema additions set

## Cost

- Total API cost: ~$0.13 (4 synthesis + 20 judge calls on Haiku 4.5)
- Model: `claude-haiku-4-5-20251001`

## Artifacts

- `summary.json` — baseline vs treatment metrics
- `personas.json` — both persona sets
- `judgments.json` — all 20 judge responses with rationales
