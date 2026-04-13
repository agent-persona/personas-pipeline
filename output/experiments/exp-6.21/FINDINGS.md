# Experiment 6.21 — Population-level Turing Test

## Hypothesis

A persona set passes the population Turing test >= 70% of the time — judges believe the set represents real customers.

## Method

- Synthesized personas from all 2 available clusters
- Presented the full set to 10 independent LLM judges (simulating human raters)
- Each judge answered: real/AI-generated/unsure, coverage % (0-100), authenticity (1-5)
- Model: `claude-haiku-4-5-20251001`, judges at `temperature=0.8`

## Results

| Metric | Value |
|--------|-------|
| Turing pass rate | **0%** (0/10 said "real") |
| AI-generated verdicts | 10/10 (100%) |
| Mean perceived coverage | 25% |
| Mean authenticity | 2.0/5 |
| Target pass rate | >= 70% |

- Total cost: $0.06

## Analysis

All 10 judges unanimously identified the set as AI-generated. The rationales reveal consistent failure modes:

1. **Persona-product misalignment:** "Creative Business Solopreneur" (a freelance designer) was universally flagged as fundamentally misaligned with a "project management tool for engineering teams." Every judge noted this disconnect.

2. **Missing core personas:** All judges noted the absence of engineering managers, tech leads, QA engineers, product managers — the actual users of engineering PM tools.

3. **Set size too small:** Only 2 personas for a B2B SaaS product was universally seen as insufficient. Real customer bases would have 4-6+ distinct segments.

4. **Individual quality vs set quality:** Judges noted Persona 2 (Devon, the infrastructure engineer) was individually credible, but the set as a whole failed because of composition issues.

## Verdict

**FAIL** — 0% pass rate vs 70% target.

The failure is primarily a **segmentation/coverage problem**, not a synthesis quality problem. The mock data produces only 2 clusters, yielding 2 personas — one of which (freelance designer) doesn't match the stated product. With production data producing 5+ diverse clusters aligned to the product's actual user base, this experiment would likely perform differently.

## Implications

- Population-level evaluation exposes coverage gaps that per-persona metrics miss entirely
- The segmentation pipeline needs more cluster diversity for credible persona sets
- Persona-product fit is a prerequisite for set-level authenticity
- This metric is valuable: judges gave specific, actionable feedback about what's missing
