# Experiment 5.05 — Rubric Ablation

## Status: HARNESS READY, NOT YET RUN

API key not available at experiment time. All code is implemented and tested.

## Hypothesis

Some of the five rubric dimensions (grounded, distinctive, coherent, actionable, voice_fidelity) are redundant (pairwise correlation > 0.95) or inert (removing them does not change persona rankings). Identifying these would allow simplifying the rubric without losing discriminative power.

## Method

1. **Persona generation**: Synthesize personas from `tenant_acme_corp` with 3 repeats per cluster to build sufficient sample size (~6 personas from ~2 clusters).
2. **Control scoring**: Score each persona with the full 5-dimension rubric (grounded, distinctive, coherent, actionable, voice_fidelity) using the LLM judge.
3. **Ablation**: For each dimension, build a 4-dimension rubric excluding that dimension. Re-score all personas.
4. **Analysis**:
   - Pairwise Pearson correlation between all dimension pairs from control scores
   - Ranking stability: Kendall tau (or simpler delta-based metric if n < 3) comparing control vs ablated overall rankings
   - Score shift: Mean delta in surviving dimensions when one dimension is removed
5. **Classification**:
   - **Redundant**: Any pair with Pearson r > 0.95
   - **Inert**: Any dimension whose removal yields Kendall tau > 0.95 with control rankings

## Design Decisions

- **Multiple synthesis repeats**: `tenant_acme_corp` produces ~2 clusters. 3 repeats gives ~6 personas — enough for rank correlation but not for robust statistical tests. This is noted as a limitation.
- **Haiku for judging**: Uses the default model (haiku) for cost efficiency. The rubric ablation tests whether *removing a dimension changes relative scores*, not whether scores are calibrated to human labels — so a cheaper model suffices.
- **Separate system prompts per variant**: Each ablated rubric is a distinct system prompt that only mentions the surviving dimensions. The judge never sees the dropped dimension name, preventing information leakage.

## Deliverables

| File | Description |
|------|-------------|
| `evals/rubric_ablation.py` | Parameterized rubric builder + ablation harness |
| `scripts/experiment_5_05.py` | Runner script |
| `tests/test_exp_5_05.py` | 16 unit tests (all passing) |
| `output/experiments/exp-5.05-rubric-ablation/results.json` | Pending — needs API key |
| `output/experiments/exp-5.05-rubric-ablation/FINDINGS.md` | This file |

## Expected Outcomes

Based on rubric design intuition:
- **grounded** and **coherent** may correlate highly (both reward evidence-backed, self-consistent content)
- **voice_fidelity** is the most likely candidate for inertness — it scores a narrow aspect (quote consistency) that may not affect the overall ranking much
- **distinctive** and **actionable** likely have the lowest mutual correlation since they measure orthogonal qualities (personality vs utility)

## Recommendation

**DEFER** — Harness ready, results pending API key. Once run, update this file with actual correlation matrix, ranking stability scores, and adopt/reject decision.

## Test Results

```
16 passed in 0.03s
```

All rubric builder, parser, statistics, and analysis tests pass.
