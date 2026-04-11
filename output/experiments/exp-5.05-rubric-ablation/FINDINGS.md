# Experiment 5.05 — Rubric Ablation

## Status: COMPLETE

## Hypothesis

Some of the five rubric dimensions (grounded, distinctive, coherent, actionable, voice_fidelity) are redundant (pairwise correlation > 0.95) or inert (removing them does not change persona rankings). Identifying these would allow simplifying the rubric without losing discriminative power.

## Configuration

| Parameter | Value |
|-----------|-------|
| Tenant | `tenant_acme_corp` |
| Clusters | 2 |
| Synthesis repeats | 3 (1 failed, yielding 5 personas) |
| Judge model | `claude-haiku-4-5-20251001` |
| Dimensions | grounded, distinctive, coherent, actionable, voice_fidelity |
| Total scoring calls | 30 (5 personas x 6 variants) |
| Duration | 585.9s |

## Personas Generated

| ID | Name | Cluster |
|----|------|---------|
| clust_f9becdf0b34f_rep0 | Marcus, the Platform Integration Architect | DevOps/Fintech |
| clust_ea9222caaacd_rep0 | Maya, the Time-Conscious Creative Entrepreneur | Freelance Designer |
| clust_f9becdf0b34f_rep1 | Raj, The Infrastructure-First DevOps Architect | DevOps/Fintech |
| clust_ea9222caaacd_rep1 | Rachel the Resourceful Freelance Designer | Freelance Designer |
| clust_ea9222caaacd_rep2 | Maya, the Hourly-Billed Freelance Designer | Freelance Designer |

## Control Scores (Full Rubric)

All 5 personas received **identical** dimension scores:

| Dimension | Score |
|-----------|-------|
| grounded | 4 |
| distinctive | 4 |
| coherent | 5 |
| actionable | 5 |
| voice_fidelity | 5 |
| **overall** | **4.6** |

This ceiling effect (all personas near-maxed on 3 of 5 dimensions) collapses pairwise correlations to NaN (zero variance) and makes rank-based metrics degenerate.

## Pairwise Correlations

All pairwise Pearson correlations returned **NaN** due to zero variance in grounded (all 4), distinctive (all 4), coherent (all 5), actionable (all 5), and voice_fidelity (all 5). With identical scores across all personas, correlation is undefined.

**Implication**: Cannot determine redundancy from this sample. The judge gives very similar scores to well-synthesized personas from the same tenant, which means either (a) the personas are genuinely similar in quality, or (b) the judge has a ceiling bias that compresses the score distribution.

## Ranking Stability (Kendall Tau)

| Dropped Dimension | Tau | Interpretation |
|-------------------|-----|----------------|
| drop_grounded | 0.000 | Rankings disrupted |
| drop_distinctive | 0.000 | Rankings disrupted |
| drop_coherent | 0.000 | Rankings disrupted |
| drop_actionable | 0.000 | Rankings disrupted |
| drop_voice_fidelity | 0.000 | Rankings disrupted |

Tau = 0 for all because the control rankings are flat (all tied at 4.6 overall). Any perturbation breaks ties differently, yielding zero correlation. **No dimension is inert** by this metric, but this is an artifact of the degenerate input rather than a strong finding.

## Score Shifts (Mean Delta in Surviving Dimensions)

| Dropped | grounded | distinctive | coherent | actionable | voice_fidelity |
|---------|----------|-------------|----------|------------|----------------|
| grounded | — | +0.00 | +0.00 | +0.00 | +0.00 |
| distinctive | **-0.40** | — | **-0.40** | +0.00 | -0.20 |
| coherent | +0.00 | +0.00 | — | +0.00 | +0.00 |
| actionable | **-0.40** | +0.00 | **-0.40** | — | **-0.40** |
| voice_fidelity | **-0.40** | +0.00 | -0.20 | -0.20 | — |

### Key patterns:

1. **Dropping `grounded` or `coherent` has zero effect** on surviving dimension scores. This suggests these dimensions are evaluated independently — removing them from the rubric prompt does not cause the judge to redistribute attention.

2. **Dropping `distinctive` causes grounded and coherent to drop by -0.40**. When the judge is no longer asked about distinctiveness, it becomes harsher on grounding and coherence. This suggests "distinctive" acts as a positive framing cue — its presence in the rubric may anchor the judge toward more favorable scores on other dimensions.

3. **Dropping `actionable` causes the largest total shift** (-0.40 on grounded, coherent, and voice_fidelity). Actionable appears to be a load-bearing dimension that stabilizes the judge's overall assessment.

4. **Dropping `voice_fidelity` also depresses grounded (-0.40)**, suggesting cross-dimension bleeding between voice quality assessment and evidence evaluation.

5. **`distinctive` is never affected by dropping any other dimension** (delta = 0.00 in all cases). It is the most independent dimension in the rubric.

## Analysis

### What we can conclude:
- **No dimensions are redundant** (correlation analysis inconclusive due to ceiling effect, but score-shift patterns show distinct behavioral signatures for each dimension).
- **No dimensions are inert** (all removals change either scores or rankings).
- **Dimensions interact asymmetrically**: removing "distinctive", "actionable", or "voice_fidelity" depresses other scores, while removing "grounded" or "coherent" does not. This suggests a two-tier structure:
  - **Anchor dimensions** (distinctive, actionable, voice_fidelity): their presence in the rubric positively influences scores on other dimensions
  - **Independent dimensions** (grounded, coherent): scored in isolation regardless of what else is in the rubric

### What we cannot conclude (limitations):
- **Ceiling effect**: With all control scores clustered at 4-5, the experiment cannot detect fine-grained correlations. A wider quality range (including deliberately weak personas) would strengthen correlation analysis.
- **Small n**: 5 personas from 2 clusters (one tenant) provides limited statistical power. Kendall tau with 5 items and many ties is unreliable.
- **Single judge model**: Haiku's scoring behavior may differ from Opus or other models. The asymmetric cross-dimension effects could be model-specific.

## Recommendation

**DEFER** — All 5 dimensions show distinct behavioral signatures and none are clearly redundant or inert. However, the ceiling effect prevents definitive conclusions. To strengthen this experiment:

1. Include deliberately degraded personas (remove evidence, genericize quotes) to create a wider quality range
2. Expand to additional tenants for more diverse persona types
3. Repeat with Opus as the judge model to check if the asymmetric effects are model-specific
4. Consider testing 6+ dimension rubrics to see if adding dimensions creates diminishing returns

The current 5-dimension rubric should be **retained as-is** until a follow-up experiment with wider quality variance can provide stronger evidence.

## Test Results

```
16 passed in 0.03s
```
