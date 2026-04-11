# Experiment 5.11: Reference-Based vs Reference-Free Judging

## Status: Complete

## Hypothesis

Providing a proxy reference persona alongside the standard rubric will:
1. Reduce score variance (tighter distribution)
2. Preserve rank ordering (high Spearman correlation with free mode)
3. Possibly introduce anchoring bias (scores clustering near reference quality)

## Design

| Dimension | Detail |
|-----------|--------|
| Golden tenant | `tenant_acme_corp` (2 clusters) |
| Synthesis repeats | 3 per cluster = 6 persona samples |
| Judge model | `claude-sonnet-4-20250514` |
| Modes | FREE (rubric only) vs REFERENCE (rubric + proxy anchor) |
| Proxy quality | Hand-crafted ~4.5/5 DevOps persona (clearly labeled as proxy) |

### Modes

- **FREE mode**: Standard 5-dimension rubric (grounded, distinctive, coherent, actionable, voice_fidelity). No reference provided. This is the current default judge behavior.
- **REFERENCE mode**: Same rubric, but with a proxy reference persona prepended as a calibration anchor. The prompt states the reference is approximately 4.5/5 quality and instructs scoring relative to it.

## Results

| Metric | Free Mode | Reference Mode |
|--------|-----------|----------------|
| Mean | 4.500 | 4.167 |
| Std | 0.548 | 0.408 |
| N | 6 | 6 |
| Scores | [4, 5, 5, 4, 4, 5] | [4, 5, 4, 4, 4, 4] |

| Comparison Metric | Value |
|-------------------|-------|
| Variance reduction ratio | 0.444 (44% reduction) |
| Mean delta (ref - free) | -0.333 |
| Spearman rho | 0.657 |
| Anchoring detected | Yes |
| Anchoring evidence | 6/6 (100%) scores within 0.5 of anchor (4.5) |

### Key Observations

1. **Variance reduction confirmed**: Reference mode reduced variance by 44% (std 0.548 -> 0.408). The tighter distribution suggests the reference anchor constrains the judge's scoring range.

2. **Anchoring bias is strong**: 100% of reference-mode scores fell within 0.5 of the proxy's declared quality (4.5). This is a clear anchoring effect — the judge treats the reference as a ceiling/floor rather than purely informational.

3. **Rank correlation moderate**: Spearman rho = 0.657 indicates the two modes partially agree on rank ordering, but the reference mode flattens distinctions. Personas scored 5.0 in free mode were pulled down to 4.0 in reference mode.

4. **Mean shift downward**: Reference mode scored 0.33 points lower on average. The proxy anchor (declared 4.5) appears to create a soft ceiling effect.

5. **Sample size caveat**: N=6 is small. Results are directionally informative but not statistically conclusive. Would need N>=20 for robust significance testing.

## Interpretation

The reference-based approach achieves its intended goal (variance reduction) but at the cost of introducing anchoring bias. The judge anchors heavily on the proxy quality rather than using it as a relative benchmark. This makes the reference mode unsuitable as a drop-in replacement for free mode.

## Decision

**Reject** — Reference mode introduces unacceptable anchoring bias (100% clustering). The 44% variance reduction is promising but the rank-order distortion (rho=0.657) means high-quality personas get penalized. The standard rubric-only (free) mode with few-shot calibration (exp 5.13) remains the better approach.

### Potential follow-ups
- Test reference mode WITHOUT declaring the anchor's quality score
- Provide multiple references at different quality levels (1, 3, 5) instead of one
- Combine reference mode with explicit "ignore the reference's absolute score" instructions
