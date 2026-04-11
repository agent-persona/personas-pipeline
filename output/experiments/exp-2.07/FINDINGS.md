# exp-2.07 — Order-of-Fields Effect

**Branch:** `exp-2.07-order-of-fields`
**Guide:** Guide 2 — Synthesis Pipeline Architecture
**Date:** 2026-04-11
**Status:** WEAK / DEFER — n=2 clusters, n=1 successful trial per condition. Loop proven; signal inconclusive.

## Hypothesis

From Guide 2: *"leading with vocabulary or sample quotes anchors voice"* better than demographics-first synthesis, yielding improved downstream character fidelity and reduced stereotyping through field sequencing alone.

Concrete prediction: voice-first field order produces ≥15% higher within-cluster distinctiveness and ≥10pp lower stereotyping rate, with no drop in groundedness.

## Method

Two Pydantic schemas with **identical field constraints, types, and descriptions** — only declaration order differs:

- **Baseline** — `PersonaV1` (demographics → firmographics → goals → pains → … → vocabulary → sample_quotes)
- **Treatment** — `PersonaV1VoiceFirst` (vocabulary → sample_quotes → demographics → firmographics → …)

Pydantic v2 preserves declaration order in `model_json_schema()`, and the Anthropic tool-use API fills fields in schema order during forced structured output. So schema reordering is the *only* independent variable — no prompt changes, no model changes, no temperature changes.

Both schemas run against the same 2 mock clusters (engineers, designers) in the same session via `synthesize(..., schema_cls=X)`. Backend: Haiku 4.5.

Metrics:
1. **Within-cluster vocab Jaccard** (baseline persona vs treatment persona for the *same* cluster) — lower = more different outputs, which is what the hypothesis predicts should hold across runs if order actually changes what gets generated
2. **Cross-cluster vocab Jaccard** within each condition — distinctiveness between the two personas generated under the same schema
3. **Stereotyping rate** — fraction of each persona's vocabulary hitting a 50-word generic-business-English stoplist (see `scripts/run_exp_2_07.py:GENERIC_STOPLIST`)
4. **Groundedness mean** — from existing structural `check_groundedness()`
5. **Success rate** — fraction of cluster syntheses that passed groundedness in ≤3 attempts

## Results

### Quantitative

| Metric | Baseline (PersonaV1) | Treatment (VoiceFirst) | Δ |
|---|---|---|---|
| Success rate | 2/2 (100%) | 2/2 (100%) | 0 |
| Groundedness mean | 0.95 | 1.00 | **+0.05** |
| Cross-cluster vocab Jaccard | 0.00 | 0.00 | 0 (saturated) |
| Stereotyping rate mean | 0.000 | 0.033 | **+0.033** |
| Total cost (USD) | $0.0783 | $0.0701 | **−$0.008** |
| Attempts mean | 2.5 | 2.0 | −0.5 |

### Within-cluster baseline-vs-treatment comparison

| Cluster | Baseline persona | Treatment persona | Vocab Jaccard |
|---|---|---|---|
| `clust_3d3483193a04` (engineer) | Marcus, the Infrastructure Integrator | Alex Chen, Infrastructure-First Engineering Lead | 0.095 |
| `clust_5eab59ba762e` (designer) | Maya, The Efficiency-Minded Freelance Designer | Sophia, the White-Label Project Operator | 0.136 |

The two conditions produced *substantially different* vocab lists (~90% different for engineers, ~86% different for designers), so field order *does* affect what gets written — this is not a no-op.

### Qualitative: money-quote stability

**Exactly one `sample_quotes` entry per cluster appeared verbatim in both conditions:**

- Engineer cluster:
  > *"Your REST API is solid but the GraphQL endpoint has some rough edges. Plans to improve the schema?"*

- Designer cluster:
  > *"I bill clients hourly so anything that saves me 10 minutes per project is worth real money."*

These are the *dominant quote-worthy inferences* from each cluster — the model lands on them regardless of whether it writes quotes first (voice-first) or last (baseline). This is the most interesting finding of the experiment and directly undermines the Guide 2 hypothesis that field order *anchors* voice: if voice were genuinely anchored by order, voice-first should have produced different money-quotes than demographic-first. Instead, the cluster data itself anchors the dominant quote, and field order only shuffles the *surrounding* vocab.

### Pre-experiment run instability

An earlier run of the runner (before I added `SynthesisError` handling) failed on the voice-first designer cluster with 3 consecutive groundedness-check failures (scores `0.48, 0.60, 0.71, 0.72` — none crossing the 0.90 threshold). The second run — same code, same model, same clusters — succeeded at groundedness 1.00 on both personas. **Voice-first appears to have higher variance in grounding quality than demographics-first,** possibly because committing to vocabulary and quotes before the model has "walked through" the evidence makes it harder to back-fill valid `source_evidence.field_path` entries for every goal/pain/motivation/objection. But n=2 runs is **not** enough to claim this confidently.

## Interpretation

**Loop works.** The schema-swap mechanism is the cleanest possible test of Guide 2's field-order hypothesis and I can now run it on any new fixture.

**Signal is weak / inconclusive:**

- **Distinctiveness:** Cross-cluster Jaccard is saturated at 0.0 because the two mock clusters (engineers vs designers) are maximally different domains. This metric is uninformative at n=2 with such orthogonal clusters. Real test needs ≥4 semantically-similar clusters.
- **Stereotyping:** Treatment introduced one stoplist word (`seamless` in the designer persona) that baseline avoided. Directionally *against* the hypothesis, but this is a single word out of ~25 on one run.
- **Groundedness:** Within this run, treatment was marginally better (+0.05). But the pre-run SynthesisError episode hints treatment has a higher *variance* failure mode. Need multi-run variance estimates to know.
- **Cost:** Treatment was 10% cheaper (fewer retries this run). Also high variance pending multi-run.

**Most important qualitative finding:** The money-quote stability result undermines the specific Guide 2 claim that *field order anchors voice*. If the cluster data has a dominant quote-worthy record, the model will surface it regardless of order. Voice-first may affect which *secondary* vocabulary gets emitted, but the headline voice is determined upstream by the evidence.

## Recommendation

**DEFER** schema v1.1 changes around field ordering until re-tested with:

1. **≥8 semantically-adjacent clusters** (not just orthogonal engineers-vs-designers) so cross-cluster Jaccard is not saturated at 0
2. **≥3 trials per condition** so variance in groundedness and stereotyping is measurable
3. **Calibrated judge** from exp-5.06 so the subjective "voice quality" claim in Guide 2 can be measured against human ratings

**Re-run this experiment after:**
- exp-5.06 ships (Track C of batch 5) and establishes judge↔human agreement
- A fixture expansion adds at least 6 additional mock clusters to `crawler/connectors/`

**Do NOT adopt voice-first ordering as default yet.** Current evidence is consistent with "field order shuffles vocab but doesn't change headline voice" — which is the null result for Guide 2's hypothesis.

## Cost

- Total API cost this experiment: **$0.149** (includes both baseline and treatment runs; excludes the aborted first run that hit `SynthesisError`)
- Model: `claude-haiku-4-5-20251001`

## Artifacts

- `baseline_personas.json` — 2 personas from PersonaV1
- `treatment_personas.json` — 2 personas from PersonaV1VoiceFirst
- `summary.json` — full metric dump
- `scripts/run_exp_2_07.py` (in branch) — reproducible runner
- `synthesis/synthesis/models/persona.py` (in branch) — new `PersonaV1VoiceFirst` class
- `synthesis/synthesis/engine/{prompt_builder,synthesizer}.py` (in branch) — `schema_cls` parameter plumbing
