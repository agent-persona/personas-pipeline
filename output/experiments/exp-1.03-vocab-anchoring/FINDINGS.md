# Experiment 1.3: Vocabulary anchoring ablation

## Metadata
- **Branch**: exp-1.03-vocab-anchoring
- **Date**: 2026-04-10
- **Problem Space**: 1 (Persona representation & schema)
- **Run ID**: 27b9eab7
- **Result JSON**: `output/exp_1_3_27b9eab7.json`

## Hypothesis
Removing `vocabulary` and `sample_quotes` from the synthesized persona
collapses style fidelity in twin replies even when factual content
(goals, pains, demographics) stays identical.

## Changes Made
- `synthesis/synthesis/engine/prompt_builder.py`: factored `SYSTEM_PROMPT`
  into `build_system_prompt(*, strip_vocabulary, strip_quotes)`; added
  the same flags to `build_tool_definition`, `build_messages`,
  `build_retry_messages`. Flag-off zero-diff verified.
- `synthesis/synthesis/engine/synthesizer.py`: threaded the flags through
  `synthesize()`; builds a per-call `PersonaV1Ablated` subclass via
  `pydantic.create_model` to relax `min_length` on the two fields.
  `PersonaV1` itself is untouched.
- `twin/twin/chat.py`: guarded the "How you talk" section so empty
  vocab/quotes drop the header entirely instead of leaving a dangling
  `"You use words like: ."`.
- `evaluation/evaluation/metrics.py`: new `stylometric_cosine` (char-wb
  3-4 gram TF-IDF cosine) and `pairing_accuracy` (async; delegates to
  `LLMJudge.same_speaker`).
- `evaluation/evaluation/judges.py`: added `LLMJudge.same_speaker(a, b)`
  — minimal yes/no prompt against a direct `AsyncAnthropic` handle.
- `evaluation/pyproject.toml`: added `scikit-learn>=1.4`.
- `scripts/exp_1_3_vocabulary_anchoring.py`: new runner. Fetches one
  fixed cluster, runs control + ablation synthesis, probes each twin
  with 15 questions (in-domain / social / adversarial), computes
  metrics, writes the JSON result file.

## Setup
- **Cluster**: `clust_dd483c27dff0`, size 4 records.
- **Reference corpus**: Intercom verbatim quotes from the cluster's
  `sample_records`. **Only 1 verbatim landed in this cluster** — a
  major limiter, see caveats.
- **Probe set**: 15 questions (5 in-domain, 5 social, 5 adversarial).
- **Twin model**: `claude-haiku-4-5-20251001`.
- **Judge model**: `claude-haiku-4-5-20251001` (matches twin → subject
  to self-preference bias).

## Results

### Target Metric: stylometric_cosine vs reference voice corpus

| Condition | stylometric_cosine | synthesis cost | twin cost (15 probes) |
|---|---|---|---|
| Control | **0.2578** | $0.0542 | $0.0227 |
| Ablation | **0.2333** | $0.0217 | $0.0204 |
| **Delta** | **+0.0245 (control)** | | |

### Secondary Metric: pairing_accuracy (Haiku judge)

| | value |
|---|---|
| pairs judged | 6 (2 same-condition, 2 cross-condition, 2 wiki distractors) |
| accuracy | **0.333** |

### Personas produced

| | Control | Ablation |
|---|---|---|
| name | Aiden — The Infrastructure Automation Architect | Dmitri, the Platform Architect |
| vocabulary length | 12 | 0 |
| sample_quotes length | 4 | 0 |
| attempts | 1 | 2 (validation fail → groundedness fail path observed) |

### Qualitative observations
- **Direction matches the hypothesis**: control is closer to the
  reference voice than the ablation by ~0.025 on stylometric cosine.
- **Gap is small and the absolute values are low** (~0.25), which is
  expected given the reference corpus is a single sentence. The cosine
  is essentially measuring character-n-gram overlap between 15 twin
  replies and one Intercom message; there is not enough reference text
  to form a stable TF-IDF distribution.
- **Ablation required a retry during synthesis** — one attempt failed
  validation, another failed groundedness (score 0.50). This is a
  secondary signal that removing the voice anchor fields makes the
  model's output shape less stable, not just stylistically blander.
  Worth reproducing on more clusters before drawing a conclusion.
- **pairing_accuracy = 0.333 is uninterpretable** as a voice-fidelity
  signal in this run. With both twin and judge on Haiku, and 4 of 6
  pairs labeled "same", the self-preference bias dominates. Useful
  only as a sanity check that the judge is answering at all.

## Signal Strength: **NOISE-ADJACENT**

The cosine delta (+0.025) is in the right direction but the reference
corpus is too thin to distinguish a real effect from n-gram noise. The
ablation's retry-on-synthesis behavior is an interesting secondary
observation but has n=1.

## Recommendation: **defer**

Do not adopt or reject the hypothesis from this run. The experimental
setup needs two fixes before the headline question can be answered:

1. **Widen the reference corpus to ≥10 Intercom verbatims.** Either
   pool verbatims across clusters, relax segmentation so the chosen
   cluster has more free-text records, or preload a per-tenant
   verbatim pool that every condition is measured against.
2. **Re-run `pairing_accuracy` with a cross-model judge** (Sonnet or
   Opus) on the same gold pairs to decompose self-preference bias
   from genuine voice-discrimination signal.

Additional follow-ups if the rerun confirms the direction:
- Split the ablation: `strip_vocabulary` alone vs `strip_quotes` alone,
  to attribute the effect between the two fields.
- Repeat across ≥3 clusters for variance bars.
- Investigate the retry-on-synthesis finding separately — is the
  ablated schema genuinely harder for the model to produce well?

## Cost
- Control synthesis: $0.0542
- Ablation synthesis: $0.0217
- Control twin (15 probes): $0.0227
- Ablation twin (15 probes): $0.0204
- Judge (6 pair calls): ~$0.000 (uncounted — Haiku, very small prompts)
- **Total: $0.1190** (well under the $2 abort cap and the $0.50 target)
