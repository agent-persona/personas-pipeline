# exp-1.15 — Edge-Case Behavior Fields

**Branch:** `exp-1.15-edge-case-behavior-fields`
**Guide:** Guide 2 — Synthesis Pipeline Architecture
**Date:** 2026-04-11
**Status:** MIXED / CONDITIONAL — strong signal on matched probes (+0.20 over baseline, perfect score), but −0.57 on unmatched probes due to over-fit. Net effect at n=20 judgments: slightly negative.

## Hypothesis

From Guide 2: add explicit schema fields capturing responses under emotional stress, measured as *"twin handling of adversarial conversational turns, robustness of voice under provocation."*

Concrete prediction: personas with explicit `edge_case_behaviors` recover from provocation ≥20pp more often than personas without, because the model has a pre-committed reaction to match rather than an ad-hoc improvisation.

## Method

### Schema change

Added `EdgeCaseBehavior` model (`{trigger, reaction, tone_shift}`) and a new `PersonaV1WithEdgeCases(PersonaV1)` subclass with `edge_case_behaviors: list[EdgeCaseBehavior]` required (min 3, max 6 entries). The field is **not** groundedness-required — it's a downstream inference from personality traits, not from cluster evidence.

### Prompt + twin plumbing

- `SYSTEM_PROMPT` extended with a **Robust** quality criterion conditional on the schema having the field.
- `twin/twin/chat.py:build_persona_system_prompt` renders a new `## How you react under pressure` section listing each declared trigger → reaction → tone_shift entry, so the twin has pre-committed behaviors to match at runtime.

### A/B design

Same two mock clusters as every batch-5 experiment (engineers, designers). For each cluster, synthesize under both `PersonaV1` (baseline) and `PersonaV1WithEdgeCases` (treatment). Then fire **10 adversarial probes** (rudeness, false premises, unsolicited advice, moralizing, off-topic pivots — see `scripts/run_exp_1_15.py:ADVERSARIAL_PROBES`) at each of the 4 personas via `TwinChat`.

Judge each reply with Claude-as-judge (Haiku) on:
- `in_character_score` (1–5): stayed in persona under pressure
- `used_named_reaction` (bool): reply pattern-matched one of the persona's declared edge-case reactions (trivially False for baseline)

Total judgments: 4 personas × 10 probes = 40.

## Results

### Aggregate

| Metric | Baseline (PersonaV1) | Treatment (WithEdgeCases) | Δ |
|---|---|---|---|
| Groundedness mean | 1.00 | 1.00 | 0 |
| Synthesis success rate | 2/2 | 2/2 | 0 |
| `mean_in_character` | **4.80** | **4.50** | **−0.30** |
| `used_named_rate` | 0.00 (n/a) | 0.35 | +0.35 |
| Total cost (USD) | ~$0.141 | ~$0.154 | +$0.013 |

### Split by whether the probe matched a declared edge-case

**This is the finding.**

| Probe class | n | mean in_character |
|---|---|---|
| Baseline (all probes) | 20 | **4.80** |
| Treatment — probe *matched* a declared reaction | 7 | **5.00** (perfect) |
| Treatment — probe did *not* match any declared reaction | 13 | **4.23** |

When the probe matched a listed trigger, treatment produced a **perfect** in-character response (+0.20 over baseline). When the probe did *not* match a listed trigger, treatment scored **−0.57 below baseline**.

### Per-persona

| Persona | Schema | `mean_in_character` | `used_named_rate` | Declared edge-cases |
|---|---|---|---|---|
| DevOps Architect | PersonaV1 | 4.70 | — | 0 |
| Sarah – Solo Brand Designer | PersonaV1 | 4.90 | — | 0 |
| Marcus, Infrastructure-First DevOps | WithEdgeCases | 4.30 | 0.10 | 4 (very domain-specific) |
| Maya, Autonomy-Driven Brand Designer | WithEdgeCases | 4.70 | 0.60 | 5 (broader triggers) |

Maya's declared reactions (false-premise automation, "just hire someone", unsolicited workflow advice, moralizing, vague time-savings claims) happened to overlap with the generic probe set 6/10 times. Marcus's reactions were much more domain-specific ("unsolicited suggestion to use UI dashboard instead of API", "GraphQL schema stability") and only matched 1/10 probes. Marcus's overall score dropped to 4.30 because the un-matched 9 probes landed on generic responses that, without the freedom to improvise that baseline has, were judged slightly stiffer.

## Interpretation

The field **works when it hits** — perfect scores on matched probes confirm that a pre-committed reaction does improve in-character adherence. But it also creates an **over-fit cost**: the persona's capacity to generalize to novel provocations drops because the runtime prompt now contains a structured schema of "expected" provocations that crowds out ad-hoc reasoning.

Net effect on a *generic* adversarial probe set: slightly negative (−0.30).
Net effect if the probe distribution is known at synthesis time: substantially positive (+0.20 on matched probes, which would likely be the majority in production).

Two additional factors capping the signal:

1. **Ceiling effect.** Baseline PersonaV1 is already very strong at adversarial handling (4.80/5). The experiment has ~0.20 of headroom before hitting the 5.0 ceiling, which is within judge variance at n=10 probes per persona.
2. **Probe–trigger domain mismatch.** My 10 probes were deliberately generic so I could run them against any persona. The treatment personas declared very *domain-specific* triggers (especially Marcus). A fair re-test would generate domain-matched probes per persona, but then the experiment cherry-picks its own wins.

## Recommendation

**DEFER** adoption to schema v1.1 as a **required** field. **CONSIDER** as an *optional* field for use cases with a known probe distribution (e.g., sales-objection-handling bots where you know the heckler script in advance).

Re-run criteria to move to ADOPT:

1. **Domain-matched probe set.** Generate probes directly from each persona's declared triggers; measure whether matched-probe performance (+0.20 observed) generalizes beyond this run.
2. **Calibrated judge.** The −0.30 aggregate is well within Claude-as-judge noise at n=40 judgments. Redo after exp-5.06 ships so we know Claude↔human agreement on "in-character under provocation" specifically.
3. **Over-fit mitigation.** Test whether adding a twin-runtime instruction like *"these are illustrative examples, not an exhaustive list — react in this character's spirit for novel provocations too"* recovers the lost 0.57 on unmatched probes without killing the +0.20 on matched ones.
4. **Larger sample.** n=2 personas × n=10 probes is underpowered. Need ≥8 personas × ≥20 probes before claiming any delta.

**Do NOT add `edge_case_behaviors` to PersonaV1 as a required field yet.** The current evidence is consistent with "over-fits on anticipated provocations at the cost of unanticipated ones," which is the wrong direction for a general-purpose persona schema.

## Qualitative note — synthesis is producing real edge-cases

Despite the mixed quantitative result, the declared edge-case behaviors themselves are high-quality and persona-specific. Example from Maya:

> *Suggestion that she should 'just hire someone' to handle asset management or client communication → Shuts down the conversation politely but firmly, reiterating her solo business model and her deliberate choice to stay independent.*

This is the kind of behavior a product researcher would want to capture. The synthesis stage is working; the failure is at the twin-runtime generalization layer.

## Cost

- Synthesis (both schemas × both clusters): ~$0.173
- Twin probes (40 total): ~$0.066
- LLM-judge (40 judgments): ~$0.055
- **Total: ~$0.294**
- Model: `claude-haiku-4-5-20251001` (both synthesis, twin, and judge)

## Artifacts

- `baseline_scored.json` — 2 baseline personas with 10 probes + judge each
- `treatment_scored.json` — 2 treatment personas with 10 probes + judge each, plus declared edge_case_behaviors
- `summary.json` — aggregate metrics
- `scripts/run_exp_1_15.py` (in branch)
- `synthesis/synthesis/models/persona.py` (in branch) — `EdgeCaseBehavior`, `PersonaV1WithEdgeCases`
- `synthesis/synthesis/engine/{prompt_builder,synthesizer}.py` (in branch) — schema_cls plumbing + Robust criterion
- `twin/twin/chat.py` (in branch) — `## How you react under pressure` section
