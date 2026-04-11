# Experiment 1.07: Field Interdependence

## Hypothesis
Some persona fields are load-bearing (removing them degrades 2+ scoring
dimensions significantly) while others are decorative (minimal impact).

## Method
1. Generated a control persona via the standard pipeline (tenant_acme_corp)
2. For each of 9 ablatable fields, set it to [] in a copy
3. Re-scored each ablated copy with the LLM judge (few-shot calibrated)
4. Computed score deltas and classified fields

- Synthesis model: claude-haiku-4-5-20251001
- Judge model: claude-sonnet-4-20250514
- Calibration: few_shot

## Dependency Matrix

```
Field removed               grounded   distinctive      coherent    actionablevoice_fidelity       overall           class
--------------------------------------------------------------------------------------------------------------------------
(control).............           5.0           5.0           5.0           4.0           4.0           5.0             ---

goals.................          -1.0          -1.0          -1.0          +0.0          +0.0          -1.0    load-bearing
pains.................          +0.0          +0.0          -1.0          +1.0          +0.0          +0.0      decorative
motivations...........          +0.0          -1.0          +0.0          +0.0          +0.0          -1.0      decorative
objections............          +0.0          -1.0          +0.0          +0.0          +0.0          -1.0      decorative
channels..............          +0.0          +0.0          +0.0          +0.0          +0.0          +0.0      decorative
vocabulary............          +0.0          +0.0          +0.0          +0.0          +0.0          +0.0      decorative
decision_triggers.....          +0.0          -1.0          +0.0          +0.0          +0.0          -1.0      decorative
sample_quotes.........          +0.0          -1.0          +0.0          +0.0          -2.0          -1.0    load-bearing
journey_stages........          +0.0          +0.0          +0.0          +0.0          +0.0          +0.0      decorative

Load-bearing fields: goals, sample_quotes
Decorative fields:   pains, motivations, objections, channels, vocabulary, decision_triggers, journey_stages
```

## Classification

**Load-bearing fields** (2):
- `goals` — drops: grounded: -1.0, distinctive: -1.0, coherent: -1.0
- `sample_quotes` — drops: distinctive: -1.0, voice_fidelity: -2.0

**Decorative fields** (7):
- `pains` (overall delta: +0.0)
- `motivations` (overall delta: -1.0)
- `objections` (overall delta: -1.0)
- `channels` (overall delta: +0.0)
- `vocabulary` (overall delta: +0.0)
- `decision_triggers` (overall delta: -1.0)
- `journey_stages` (overall delta: +0.0)

## Interpretation

**Load-bearing fields (2/9):**

- `goals`: Removing goals degraded grounded (-1), distinctive (-1), and coherent (-1).
  Goals anchor the persona's identity and provide the coherence backbone that ties
  demographics, pains, and vocabulary together. Without goals, the judge saw the
  persona as less grounded and less differentiated.

- `sample_quotes`: Removing quotes hit distinctive (-1) and voice_fidelity (-2, the
  largest single drop). Quotes are the primary vehicle for voice — without them the
  judge has no signal to evaluate whether the persona sounds like a real person.

**Near-miss fields:**

- `motivations`, `objections`, `decision_triggers` each dropped distinctive by -1
  and overall by -1, but only hit a single dimension at the >= 0.5 threshold.
  These are "supporting" fields — individually removable without catastrophic
  degradation, but removing multiple simultaneously would likely compound.

**Truly decorative fields:**

- `channels`, `vocabulary`, `journey_stages` showed zero delta across all dimensions.
  The judge does not penalize their absence, suggesting these fields primarily serve
  downstream consumers (e.g., marketing channel selection, SEO) rather than persona
  quality per se.

- `pains` surprisingly showed no overall impact (coherent -1, but actionable +1).
  This may be because the judge's rubric evaluates goals and pains somewhat
  interchangeably under "actionable."

**Limitations:**

- Single persona, single run — results may vary across different persona archetypes
- Judge scores use integer 1-5 scale, so deltas are coarse-grained
- Interaction effects (removing 2+ fields simultaneously) not tested

## Decision

**Adopt** — The load-bearing classification is actionable:

1. `goals` and `sample_quotes` should be prioritized in synthesis retries and
   quality gates. A persona failing on these fields needs re-synthesis.
2. `channels`, `vocabulary`, and `journey_stages` can be deprioritized in
   cost-constrained scenarios (e.g., batch generation with tight token budgets).
3. Future work: test pairwise ablation to detect interaction effects.
