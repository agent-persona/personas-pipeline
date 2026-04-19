# Experiment: Humanization A/B

## Hypothesis
Humanized personas (with backstory, speech patterns, emotional triggers)
produce twin chat replies that read more like real humans than baseline personas.

## Method
1. Shared ingest + segmentation on tenant_acme_corp
2. Baseline: synthesize v1 -> score with default judge -> twin chat -> score replies
3. Humanized: synthesize v2 -> score with humanized judge -> twin chat (humanized prompt) -> score replies
4. Compare per-stage scores and twin reply humanness

- Synthesis model: claude-haiku-4-5-20251001
- Judge model: claude-sonnet-4-20250514

## Persona Score Comparison

```
Stage      Dimension       Baseline  Humanized     Delta
--------------------------------------------------------
Persona 0  grounded            5.00      5.00     +0.00
           distinctive         5.00      5.00     +0.00
           coherent            5.00      5.00     +0.00
           actionable          5.00      5.00     +0.00
           voice_fidelity      5.00      5.00     +0.00

Persona 1  grounded            5.00      5.00     +0.00
           distinctive         5.00      5.00     +0.00
           coherent            5.00      5.00     +0.00
           actionable          5.00      5.00     +0.00
           voice_fidelity      5.00      5.00     +0.00

```

## Twin Reply Humanness Comparison

```
Stage   Dimension               Baseline  Humanized     Delta
-------------------------------------------------------------
Twin 0  discourse_markers           2.67      4.67     +2.00
        hedging                     2.33      3.67     +1.33
        specificity                 4.67      4.67     +0.00
        sentence_variety            3.00      4.67     +1.67
        emotional_authenticity      2.67      4.00     +1.33
        overall                     3.33      4.67     +1.33

Twin 1  discourse_markers           3.00      2.33     -0.67
        hedging                     2.67      2.00     -0.67
        specificity                 3.67      2.00     -1.67
        sentence_variety            3.33      2.33     -1.00
        emotional_authenticity      3.00      2.33     -0.67
        overall                     3.33      2.33     -1.00

```

## Per-Persona Details

### The Platform Engineer

- Persona score: 5.00 -> 5.00 (delta: +0.00)
- Twin humanness: 3.33 -> 4.67 (delta: +1.33)

### The Independent Visual Consultant

- Persona score: 5.00 -> 5.00 (delta: +0.00)
- Twin humanness: 3.33 -> 2.33 (delta: -1.00)

## Interpretation

TBD after reviewing results.

## Decision

TBD — adopt / reject / iterate.
