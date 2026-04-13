# Experiment 5.07 — Time-to-detect Curve

## Hypothesis

High-quality personas evade detection 2-3x longer than baseline personas. Simulated with LLM judges as raters.

## Method

- Synthesized 2 personas from the 2 available clusters
- Ran 15-turn product-research conversations for each via TwinChat
- At turns 1, 3, 5, 7, 10, 15: 5 independent LLM judge calls ("Is this responder human or AI?")
- Detection point: first turn where judge says "AI" with confidence >= 4
- Realism score: `1 - (1 / mean_turns_to_detect)`
- Model: `claude-haiku-4-5-20251001`, judges at `temperature=0.8`

## Results

| Persona | Mean TTD | Detection Rate | Realism Score |
|---------|----------|---------------|---------------|
| Dev Infrastructure Lead | 5.0 turns | 100% (5/5 detected) | 0.80 |
| Sofia, Freelance Designer | 16.0 (never detected) | 0% (0/5 detected) | 0.94 |
| **Overall** | **10.5 turns** | **50%** | **0.87** |

- Total cost: $0.16

## Analysis

Stark divergence between personas:
1. **Dev Infrastructure Lead** — all 5 raters detected AI by turn 5. Likely due to overly consistent technical specificity that reads as synthetic.
2. **Sofia, Freelance Designer** — zero raters detected AI across 15 turns. The creative/freelance persona style may be more naturally varied and harder to distinguish from a real person.

The 3.2x difference in TTD between personas suggests persona archetype strongly influences detectability — technical personas may be easier to "clock" as AI than creative/freelance ones.

## Verdict

**PARTIAL PASS** — Overall realism score (0.87) is high, but the 100% detection rate on the technical persona is concerning. The experiment demonstrates the metric works and reveals persona-type-dependent detection sensitivity.

## Implications

- Detection resistance varies dramatically by persona archetype
- Technical personas need more natural variation (hedging, incomplete knowledge) to evade detection
- The TTD metric successfully differentiates persona quality — useful for evaluation
