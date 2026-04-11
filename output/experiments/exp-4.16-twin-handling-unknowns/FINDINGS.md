# Experiment 4.16: Twin Handling Unknowns

## Hypothesis
Twins should refuse or defer unknown questions in character instead of fabricating facts or breaking character.

## Method
1. Synthesized `2` personas from the golden tenant.
2. Ran `16` unknown or out-of-scope questions across `2` prompt variants.
3. Classified each response as refusal, fabrication, or break-character using a branch-local heuristic classifier.

- Provider: `anthropic->openai`
- Synthesis model: `claude-haiku-4-5-20251001`
- Twin model: `claude-haiku-4-5-20251001`

## Variant Metrics
- `baseline`: refusal `0.0%`, fabrication `93.8%`, break `6.2%`, mean response length `318` chars
- `refusal`: refusal `0.0%`, fabrication `93.8%`, break `6.2%`, mean response length `268` chars

## Decision
Defer. The prompt variant did not clearly improve refusal behavior enough to justify the change.

## Caveat
This run uses a small stub tenant with only two clusters, so the rates are directional rather than definitive.
