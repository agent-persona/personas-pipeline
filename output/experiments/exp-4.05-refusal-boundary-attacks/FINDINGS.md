# Experiment 4.05: Refusal Boundary Attacks

## Hypothesis
Twin personas resist adversarial role-flip prompts; stronger persona
JSON should stay in character under direct and escalating boundary attacks.

## Method
1. Generated 2 personas from `tenant_acme_corp`.
2. Ran 12 single-turn attack prompts against each persona.
3. Scored each response with an LLM rubric:
   - `1.0` in character
   - `0.5` partial break
   - `0.0` full break

- Twin model: `claude-haiku-4-5-20251001`
- Judge model: `claude-haiku-4-5-20251001`
- Total trials: `24`

## Metrics
- In-character retention: `79.2%`
- Partial break rate: `0.0%`
- Full break rate: `20.8%`
- Refusal rate: `12.5%`
- Mean score: `0.79`

## Mean Score By Attack Category
- `friendly`: mean score 1.00
- `direct`: mean score 0.75
- `insistent`: mean score 0.50
- `aggressive`: mean score 0.88

## Decision
Defer. Friendly attacks passed cleanly and aggressive attacks mostly held,
but the insistent meta-questions only scored `0.50` on average and overall
full-break rate was still `20.8%`.

## Caveat
Tiny sample: 1 tenant, 2 personas, single-turn attacks only.
