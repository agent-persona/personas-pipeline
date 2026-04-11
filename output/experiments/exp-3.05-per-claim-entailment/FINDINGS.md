# Experiment 3.05: Per-Claim Entailment

## Hypothesis
LLM-as-judge entailment on individual persona claims is measurable and exposes unsupported claims more directly than structural groundedness alone.

## Method
1. Generated personas for each golden-tenant cluster.
2. Extracted every claim from `goals`, `pains`, `motivations`, and `objections`.
3. Sent each claim and its cited source records to a branch-local judge helper.
4. Aggregated entailment, neutral, and contradiction rates and compared them to structural groundedness.

- Tenant: `tenant_acme_corp`
- Provider: `anthropic->openai`
- Judge provider: `anthropic->openai`
- Synthesis model: `heuristic`
- Judge model: `claude-sonnet-4-20250514`
- Claims evaluated: `14`

## Persona Results
- `Api Setup Owner`: structural `1.00`, entailed `0.00`, neutral `1.00`, contradicted `0.00`
- `Template Browsing Owner`: structural `1.00`, entailed `0.00`, neutral `1.00`, contradicted `0.00`

## Aggregate Metrics
- Mean structural groundedness: `1.00`
- Mean claim entailment rate: `0.00`
- Mean neutral rate: `1.00`
- Mean contradiction rate: `0.00`
- Mean false-positive grounding rate: `1.00`
- Persona-level groundedness/entailment correlation: `undefined`

## Decision
Adopt.

## Caveat
Tiny sample: 1 tenant and only the golden clusters. The correlation is descriptive, not a stable estimate.
