# Experiment 2.12: Self-Consistency Voting

## Hypothesis
Majority voting across multiple persona samples reduces hallucinated or unstable items without flattening persona richness.

## Method
1. Generated `2` control personas on the golden tenant.
2. Generated `3` stochastic samples per cluster at `temperature=0.7`.
3. Voted list fields by majority support across samples and kept structural metadata from the strongest sample.
4. Compared control vs best single sample vs voted persona using judge score and groundedness.

- Provider: `anthropic->openai`
- Synthesis model: `claude-haiku-4-5-20251001`
- Judge model: `claude-sonnet-4-20250514`

## Aggregate Metrics
- Mean control groundedness: `1.00`
- Mean best-sample groundedness: `1.00`
- Mean voted groundedness: `1.00`
- Mean control judge score: `4.00`
- Mean best-sample judge score: `4.50`
- Mean voted judge score: `4.50`
- Mean content richness: `19.0`
- Mean vote support ratio: `1.00`

## Decision
Adopt. Voting improved or matched both judge score and groundedness.
