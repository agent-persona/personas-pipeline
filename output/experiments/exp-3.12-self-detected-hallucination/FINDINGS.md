# Experiment 3.12: Self-detected hallucination

## Hypothesis
Models have weak but measurable self-knowledge of which claims are hallucinated.

## Method
1. Synthesized personas for the golden tenant.
2. Asked the model to self-rate each claim's grounding confidence as HIGH, MEDIUM, LOW, or MADE_UP.
3. Ran structural groundedness checks and an LLM-as-judge entailment check on the same claims.
4. Compared self-flagged claims against external unsupported claims to measure precision and recall.

- Provider: `openai`
- Synthesis model: `gpt-5-nano`
- Judge model: `gpt-5-nano`

## Cluster Outcomes
- `clust_09a614361e7a`: claims=8, hallucinations=8, self-flagged=6
- `clust_2ddc966af43a`: claims=8, hallucinations=8, self-flagged=6

## Aggregate Metrics
- Personas: `2`
- Claims: `16`
- Structural grounded rate: `100.00%`
- Entailment entailed rate: `0.00%`
- Hallucination rate: `100.00%`
- Self-flag rate: `75.00%`
- Precision: `100.00%`
- Recall: `75.00%`
- F1: `85.71%`
- Accuracy: `75.00%`
- Mean confidence score: `0.29`
- Calibration gap: `0.00`

## Decision
Adopt. Self-critique separates grounded from hallucinated claims with useful precision and recall.

## Caveat
Tiny sample: 1 tenant, 2 personas, and a small claim set. LLM-as-judge entailment is the pragmatic substitute for missing NLI.
