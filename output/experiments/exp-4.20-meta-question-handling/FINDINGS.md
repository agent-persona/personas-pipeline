# Experiment 4.20: Meta-Question Handling

## Hypothesis
In-character acknowledgment of meta-questions should produce the highest realism without breaking character.

## Method
1. Generated 2 personas from `tenant_acme_corp`.
2. Ran 10 meta-questions against each persona under 3 prompt variants.
3. Classified each response as refusal, fabrication, or break-character.
4. Scored realism, in-character, and helpfulness on a 1-5 scale using the branch-local classifier.

- Provider: `anthropic`
- Model: `claude-haiku-4-5-20251001`

## Variant Metrics
- `deny`: realism `4.00`, in-character `4.70`, helpfulness `3.55`, break `10.0%`
- `deflect`: realism `3.65`, in-character `4.20`, helpfulness `3.15`, break `25.0%`
- `acknowledge`: realism `2.65`, in-character `2.40`, helpfulness `2.55`, break `70.0%`

## Best Variant
- Best composite: `deny`
- Best realism: `deny`

## Decision
Reject the acknowledgment variant. `deny` had the highest realism (`4.00`),
the strongest in-character score (`4.70`), and the lowest break rate (`10%`),
while `acknowledge` broke character in `70%` of trials.

## Caveat
Heuristic classification fallback is used when the model path is unavailable; keep that in mind when reading the break/fabrication split.
