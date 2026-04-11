# Experiment 3.16: Synthetic Ground Truth Injection

## Hypothesis
Known planted facts should survive synthesis if the pipeline is truly grounded.

## Method
1. Loaded `38` mock tenant records for `tenant_acme_corp`.
2. Injected `6` distinctive facts into payloads without changing clustering features.
3. Ran the pipeline on control and injected fixtures with the same cluster topology.
4. Scored survival by matching planted fact signals against persona text and source evidence.

- Provider: `anthropic->openai`
- Synthesis model: `claude-haiku-4-5-20251001->gpt-5-nano`

## Cluster Comparison
- `clust_363c33ff2fdb`: control `1.00` -> injected `1.00`, survival `0.33`
- `clust_cda5d313efe3`: control `1.00` -> injected `1.00`, survival `0.33`

## Fact Survival
- `engineer_csv_board_exports` on `intercom_002`: control `False` / injected `False` (evidence `False`)
- `engineer_slack_webhook_alerts` on `ga4_006`: control `False` / injected `True` (evidence `False`)
- `engineer_ex_google_vp` on `hubspot_003`: control `False` / injected `False` (evidence `False`)
- `designer_white_label_share` on `intercom_004`: control `True` / injected `True` (evidence `True`)
- `designer_presentation_mode` on `intercom_005`: control `False` / injected `False` (evidence `False`)
- `designer_unlimited_share_links` on `intercom_007`: control `False` / injected `False` (evidence `False`)

## Summary
- Control survival rate: `0.17`
- Injected survival rate: `0.33`
- Mean control groundedness: `1.00`
- Mean injected groundedness: `1.00`
- Mean control cost: `$0.0000`
- Mean injected cost: `$0.0000`

## Decision
Defer. The injected facts were not retained strongly enough to justify calling the pipeline robust.

## Caveat
Small sample: 1 tenant, 2 clusters, and synthetic facts were intentionally distinctive. This is a retention probe, not a general generalization benchmark.
