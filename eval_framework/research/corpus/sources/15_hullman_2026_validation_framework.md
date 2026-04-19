# Hullman et al. (2026) — Validating LLM Simulations as Behavioral Evidence

**URL:** https://arxiv.org/abs/2602.15785
**Venue:** arXiv, February 2026 (Northwestern)

## Why This Is The Most Important Methodological Paper

Provides a formal statistical framework for WHEN LLM persona simulations support valid inference about human behavior. Answers the question every other paper dances around: under what conditions can you trust synthetic persona data?

## Three Validation Strategies
1. **Heuristic/Validate-Then-Simulate:** Run LLM on known tasks, check results, then use on new tasks
2. **Simulate-Then-Validate:** Generate synthetic data, then spot-check with humans
3. **Statistical Calibration via PPI:** Prediction Powered Inference — combine small human sample + large LLM sample with formal guarantees

## Key Results
- Heuristic validation: 81% main effect replication BUT **83% false positives on null findings** — looks like it works but accepts things it shouldn't
- Effect size correlation: r=0.5-0.85 with **systematic overestimation**
- 90% prediction accuracy can yield ~30% relative bias in regression coefficients
- PPI augmentation: effective sample size from 10,000 to 11,275 (modest 13% gain)
- Calibration improvements up to 14% with just 100 human ground truth responses

## 8 Validation Patterns Cataloged
The paper surveys the entire literature and identifies 8 distinct approaches people use to validate LLM simulations.

## Key Takeaway for Our Experiment
PPI is the gold-standard calibration method. Even small human samples (100 responses) provide meaningful calibration. But the 83% false positive rate on null findings is devastating — it means heuristic validation (the most common approach) is actively misleading. We should use PPI from the start.
