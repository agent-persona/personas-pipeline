# Hu & Collier (2024) — Quantifying the Persona Effect in LLM Simulations

**URL:** https://arxiv.org/abs/2402.10811
**Venue:** ACL 2024 Main Conference

## Research Question
How much variance in subjective annotations do persona variables actually explain?

## Setup
- 10 subjective NLP datasets with annotator demographic info
- 6 LLM variants (GPT-4, GPT-3.5-Turbo, Llama-2-70b variants, Tulu-2 variants)
- Mixed-effect linear regression: marginal R² (persona only) vs conditional R² (persona + text)

## Key Results
- Persona variables explain only **1.4%-10.6%** of annotation variance
- Exception: ANES presidential voting at 71.9% (extreme political polarization)
- Text-specific variation: up to 70%
- Critical threshold: when target R² < 0.1, predicted R² → zero
- Tulu-2-dpo-70b captures 81% of achievable annotation variance
- Persona prompting works best for high-entropy, low-SD samples

## Key Takeaway for Our Experiment
The persona effect has a fundamental ceiling. We should measure the R² of our persona variables on our ground truth FIRST, before attributing any results to persona quality. Most observed effects may be driven by task content, not persona conditioning.
