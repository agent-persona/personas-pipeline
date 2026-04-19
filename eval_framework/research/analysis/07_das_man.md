# ChatGPT is not A Man, but Das Man

**Source:** arXiv:2507.02919 | Li, Li & Qiu, 2025

## Key Contributions
- Das Man (Heidegger) argument: LLMs output stereotypical group opinions ("what one says") rather than genuine individual perspectives
- **Accuracy Optimization Hypothesis**: mathematically proves next-token prediction incentivizes always answering with the mode, guaranteeing homogenization regardless of training data quality. For beliefs x1...xn where p1 > p2 > ... > pn, expected accuracy U(q) = sum(pi * qi) is maximized when q1=1, all others 0
- Contrasted with Biased Sample Hypothesis (conventional explanation)

## Experiments
- ANES 2020: abortion (5-option) + immigration (4-option)
- 395 intersectional subgroups from 4 covariates: sex, race (6), education (5), religion (11)
- Models: GPT-4, Llama 3.1 8B/70B/405B
- Log probability extraction method (not sampling)

## Results
- Immigration: virtually all 395 subgroups show >95% probability on single answer (GPT-4). ANES shows only 30% of subgroups above this threshold
- Abortion: ANES P(VR<0.05)=0.4, GPT-4/Llama 405B show 0.8
- Structural inconsistency: querying "female" directly ≠ aggregating all female subgroups. Dominant answer FLIPS between aggregation levels
- Scaling does NOT fix: Llama 8B through 405B show same patterns; larger models may be MORE homogenized

## Error Analysis
- Only 2 survey questions tested (abortion, immigration)
- Only 4 demographic covariates — issues "would likely intensify with additional variables"
- US-focused political opinions only

## DFS References
### Bisbee et al. 2024 — "Synthetic Replacements for Human Survey Data?"
- Direct rebuttal to Argyle et al. — 48% of regression coefficients significantly different from ANES
- 32% of coefficients have FLIPPED SIGNS (opposite conclusion)
- Variance compression: 0.5-1.0 SDs smaller on 100-point thermometer
- Temporal instability: identical prompts produce different results 3 months apart
- Exaggerated polarization: synthetic Democrats more liberal, synthetic Republicans more hostile

### Santurkar et al. 2023 — OpinionsQA
- 1,498 questions from Pew surveys, 60 US demographic groups, 9 LMs
- RLHF makes representativeness WORSE (text-davinci-003 lowest of all)
- Modal collapse: >99% probability on single option for most questions
- Consistently underrepresented: 65+, Mormon, widowed, high religious attendance
- Steering helps only modestly — gap between best/worst groups persists

### Boelaert et al. 2025 — "Machine Bias"
- WVS data across 5 countries, 3 time periods
- "Machine bias" finding: errors are unpredictable across topics (not consistently favoring one group)
- Low adaptability: model outputs essentially same distribution regardless of persona
- Cannot replace human subjects for opinion research

### Perez et al. 2022 — Discovering LM Behaviors
- RLHF amplifies political views on gun rights and immigration
- Sycophancy: largest models match user views >90% of the time
- RLHF shapes religious views (favors Eastern over Western religions)

### Hartmann et al. 2023 — Political Ideology of Conversational AI
- ChatGPT consistently pro-environmental, left-libertarian across 4 languages
- Would have voted Green in Germany and Netherlands

## Relevance to Persona Accuracy
- Subgroup fidelity must check structural consistency across aggregation levels
- Variation ratio is a direct homogenization measure
- Standard LLM training will ALWAYS produce homogenization (mathematical proof)
- Minority viewpoints within subgroups are systematically erased
- Measurable dimensions: Wasserstein distance per subgroup, VR comparison, structural consistency, regression coefficient fidelity, temporal stability
