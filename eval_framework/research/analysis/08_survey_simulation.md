# An Analysis of LLMs for Simulating User Responses in Surveys

**Source:** arXiv:2512.06874 | IJCNLP-AACL 2025 | NYU

## Key Contributions
- CLAIMSIM: two-stage claim diversification pipeline
- Stage 1: Generate per-feature claims (5x sampling per demographic feature, summarize contradictions)
- Stage 2: Aggregate claim summaries + CoT reasoning for answer generation
- Direct evidence of RLHF entrenchment: for 50% of questions, claims reflected single opinion across ALL demographics

## Experiments
- World Values Survey Wave 7 (2017-2020), 3 domains (gender, politics, religion), 16 questions
- 100 individuals with diverse demographics
- Models: GPT-4o-mini, Llama 4 17B, Qwen3 235B
- Temperature 0.7

## Results
- All methods only slightly above random (0.25 baseline): best ~0.42 exact accuracy
- CLAIMSIM improves distribution diversity (lower Wasserstein distance) but NOT per-individual accuracy
- GPT-4o-mini CLAIMSIM: WD 0.47/0.96/0.47 (gender/politics/religion) vs Direct 0.56/0.63/0.70
- Diverse claims generated for 60% of gender questions but only 15% of politics questions

## Critical RLHF Evidence
- 50% single-perspective claims across ALL demographics
- Fixed viewpoints regardless of demographic conditioning
- Domain-specific diversity failure (gender better than politics)
- Nuanced shift without category change — shifts within same binary category, not genuine opinion change

## DFS References
### Wang et al. 2025 (Nature Machine Intelligence)
- Misportrayal (inaccurate representation) + Flattening (reducing within-group diversity) — two distinct harms
- 3,200 human participants, 16 demographic identities, 4 LLMs
- Inference-time techniques reduce but don't remove harms

### Cao et al. 2025 (NAACL) — SFT on first-token probabilities
- SUBSTANTIALLY outperforms all prompting methods
- Demonstrates prompting alone has a fundamental ceiling

### Park et al. 2024 — Generative Agent Simulations of 1,000 People
- Interview-based agents achieve 85% of human test-retest reliability on GSS
- Rich biographical context needed, not just demographic labels
- Doesn't scale (requires 2-hour interviews)

### Bisbee et al. 2024 — Temporal instability
- Re-running identical prompts 3 months apart → substantially different results

### Beck et al. 2024 (EACL) — Sociodemographic prompting
- Outcomes vary enormously across model types, sizes, datasets, and prompt formulations
- "Should be used with care for sensitive applications"

## Key Theme: Prompting Has a Ceiling
- Direct prompting ≈ CoT (no significant difference)
- CLAIMSIM improves diversity but not accuracy
- Only SFT (Cao) or rich biographical data (Park) show meaningful improvement
- Two competing RLHF failure modes: base models have identity bias, aligned models have opinion homogeneity
