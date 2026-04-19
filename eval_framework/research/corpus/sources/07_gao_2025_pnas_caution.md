# Gao et al. (2025) — Take Caution in Using LLMs as Human Surrogates

**Venue:** PNAS
**Referenced via:** UXtweak meta-review, multiple DFS sources

## Research Question
Can LLMs replicate human strategic reasoning behavior?

## Setup
- 11-20 Money Request Game: Players request 11-20 shekels; bonus of 20 if you request exactly one less than opponent
- 1,000 independent sessions per LLM, temperature 0.5

## Key Results
- ALL LLMs diverge from humans (Jensen-Shannon divergence, P < 0.001)
- Humans: Level-3 reasoning. LLMs: Level-0/1 — a 2-level strategic depth gap
- **Scaling does NOT help:** GPT-4 performed WORSE than GPT-3.5
- Memorization test: Beauty Contest Game: 75-100% accuracy. 11-20 Game: ~0% (max 2.9%)
- Demographic information had ZERO effect on output diversity
- Self-explanation inconsistencies: Claude chose large numbers citing "fairness" (contradicting incentives)
- Few-shot prompting: pronounced demand effect (models replicated examples, didn't learn)
- Fine-tuning on complete dataset achieved match — but requires the data it replaces

## Key Takeaway for Our Experiment
The memorization-vs-prediction distinction is critical. We MUST include novel tasks not in training data alongside well-known ones. The scaling paradox means we shouldn't assume bigger = better for personas. The demand effect means few-shot examples need careful design.
