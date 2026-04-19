# Taday Morocho et al. (2026) — Assessing Reliability of Persona-Conditioned LLMs as Synthetic Survey Respondents

**URL:** https://arxiv.org/html/2602.18462v1
**Venue:** ACM Web Conference 2026

## Research Question
Does conditioning LLMs on demographic personas improve their reliability as synthetic survey respondents?

## Setup
- World Values Survey wave 7, US respondents, 31 questions, 70,000+ instances
- 8 persona dimensions (gender, age, education, employment, occupation, income, religion, ethnicity)
- 2 models: Llama-2-13B, Qwen3-4B
- 3 conditions: Vanilla, Persona-Based, Random Guesser

## Key Results
- Random: HS=0.273, SS=0.537
- Llama Vanilla: HS=0.370 | Persona: HS=0.366 (slight DEGRADATION)
- Qwen Vanilla: HS=0.391 | Persona: HS=0.398 (marginal improvement, not significant)
- **Persona conditioning did NOT statistically significantly improve alignment**
- Most significant shifts occur in low-n strata — redistributing error, not reducing it

## Key Takeaway for Our Experiment
Persona conditioning's effect is heterogeneous and concentrated in specific items. The null result is important — it means we should test whether persona prompting helps AT ALL before building on it.
