# Prama, Danforth & Dodds (2025) — Misalignment of LLM-Generated Personas in Low-Resource Settings

**URL:** https://arxiv.org/abs/2512.02058
**Venue:** PersonaLLM Workshop, NeurIPS 2025

## Research Question
Can LLMs generate culturally authentic personas for Bangladeshi demographics (religion, gender, politics)?

## Setup
- 7 LLMs: GPT-5.0, GPT-4.1, GPT-4.0, Grok 3, Llama 3.3, DeepSeek V3, AI21 Jamba 1.5 Large
- 8 persona categories: 2 political, 4 religious, 2 gender
- 100 culturally-specific questions (40 political, 30 religious, 30 gender)
- 2,080 annotations, 3 Bangladeshi annotators, majority voting
- Persona Perception Scale (PPS): 6 dimensions x 7-point Likert

## Key Results
- Human accuracy: 87% | Best LLM (GPT-5.0): 61.7% | Worst (AI21 Jamba): 37.3%
- PPS Empathy gap: ~1 full Likert point (Humans 5.46 vs Grok 4.51)
- Pollyanna bias: LLM sentiment 5.99 vs Human 5.60 (+0.39)
- Over-used: "freedom," "harmony," "liberation" | Suppressed: "violence," "failure," "corruption"
- Buddhist personas consistently worst across all models
- Male > Female accuracy in most models

## Caveats
- Small questionnaire (100 questions)
- Single-country focus
- labMT dictionary limitations for contextual analysis

## Key Takeaway for Our Experiment
The Pollyanna bias is quantifiable and consistent. The PPS framework is validated and adoptable. Low-resource/minority personas are where models fail hardest — these should be oversampled in any evaluation.
