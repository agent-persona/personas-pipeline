# Shin et al. (2025) — Spotting Out-of-Character Behavior: Atomic-Level Evaluation

**URL:** https://arxiv.org/abs/2506.19352
**Venue:** ACL 2025 Findings

## Research Question
Can we evaluate persona fidelity at the sentence level, not just the response level?

## Setup
- 12 LLMs, 15 Big Five personality personas, 3 tasks (questionnaires, essays, social media)
- 30 runs per condition

## Three Atomic Metrics
- **ACC_atom**: fraction of atomic units matching target trait range
- **IC_atom**: internal consistency within single responses (1 - normalized STD)
- **RC_atom**: test-retest reproducibility via Earth Mover's Distance

## Key Results
- Holistic vs atomic correlation: r=0.91 for accuracy but **r=0.40-0.45 for internal consistency**
- Neutral personas: catastrophic failure (ACC_atom as low as 0.01)
- Structured tasks: 0.73 ACC_atom vs free-form: 0.52 (GPT-4o)
- Socially desirable bias: underperformance on Low Conscientiousness, Neutral Openness
- Human validation: Kendall's tau 0.67-0.76

## Key Takeaway for Our Experiment
**Atomic metrics are essential.** Holistic scoring hides sentence-level contradictions. We should adopt ACC_atom/IC_atom/RC_atom or similar decomposition. The task-type dependency (structured vs free-form) means we must test across formats.
