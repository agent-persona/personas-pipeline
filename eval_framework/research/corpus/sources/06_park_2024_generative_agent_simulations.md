# Park et al. (2024) — Generative Agent Simulations of 1,000 People

**URL:** https://arxiv.org/abs/2411.10109
**Venue:** arXiv, November 2024

## Research Question
Can interview-grounded generative agents accurately simulate specific real individuals?

## Setup
- 1,052 real people, 2-hour qualitative interviews (avg 6,491 words)
- LLM agents conditioned on interview transcripts
- Compared: interview-based vs demographic-only vs persona-based

## Key Results
| Task | Interview-Based | Persona-Based | Demographic-Only |
|------|----------------|---------------|-----------------|
| GSS | 0.85 | 0.70 | 0.71 |
| Big Five | 0.80 | 0.75 | 0.55 |
| Economic Games | 0.66 | 0.66 | 0.66 |

- Population effect size correlation: 0.98
- Trimmed transcripts (80% shorter) retained 0.79-0.83 accuracy
- Bias reduction: 36% political, 38% racial (GSS)

## The Catch
Requires 2-hour real human interviews. Economic games stuck at 0.66 regardless of method. Interview-based is clearly superior but expensive.

## Key Takeaway for Our Experiment
This is the gold standard for persona accuracy — but it tells us the ceiling (0.85) and the floor (0.66 for strategic reasoning). We should test a gradient of grounding depth to find the cost-accuracy curve. Trimmed transcripts retaining most accuracy suggests a practical middle ground.
