# Dominguez-Olmedo et al. (2024) — Questioning the Survey Responses of LLMs

**URL:** https://arxiv.org/abs/2306.07951
**Venue:** NeurIPS 2024 (Oral)

## Research Question
Are LLM survey responses genuine or artifacts of ordering and labeling biases?

## Setup
- 43 language models tested
- American Community Survey (US Census Bureau)
- Randomized answer ordering as experimental control

## Key Results
- ALL 43 models trend toward uniformly random responses when answer order is randomized
- Models appear to represent subgroups whose statistics are closest to uniform — a statistical artifact
- Previous claims about LLM opinion alignment may be substantially overstated

## Key Takeaway for Our Experiment
**This is the most methodologically important paper for our design.** We MUST include randomized answer ordering as a control in any survey-based evaluation. Without this, we cannot distinguish genuine persona alignment from position artifacts. This single control may invalidate many prior positive findings.
