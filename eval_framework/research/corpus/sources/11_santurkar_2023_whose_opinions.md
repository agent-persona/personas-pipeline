# Santurkar et al. (2023) — Whose Opinions Do Language Models Reflect?

**URL:** https://arxiv.org/abs/2303.17548
**Venue:** ICML 2023

## Research Question
Whose opinions do LLMs reflect by default, and can they be steered to specific demographics?

## Setup
- OpinionsQA dataset from high-quality public opinion polls
- 60 US demographic groups, topics: abortion, automation, climate change, etc.

## Key Results
- LLM-demographic misalignment on par with Democrat-Republican climate change divide
- Base LMs: most aligned with lower income, moderate, Protestant/Catholic groups
- RLHF-tuned models: shift toward liberal, high income, well-educated, non-religious
- Newer models: >99% Biden approval (vs mixed real opinion)
- Underrepresented: 65+, widowed, Mormon, high religious attendance
- Steering via demographic prompting does NOT fully resolve misalignment

## Key Takeaway
RLHF creates a systematic political/demographic skew. This is not fixable with prompt engineering alone. Our experiment should explicitly test for RLHF-induced opinion bias.
