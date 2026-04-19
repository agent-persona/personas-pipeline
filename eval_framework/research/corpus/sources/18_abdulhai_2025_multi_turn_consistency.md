# Abdulhai et al. (2025) — Consistently Simulating Human Personas with Multi-Turn RL

**URL:** https://arxiv.org/abs/2511.00222
**Venue:** arXiv, November 2025 (UC Berkeley, UW, Google DeepMind)

## Research Question
Can multi-turn RL reduce persona inconsistency in LLM user simulators?

## Three Consistency Metrics
1. **Prompt-to-Line:** Does each utterance align with the base persona prompt?
2. **Line-to-Line:** Does the utterance contradict the conversation history?
3. **Q&A Consistency:** Does the persona give stable answers to diagnostic belief questions?

## Setup
- Three domains: open-ended conversation, education (27 learning styles), mental health (100 conditions)
- ~800 dialogues per task at 10/20/40/60 turns
- PPO fine-tuning with turn-level consistency rewards

## Key Results
- PPO reduced inconsistency by over 55%
- Open-ended: +58.5% improvement | Education: +20.6% | Mental health: +37.6%
- **Line-to-line consistency uniformly high (~0.9+)** — surface coherence looks good
- **Q&A consistency reveals hidden belief inconsistencies** — the real failures are hidden beneath fluent text
- Human-LLM agreement: 76.73% (exceeds human-human 69.16%)

## Key Takeaway for Our Experiment
The three consistency metrics are directly adoptable. The critical insight: surface coherence (line-to-line) MASKS belief inconsistencies that only Q&A probing reveals. We MUST include diagnostic questioning in our evaluation, not just assess response quality.
