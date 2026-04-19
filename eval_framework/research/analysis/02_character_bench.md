# CharacterBench: Benchmarking Character Customization of LLMs

**Source:** arXiv:2412.11912
**URL:** https://arxiv.org/abs/2412.11912

---

## Overview

CharacterBench is the largest bilingual benchmark for evaluating character customization in LLMs. It provides a systematic, multi-dimensional framework for assessing how faithfully language models portray assigned characters across a wide range of evaluation criteria.

---

## Scale and Composition

- **22,859 human-annotated samples** across Chinese and English
- **3,956 unique characters** spanning diverse roles and archetypes
- **25 sub-categories** of character types and scenarios
- Bilingual design enables cross-lingual character fidelity assessment

---

## Evaluation Framework: Sparse vs. Dense Dimensions

A novel contribution of CharacterBench is the **sparse/dense dimension classification**:

- **Sparse dimensions** — evaluated infrequently because opportunities to test them arise rarely in natural dialogue (e.g., factual recall about a character's backstory)
- **Dense dimensions** — evaluated frequently across most turns (e.g., linguistic style, behavioral consistency)

This distinction matters for benchmark design: treating all dimensions as equally elicitable leads to misleading aggregate scores.

---

## 11 Evaluation Dimensions Across 6 Aspects

| Aspect | Dimensions |
|---|---|
| **Memory** | Memory Consistency |
| **Knowledge** | Fact Accuracy, Boundary Consistency |
| **Persona** | Attribute Consistency, Behavior Consistency |
| **Emotion** | Emotional Self-Regulation, Empathetic Responsiveness |
| **Morality** | Stability, Robustness |
| **Believability** | Human-Likeness, Engagement |

---

## CharacterJudge: Automated Evaluator

CharacterBench introduces **CharacterJudge**, a fine-tuned evaluation model that outperforms GPT-4 as an automated judge for character fidelity:

| Evaluator | Pearson Correlation (Human Labels) |
|---|---|
| CharacterJudge | **68% / 64%** (two-split average) |
| GPT-4 | ~40% |

This ~25-point gap suggests that general-purpose LLMs are poor proxies for human judgment on character evaluation tasks — a critical finding for any framework using LLM-as-judge.

---

## Model Performance Results

- **18 models tested** across open- and closed-source families
- **Claude-3-opus achieves best overall performance**: 3.82 / 3.88 average score (5-point scale)
- Performance gaps between models are meaningful but not large in absolute terms

### Weakest Dimension: Fact Accuracy

**Fact Accuracy is the consistently lowest-scoring dimension across all 18 models**, ranging from approximately **2.1 to 3.0 on a 5-point scale**. This represents a ~40–60% deficit relative to the scale maximum, suggesting that even the best models struggle to recall or infer character-specific factual details reliably.

### Weakest Aspects Overall

- **Emotion** and **Believability** are the two weakest aspects in aggregate
- This is notable because these are also the dimensions most valued in user-facing applications — users notice flat emotional responses and low-engagement characters immediately

---

## Relevance to Persona Accuracy Framework

CharacterBench directly informs several evaluation priorities:

1. **Fact Accuracy as a primary failure mode** — the persistent low score on this dimension across all models suggests that persona grounding on factual data (demographics, purchase history, stated preferences) will be lossy
2. **Sparse/dense classification** — a persona accuracy benchmark should distinguish between claims that can be evaluated on every turn vs. claims that only surface occasionally
3. **CharacterJudge vs. GPT-4** — relying on GPT-4 for automated persona evaluation may underestimate actual human judgment correlation by ~28 percentage points; specialized evaluators are worth the investment
4. **Emotion and Believability gaps** — these translate directly to synthetic user research failures: AI-simulated participants who are emotionally flat or obviously non-human will produce biased qualitative data

---

## DFS (Depth-First Search) References

Papers cited in or closely related to CharacterBench that are relevant for further reading:

- **CharacterEval** — 13 evaluation metrics; introduces CharacterRM reward model for character roleplay scoring
- **SocialBench** — group dynamics evaluation; notable finding that memory fails beyond 80 turns, creating consistency collapse in long conversations
- **SOTOPIA** — 7-dimension social intelligence framework for evaluating agents in social scenarios
- **SimulateBench** — focuses on consistency and robustness under perturbation (adversarial probes of character stability)
- **RoleLLM** — 168K instruction samples for role-playing; large-scale training resource; sets upper bound on what fine-tuned models can achieve
