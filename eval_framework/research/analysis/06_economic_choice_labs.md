# Can LLMs Replace Economic Choice Prediction Labs? The Case of Language-based Persuasion Games

**Source:** arXiv:2401.17435v5 | Technion
**Authors:** Shapira, Madmon, Reichart, Tennenholtz

---

## Critical Framing Note

This paper was initially described as "one of the strongest direct negatives in the literature." That characterization needs revision. The headline finding is actually **positive**: LLM-generated data CAN predict human economic choices, sometimes outperforming human training data. The negative evidence lives in the nuances — calibration gaps, strategy-specific failures, and history-dependency breakdowns. The truly devastating negatives come from the DFS references (Gao et al. "Scylla Ex Machina" and GTBench).

---

## Key Contributions

1. First systematic study of LLM-generated synthetic data for human choice prediction in complex economic settings
2. Dual-role framework: LLMs as both data generators (simulating players) AND predictors (classifying choices)
3. Surprising positive: Models trained on 1024+ synthetic players outperform models trained on 110 real humans
4. Critical insight: **Interaction history (not linguistic sentiment) drives prediction accuracy**
5. "No free lunch" finding: Accuracy gains come at the cost of calibration

---

## Experiments

### Language-Based Persuasion Game
- **Setup:** Expert selects 1 of 7 hotel reviews to present to a Decision-Maker who makes binary accept/reject. T=10 rounds. Expert always wants acceptance; DM only benefits from accepting high-quality hotels.
- **Expert strategies:** Six types (Naive/Stationary/Adaptive)
- **LLMs tested as generators:** Chat-Bison, Gemini-1.5, Qwen-2 72B, Llama-3 70B, Llama-3 8B
- **Human baseline:** 210 DMs, ~71,579 decisions

---

## Methods

- Persona-prompted LLMs generate synthetic gameplay
- Synthetic data trains LSTM predictor tested on held-out human decisions
- Comparison: synthetic-trained vs. human-trained vs. mixed (DUAL)
- Action similarity analysis isolates history-pattern vs. sentiment-pattern contributions
- Markov truncation experiments restrict LLM context to T=1,2 rounds

---

## Claims (The Nuanced Picture)

### Positive Headline
- Qwen-2-72B at 4096 synthetic players: **79.08% accuracy**
- Fine-tuned Llama-3-8B: **80.1% accuracy**
- Both exceed the ~74-78% human baseline

### Negative Specifics
- **Calibration nearly doubles:** ECE ~0.15 (LLM-only) vs ~0.08 (human-trained) — an **87% degradation**
- **Honest strategy exception:** Sentiment baseline outperforms LLM approach when expert is truthful
- **Truncated history degrades quality:** Markov T=1,2 noticeably worse
- **Negative sentiment-similarity correlation:** When LLMs match human sentiment patterns, prediction actually WORSENS
- Authors: "A key question that remains unanswered is WHEN and WHY the LLM-based approach is less effective"

---

## Results

| Configuration | Accuracy | ECE (Calibration) |
|---|---|---|
| Qwen-2-72B, 4096 players | 79.08% | ~0.15 |
| Fine-tuned Llama-3-8B | 80.1% | ~0.15 |
| DUAL (FT-Llama + 110 humans) | 80.1% | ~0.08 |
| Human 110 players baseline | ~74-78% | ~0.08 |
| Llama-3 8B, 4096 players | 71.93% | -- |

---

## Error Analysis / Caveats

1. **Calibration miscalibration**: LLMs introduce "systematic but artificial regularities" — right answers for wrong reasons, meaningless confidence signals
2. **Honest strategy failure**: LLMs overcomplicate simple sentiment-reading tasks, adding strategic noise to non-strategic settings
3. **History dependency**: LLMs need full 10-round context for realistic strategic learning; cannot infer dynamics from limited windows
4. **Persona masking**: Without persona diversification prompts, far more synthetic samples needed — personas mask model limitations rather than enabling genuine behavioral diversity
5. **Domain specificity**: All results from one game type; "Different tasks and setups could behave quite differently"

---

## DFS Level 2 Reference A: Gao et al. — "Take Caution in Using LLMs as Human Surrogates: Scylla Ex Machina"

**Source:** arXiv:2410.19599

**This is the strongest negative paper in the entire DFS tree.**

### Key Contributions
Systematic demolition of the LLM-as-human-surrogate claim through the 11-20 Money Request Game, exposing fundamental and unpredictable failure modes across 8 models and every prompting strategy.

### Experiment
- **11-20 Money Request Game:** Request 11-20 shekels, get your amount + 20 bonus if you pick exactly 1 less than opponent
- Choosing 20 = level-0 reasoning (greedy); each step down = one deeper reasoning level
- **Human modal choice: 17 (level-3)**
- **8 LLMs tested** with 1,000 sessions each
- Zero-shot, few-shot, CoT, emotional prompting, RAG, and fine-tuning all evaluated
- Cross-language (English/Chinese/Spanish/German) and role-manipulation testing

### Claims
**"Nearly all advanced approaches fail to replicate human behavior distributions."** All combinations show statistically significant divergence (p <<< 0.001) except ONE: fine-tuned GPT-4o on 108 human datapoints (p = 0.3417).

### Critical Failures

| Failure Mode | Detail |
|---|---|
| **Shallow strategic reasoning** | GPT-4 picks 19-20 (level-0/1); humans center on 17 (level-3). 2-3 level depth gap. |
| **RLHF fairness bias** | Claude 3-Opus selects 20 to "ensure fairness" — safety-training artifact with zero human analog |
| **Rule misunderstanding** | GPT-3.5 and Claude frequently misunderstand game mechanics |
| **Language sensitivity** | GPT-3.5 mean shifts from 15.51 (English) to 18.09 (German) — **2.58-point shift from language alone** |
| **Few-shot demand effects** | Models copy provided examples instead of learning reasoning patterns |
| **Memorization dependence** | 75-100% accuracy on beauty contest instructions, near-0% for this lesser-known game |
| **CoT failure** | Minimal improvement; "take a deep breath" causes Llama-2-7b to shift to extreme levels |
| **RAG insufficient** | Even providing the original paper as context fails to align distributions |

### Relevance to Persona Accuracy
**Devastating for persona simulation**: Failure is unpredictable across model/prompt/task combinations. RLHF corrupts strategic behavior systematically. Language changes persona "personality." Prompting cannot fix the problem. Only fine-tuning on actual human data works (defeating the purpose). Memorization masquerades as reasoning on well-known tasks.

---

## DFS Level 2 Reference B: Duan et al. — "GTBench: Uncovering the Strategic Reasoning Limitations of LLMs via Game-Theoretic Evaluations"

**Source:** arXiv:2402.12348 | NeurIPS 2024

### Key Contributions
Comprehensive 10-game benchmark proving LLMs fail catastrophically at strategic reasoning in deterministic settings.

### Experiments
10 games spanning complete taxonomy: Tic-Tac-Toe, Connect-4, Breakthrough, Nim (complete/deterministic), Kuhn Poker, Liar's Dice, Blind Auction, Negotiation, Pig, Iterated Prisoner's Dilemma. LLMs play against MCTS opponents.

### Results
- **Complete + Deterministic games: ALL models achieve NRA ≈ -1.0** (near-total failure against basic tree search)
- **Incomplete/Probabilistic games:** Competitive (NRA near 0)
- GPT-4 error taxonomy (157 turns): **45.1% endgame misdetection**, 33.3% factual errors, 15.7% overconfidence, 9.8% misinterpretation, 9.8% calculation errors
- **CoT BACKFIRES:** Llama-2 + CoT degrades by -0.23 NRA
- **Code-pretraining helps more than scale:** CodeLlama-34b (-0.01 NRA) vastly outperforms Llama-2-70b (-0.20 NRA)

### Relevance to Persona Accuracy
LLM personas **cannot do genuine strategic reasoning** in deterministic settings. They fail at Tic-Tac-Toe against basic search. The failure is in planning (45% endgame misdetection), not knowledge. LLMs succeed only where randomness accidentally aligns with mixed strategies. CoT makes strategic reasoning worse.

---

## DFS Level 2 Reference C: Horton et al. — "Homo Silicus"

**Source:** arXiv:2301.07543 (see 01_generative_agents.md for full analysis)

Most optimistic paper in this tree, but with crucial caveats:
- **Without persona instructions, LLMs diverge significantly from humans** in dictator games
- Llama = selfish, GPT-4o = efficiency-maximizing, Claude = fairness-oriented
- Status quo bias coefficient ranges from 0.316 (GPT-4o) to 0.931 (GPT-4) — **model selection determines the qualitative finding**
- Authors conclude LLM experiments **supplement but cannot replace** human experiments

---

## Synthesis: The Numbers That Matter

| Metric | Finding | Source |
|---|---|---|
| Strategic reasoning (deterministic) | NRA = -1.0 (total failure) | GTBench |
| Human distribution replication | p <<< 0.001 for 7/8 models | Gao et al. |
| Calibration gap | ECE 0.15 vs 0.08 (87% worse) | Shapira et al. |
| Language sensitivity | 2.58-point mean shift, same rules | Gao et al. |
| Reasoning depth gap | Level-0/1 (LLM) vs Level-3 (human) | Gao et al. |
| Endgame detection failure | 45.1% of all GPT-4 errors | GTBench |
| Fine-tuning success rate | 1 of 8 models matched humans | Gao et al. |
| CoT strategic degradation | -0.23 NRA (makes it worse) | GTBench |

---

## Relevance to Persona Accuracy Framework

### Bottom Line
LLM personas can generate statistically useful synthetic data at volume for prediction tasks, but they **cannot**:
- Replicate human strategic reasoning
- Produce calibrated uncertainty
- Maintain behavioral consistency across languages, prompts, or novel tasks

The appearance of human-like behavior is largely:
- **Memorization** on well-known tasks
- **Stochastic approximation** on probabilistic ones
- **Volume** compensating for individual inaccuracy

### Key Evaluation Dimensions Revealed
1. **Calibration** (ECE): Are confidence levels human-like?
2. **Strategic depth**: Can the persona reason multiple steps ahead?
3. **Cross-language stability**: Same persona, different language — same behavior?
4. **Novel task performance**: Does the persona work on tasks not in training data?
5. **History dependency**: Can the persona learn from interaction history?
6. **RLHF artifact detection**: Can we identify safety-training-induced behavioral distortions?
