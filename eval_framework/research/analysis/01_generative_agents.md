# Park et al. — Generative Agents: Interactive Simulacra of Human Behavior

**Source:** arXiv:2304.03442 | UIST '23 | Stanford + Google
**Authors:** Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein

---

## Key Contributions

1. **Generative agents** as a concept: Computational software agents that simulate believable human behavior by drawing on an LLM, conditioned dynamically on changing experiences and environment. Distinct from prior NPC approaches (finite-state machines, behavior trees, RL agents) because behavior is generated rather than scripted.

2. **Three-component architecture** (memory stream + reflection + planning) that extends LLMs for long-term agent coherence. The key insight: raw LLM prompting cannot maintain consistency over time — you need structured memory retrieval, periodic higher-order synthesis (reflection), and hierarchical planning.

3. **Smallville**: A sandbox world (25 agents, Sims-like environment) demonstrating emergent social behaviors from simple initialization — information diffusion, relationship formation, group coordination (Valentine's Day party).

4. **Two evaluations**: A controlled ablation study and an end-to-end community evaluation demonstrating that each architectural component contributes meaningfully to believability.

---

## Experiments

### Experiment 1: Controlled Evaluation (Ablation Study)
- **Setup:** 25 agents simulated over 2 game days. At the end, agents were "interviewed" with 25 questions across 5 categories: self-knowledge, memory, plans, reactions, reflections (5 questions each).
- **Design:** Within-subjects, 100 human evaluators (Prolific, $15/hr, median age 25-34). Each participant ranked responses from 5 conditions for one randomly selected question per category.
- **Conditions:** (1) Full architecture, (2) No reflection, (3) No reflection + no planning, (4) No reflection + no planning + no observation, (5) Human crowdworker baseline.
- **Statistical analysis:** TrueSkill ratings + Kruskal-Wallis test + Dunn post-hoc tests with Holm-Bonferroni correction.

### Experiment 2: End-to-End Evaluation
- **Setup:** 25 agents running continuously over 2 full game days, no user intervention.
- **Measured:** (a) Information diffusion — spread of two seeded pieces of info, (b) Relationship formation — network density changes, (c) Coordination — agents showing up to the Valentine's Day party.

---

## Methods (Architecture Details)

### Memory Stream
- Comprehensive, append-only list of memory objects: natural language description, creation timestamp, most recent access timestamp.
- Memory objects include observations, reflections, and plans.

### Retrieval Function
- **Score** = α_recency × recency + α_importance × importance + α_relevance × relevance (all α = 1)
- **Recency:** Exponential decay (factor = 0.995) over game hours since last retrieval
- **Importance:** Integer 1-10 assigned by LLM at creation. Prompt: "rate the likely poignancy"
- **Relevance:** Cosine similarity between memory embedding and query embedding
- Scores min-max normalized to [0,1] before combining

### Reflection
- **Triggered** when cumulative importance scores of recent events exceed threshold of 150 (~2-3 times per day)
- **Step 1:** Feed 100 most recent memories → LLM generates "3 most salient high-level questions"
- **Step 2:** Use questions as retrieval queries → LLM produces insights with citations to source memories
- Reflections stored back into memory stream, creating recursive abstraction trees

### Planning
- Top-down hierarchical decomposition: Day plan (5-8 chunks) → Hour-long chunks → 5-15 minute action items
- Plans stored in memory stream and considered during retrieval
- **Reacting/Updating:** At each time step, agent perceives environment, retrieves context, LLM decides whether to continue plan or react

### Dialogue Generation
- Each utterance conditioned on: (a) agent's summary description, (b) summarized memories about the other agent, (c) current context, (d) dialogue history

**Underlying model:** gpt-3.5-turbo

---

## Claims

1. **Full architecture produces the most believable behavior** among all conditions, including human crowdworkers (TrueSkill 29.89 vs. 22.95).
2. **Each component contributes**: Removing reflection was next best (26.88), then no planning (25.64), then crowdworkers (22.95), then fully ablated (21.21). Effect size: Cohen's d = 8.16.
3. **Emergent social behaviors arise**: Information diffuses (4% → 32% for election, 4% → 52% for party), relationships form (network density 0.167 → 0.74), coordination succeeds (5 of 12 invited agents attended).
4. **Agents beat crowdworkers** at generating believable persona-consistent responses.

---

## Results (Quantitative)

| Condition | TrueSkill Mean (μ) | TrueSkill StdDev (σ) |
|---|---|---|
| Full Architecture | 29.89 | 0.72 |
| No Reflection | 26.88 | 0.69 |
| No Reflection, No Planning | 25.64 | 0.68 |
| Human Crowdworker | 22.95 | 0.69 |
| Fully Ablated | 21.21 | 0.70 |

- Kruskal-Wallis: H(4) = 150.29, p < 0.001
- All pairwise Dunn tests significant at p < 0.001 except crowdworker vs. fully ablated
- Information diffusion: Election awareness 4% → 32%, party awareness 4% → 52%
- Network density: 0.167 → 0.74
- Valentine's Day party: 12 invited, 5 attended (3 cited conflicts, 4 expressed interest but didn't follow through)

---

## Error Analysis / Caveats

1. **Embellishment/Hallucination:** Agents add fabricated details. Isabella knew about Sam's candidacy but added "he's going to make an announcement tomorrow" (never discussed). Yuriko described neighbor Adam Smith as having "authored Wealth of Nations."
2. **Memory retrieval failures:** Agents sometimes retrieve wrong conversations, leading to confusion.
3. **Hallucination rate:** Out of 453 agent responses about other agents, only 1.3% (n=6) were hallucinated. Never complete fabrication, but embellishment.
4. **Over-agreeableness:** Instruction tuning made agents excessively cooperative. Isabella rarely said no to suggestions, and other agents' interests contaminated her stated interests over time.
5. **Spatial/normative confusion:** Agents entered occupied bathrooms, closed stores after hours.
6. **Memory-induced drift:** As agents learned about more locations, they chose less typical ones (e.g., bar for lunch → "afternoon drinking habit").
7. **Cost and scale:** 25 agents for 2 days cost "thousands of dollars" and took "multiple days."
8. **Memory hacking vulnerability:** A crafted conversation could convince an agent of a false past event.
9. **Crowdworker baseline weakness:** Human baseline was basic competency, not expert performance.
10. **Short evaluation window:** Only 2 game days. Longer simulations may produce worse drift.
11. **LLM biases inherited:** May exhibit stereotypes or struggle with underrepresented populations.

---

## DFS Level 2 Reference A: Horton, Filippas & Manning — "Large Language Models as Simulated Economic Agents: What Can We Learn from Homo Silicus?"

**Source:** arXiv:2301.07543 (revised Feb 2026) | MIT & NBER

### Key Contributions
1. **"Homo Silicus" framework**: LLMs as implicit computational models of humans, analogous to Homo economicus.
2. **Persona-as-instruction**: Endowing AI agents with personas ("You are a socialist") dramatically changes behavior in theory-consistent directions.
3. **Calibrated mixture methodology**: Construct optimized mixtures of theory-grounded persona types to reproduce human aggregate behavior, test out-of-sample.
4. **Robustness testing**: Systematic checks — 10 languages, alternative phrasings, adversarial prompt variations.

### Experiments
- **Fairness perceptions** (Kahneman et al. 1986): Snow shovel price gouging. Political personas shift fairness ratings by ~1-1.5 Likert points. Robust across all permutations.
- **Social preferences** (Charness & Rabin 2002): Dictator games. Without personas, LLMs diverge significantly from humans. With personas, high fidelity.
- **Calibrated mixtures (out-of-sample)**: MSE drops from 0.182 to 0.094 (~48% improvement).
- **Status quo bias** (Samuelson & Zeckhauser 1988): Significant for all 5 LLMs tested.
- **Prospect theory** (Oprea 2024): "Very bad at math" personas show stronger fourfold pattern deviations.
- **Labor-labor substitution** (Horton 2025): Minimum wage hiring scenario consistent with field data.

### Key Finding for Persona Accuracy
- **Without personas, off-the-shelf LLMs are poor human predictors**
- **Theory-grounded personas dramatically outperform arbitrary demographic conditioning**
- **Meaningless attributes (hobbies, favorite TV shows) do NOT improve fidelity** — only theoretically meaningful dimensions matter
- Model heterogeneity: Different LLMs have different "default personalities" (Llama-3 more selfish, GPT-4o/Claude more efficiency-minded)

### Caveats
- Memorization risk (famous experiments in training data)
- Status quo bias varies by model (GPT-4o: 0.316 vs GPT-4: 0.931)
- Training data opacity limits contamination checks
- Non-WEIRD population coverage unknown
- Calibrated mixtures still have nonzero error

---

## DFS Level 2 Reference B: Argyle et al. — "Out of One, Many: Using Language Models to Simulate Human Samples"

**Source:** Political Analysis, 2023, Vol 31, Issue 3, pp. 337-351 | BYU

### Key Contributions

1. **"Algorithmic Fidelity" concept**: The degree to which patterns of relationships between ideas, attitudes, and socio-cultural contexts within a model mirror those within human sub-populations.

2. **Four criteria for sufficient algorithmic fidelity:**
   - **Criterion 1 (Social Science Turing Test):** Generated responses indistinguishable from human texts
   - **Criterion 2 (Backward Continuity):** Responses consistent with socio-demographic conditioning
   - **Criterion 3 (Forward Continuity):** Responses proceed naturally from conditioning context
   - **Criterion 4 (Pattern Correspondence):** Responses reflect underlying relationship patterns matching human data

3. **"Silicon Sampling" methodology**: Condition model on backstories drawn from real survey participants (ANES data), generating one "silicon subject" per real human subject.

4. **Reframing:** "Algorithmic bias" is fine-grained and demographically correlated — not monolithic. The same property that causes bias in deployment can be leveraged as fidelity in simulation.

### Results
- **Turing Test:** Evaluators guessed 61.7% of human lists were human, 61.2% of GPT-3 lists were human (p=0.44, not significantly different)
- **Vote prediction tetrachoric correlations:** 0.90 (2012), 0.92 (2016), 0.94 (2020)
- **Best subgroups:** Strong partisans (0.97-1.00), high political interest (0.93-0.97)
- **Worst subgroups:** Pure independents (0.02-0.41), weak partisans (0.71-0.74)
- **Cramer's V mean difference:** -0.026 (correlation structure remarkably preserved)
- **Cost:** Study 1 = $29

### Critical Limitation
- **Fidelity is distributional, not individual** — the model is a good population simulator, not a good individual simulator
- **No quantitative thresholds proposed** for the four criteria
- **Domain-specific only** — validated in U.S. politics, unknown elsewhere
- **Internet-text bias** cannot be fully corrected by silicon sampling

---

## Relevance to Persona Accuracy Framework

### What Makes Personas More Accurate
- Architecture (memory + reflection + planning) matters more than model scale — reflection alone = 8 standard deviations improvement
- Theory-grounded personas outperform arbitrary demographic conditioning
- Rich demographic backstories > names alone
- Demographic conditioning produces fine-grained, accurate sub-population distributions for well-documented groups (0.90+ correlations)

### What Makes Personas Less Accurate
- Weakly-patterned or unpredictable subgroups (independents: correlation drops to 0.02)
- Individual-level prediction (fidelity is distributional)
- Long-term coherence without memory architecture (drift and embellishment)
- Instruction-tuning-induced over-agreeableness
- Normative/physical world knowledge gaps
- Model-specific baseline heterogeneity

### Evaluation Methodology Toolkit
- Interview-based probing across 5 categories (Park)
- Ablation study with TrueSkill ranking (Park)
- Tetrachoric correlation against real survey data (Argyle)
- Cramer's V for inter-variable relationship matching (Argyle)
- Social Science Turing Test with human evaluators (Argyle)
- Robustness testing via multilingual/alternative/adversarial prompts (Horton)
- Calibrated mixture optimization + out-of-sample validation (Horton)
- Theory-grounded persona construction (Horton)
