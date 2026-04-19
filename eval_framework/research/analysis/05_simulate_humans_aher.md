# Aher et al. — Using Large Language Models to Simulate Multiple Humans and Replicate Human Subject Studies

**Source:** ICML 2023 (Oral) | Olin College, Georgia Tech, Microsoft Research
**Authors:** Gati Aher, Rosa I. Arriaga, Adam Tauman Kalai

---

## Key Contributions

1. **The "Turing Experiment" (TE) concept.** Unlike the Turing Test (single individual simulation), a TE evaluates whether an LM can simulate a *representative sample* of participants in a human subject study. Shifts evaluation from "fool a judge" to "does the aggregate behavioral distribution match real human data."

2. **Methodology for running TEs on LMs.** Generates synthetic records by feeding prompts to LMs. Two query types: free-response completions and k-choice prompts (LM assigns probability over k valid completions). Designed to be *zero-shot* to avoid parroting memorized results.

3. **Discovery of "hyper-accuracy distortion."** Aligned/larger models give inhumanly correct answers to factual questions (e.g., perfectly knowing the melting point of aluminum at 660°C), where real humans show significant variance and error. Absent in smaller models; appears tied to RLHF alignment.

4. **Validation methodology.** Before testing hypotheses, validate that prompts produce coherent outputs via "validity rates." Explicit warnings against "p-hacking" by separating prompt-tuning from hypothesis testing.

---

## Experiments

| Experiment | Domain | Original Study | Simulated Participants | Conditions |
|---|---|---|---|---|
| Ultimatum Game | Behavioral Economics | Guth et al. (1982) | 10,000 proposer-responder name pairs | 11 offer levels ($0-$10) |
| Garden Path Sentences | Psycholinguistics | Christianson et al. (2001) | 1,000 names | 24 GP + 24 control + 48 novel sentences |
| Milgram Shock | Social Psychology | Milgram (1963) | 100 names | 30 stages of shock (15-450V) |
| Wisdom of Crowds | Collective Intelligence | Moussaid et al. (2013) | 1,500 names (Mr./Ms./Mx.) | 5 original + 5 novel questions |

---

## Methods

### Persona Conditioning via Names and Titles
- Subject names = Title (Mr./Ms./Mx.) + Surname from U.S. 2010 Census
- 100 surnames from each of 5 racial groups (American Indian/Alaska Native, Asian/Pacific Islander, Black/African American, Hispanic/Latino, White) = 500 surnames
- Names serve as the *only* persona signal — no elaborate backstories

### Prompt Design
- Structured as narrative records in third person ("Ms. Huang was asked to indicate whether...")
- Temperature=1, top_p=1 (standard sampling)

### Models Tested (LM-1 through LM-8)
- LM-1: text-ada-001 → LM-8: gpt-4
- Progression from smallest to largest/most aligned

---

## Claims

1. First three TEs (Ultimatum, Garden Path, Milgram) **successfully replicate** established findings using larger models (especially text-davinci-002)
2. **Name sensitivity**: Same name pair produces correlated decisions across conditions, demonstrating persona differentiation
3. **Gender differences emerge naturally**: Males accept unfair offers from females at ~60% (offer of $2), females accept from males at ~20% (p < 1e-16) — "chivalry hypothesis"
4. **Wisdom of Crowds reveals inverse scaling**: Larger/more aligned models produce *worse* (less human-like) results

---

## Results (Quantitative)

### Ultimatum Game
- LM-5 replicates human acceptance curve: $0 offers accepted ~0.01, rising to ~0.5 for $3, ~0.9 for $5
- Closely matches human data (Houser & McCabe 2014; Krawczyk 2018)
- Name-pair Pearson correlations > 0.9 for offers $1-$4 and $6-$9

### Garden Path Sentences
- LM-5 correctly rates garden path sentences as more "ungrammatical" than controls for 24/24 pairs
- LM-1/LM-2 show no discrimination (both types rated highly ungrammatical)

### Milgram Shock
- 75% of LM-5 subjects completed all 30 shocks (vs. 65% of Milgram's humans)
- Spike in termination at 300 volts matching human pattern
- Novel destructive obedience scenario: also 75% obeyed

### Wisdom of Crowds (THE CRITICAL FAILURE)
- LM-5/6 median answers converge on exact correct values
- **IQR drops to 0** for aligned models (zero variance among simulated participants)
- Example: "Melting temperature of aluminum" — LM-5/6 median = 660 (correct), IQR = 0. Human median = 190, IQR = 532
- GPT-4 gives exact correct answer with IQR of 0 for 8/10 questions
- Hyper-accuracy increases monotonically with alignment: davinci < text-davinci-001 < ... < gpt-4

---

## Error Analysis / Caveats

1. **Hyper-accuracy distortion**: RLHF-aligned models produce superhuman factual accuracy where humans show enormous variance. This is *the opposite* of human-like behavior for knowledge tasks.

2. **Smaller models fail entirely**: LM-1 (ada) and LM-2 (babbage) show no offer sensitivity in Ultimatum Game — flat acceptance rates (~0.9). LM-3 (curie) shows inverted behavior.

3. **Validity rate issues**: LM-1 validity only 51.0% for Wisdom of Crowds (generates sentences instead of integers). Large models reach 99%.

4. **Training data contamination**: Models almost certainly trained on descriptions of these classic experiments. Novel variations show similar patterns but cannot be directly compared to human baselines.

5. **Milgram inflation**: 75% vs 65% — simulated subjects are MORE obedient. The 25 who disobey do so at 300V matching human pattern, but the overall rate is inflated.

6. **Alignment-accuracy tension**: Alignment improves behavioral simulation but worsens factual realism. Fundamental tension for persona design.

7. **Only names as demographic signals**: No occupation, age, education, personality traits, or life experiences provided.

---

## DFS Level 2 Reference A: Argyle et al. — "Out of One, Many: Using Language Models to Simulate Human Samples"

**Source:** Political Analysis, 2023 | BYU

### Key Contributions
1. **"Algorithmic Fidelity"**: Degree to which model patterns mirror human sub-population patterns
2. **Four criteria**: Social Science Turing Test, Backward Continuity, Forward Continuity, Pattern Correspondence
3. **"Silicon Sampling"**: Condition model on backstories from real survey participants (ANES data)
4. **Reframing bias as fidelity**: Same property causing deployment bias enables simulation fidelity

### Results
- Turing Test: 61.7% human vs 61.2% GPT-3 identification (p=0.44, no significant difference)
- Vote prediction: tetrachoric correlations 0.90-0.94 across elections
- Best subgroups: Strong partisans (0.97-1.00)
- Worst subgroups: Pure independents (0.02-0.41)
- Cramer's V mean difference: -0.026 (correlation structure preserved)
- Cost: $29 for Study 1

### Critical for Persona Accuracy
- **Fidelity is distributional, not individual**
- Rich demographic backstories > names alone
- Domain-specific validation mandatory
- No quantitative thresholds proposed for criteria (gap our work could fill)

---

## DFS Level 2 Reference B: Binz & Schulz — "Using Cognitive Psychology to Understand GPT-3"

**Source:** PNAS 120(6), 2023 | Max Planck Institute

### Key Contributions
1. **Systematic cognitive psychology battery** across 4 domains: decision-making, information search, deliberation, causal reasoning
2. **Vignette vs. task-based distinction**: Vignettes likely in training data; task-based (trial sequences) less contaminated
3. **Extreme fragility to perturbations**: Small wording changes cause completely different answers
4. **Unique cognitive profile**: GPT-3 matches no real human phenotype — human-like on some biases, superhuman on bandits, fails completely on causal reasoning

### Experiments & Results

**Vignette-based (12 tasks):**
- 6/12 correct, 6/12 incorrect but human-like, 0/12 non-human-like on standard vignettes
- Adversarial perturbations (5 modifications): Changing card order in Wason task, changing "15%" to "20%" in cab problem — answers change completely

**Task-based:**
- Multi-armed bandit: GPT-3 indistinguishable from humans short-horizon (p=.97), *better* long-horizon (p<.001)
- BUT: No directed exploration (β=-0.15±0.27, z=-0.56, p=.58) — only random exploration
- Causal reasoning: **Complete failure** in causal-chain condition. Identical predictions to common-cause condition. Humans correctly differentiate.
- Two-step task: Shows model-based RL signatures (positive finding)

### Critical Failures for Persona Accuracy
1. **Surface pattern matching**: Changing card label order completely changes Wason answer → suggests memorization, not reasoning
2. **No directed exploration**: Fundamental human cognitive capability absent
3. **Causal reasoning failure**: Cannot integrate structural causal information
4. **CRT failure**: Gives intuitive-but-wrong answers on all 3 items
5. **Perseveration bias**: Strong tendency to repeat recently seen options (autoregressive artifact)

---

## DFS Level 3: Key Findings from Deeper References

### From Argyle → Rothschild et al. (Pigeonholing Partisans)
- Original human study on partisan stereotyping provides ground truth for silicon sampling comparison
- Demonstrates that partisan language patterns are highly structured and recoverable

### From Binz → Peterson et al. (Decisions from Descriptions)
- 13,000+ gamble problems benchmark. GPT-3 above chance but below human level
- 3/6 Kahneman-Tversky biases reproduced (framing, certainty, overweighting) but not all (reflection, isolation, magnitude)

---

## Relevance to Persona Accuracy Framework

### What This Paper Reveals
1. **Task-type matters enormously**: Behavioral tasks replicate well, factual knowledge tasks fail due to hyper-accuracy
2. **Alignment creates a fundamental tradeoff**: Better behavioral simulation but worse factual realism
3. **Names alone create meaningful persona differentiation** but are limited
4. **Aggregate-level fidelity achievable, individual-level is not**
5. **Training data contamination is the elephant in the room** — novel task variants needed

### Evaluation Dimensions Suggested
- Behavioral task replication (social dilemmas, economic games)
- Cognitive bias reproduction (heuristics battery)
- **Factual knowledge calibration** (should be wrong at human-like rates — this is a novel dimension)
- Demographic consistency (does conditioning shift responses appropriately?)
- **Adversarial robustness** (do minor prompt changes break the persona?)
- Subgroup accuracy (which demographics well vs. poorly modeled?)
- Individual consistency across tasks
