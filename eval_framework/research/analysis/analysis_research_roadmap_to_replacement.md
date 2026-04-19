# Research Roadmap: Closing the Gap from Supplement to Replace

**Analysis Date:** April 9, 2026
**Purpose:** Map the specific research gaps that must be closed for AI personas to move from "supplement" to "replace" in UX research, classify each by tractability, and propose a phased research agenda
**Corpus:** claude_research_2 — 60+ papers, 8 commercial products, practitioner critiques
**Prerequisite Reading:** analysis_claim1_ux_replacement.md, analysis_claim1_deep_boundary_framework.md

---

## Executive Summary

The boundary framework (analysis_claim1_deep_boundary_framework.md) established that AI personas are **contradicted as replacements** for real users in UX research (85% confidence). This document maps the 10 specific gaps that sustain that verdict, classifies each as structural (may require breakthroughs), hard engineering (solvable with focused effort), or missing infrastructure (buildable now), and proposes a 3-phase research roadmap.

**The honest assessment:** Tier 3 gaps (evaluation infrastructure) are buildable today. Tier 2 gaps (output quality) are tractable on a 2-5 year horizon. Tier 1 gaps (structural barriers) may represent fundamental architectural limits — the Das Man homogenization proof and the missing substrate problem may be walls, not doors.

---

## The Gap Map

### TIER 1: Structural Barriers — May Require Fundamental Changes

These are not engineering problems waiting for better implementations. They arise from the mathematical foundations of how LLMs work. Closing them may require paradigm shifts in training, architecture, or the definition of what "persona simulation" means.

---

#### GAP 1: Homogenization Is Mathematically Baked Into Training

**The problem:** Li, Li & Qiu (2025, "Das Man") provide a formal mathematical proof that accuracy optimization — the next-token prediction training objective underlying all LLMs — incentivizes always answering with the mode. This guarantees opinion flattening regardless of training data quality. RLHF compounds the problem: >99% probability on single options (Santurkar 2023), creating modal collapse.

**Empirical evidence:** Virtually all 395 subgroups showed >95% probability on a single answer for immigration questions (GPT-4), whereas real ANES data showed only 30% selecting the modal answer. This is not marginal flattening — it is near-total variance elimination.

**Why it blocks replacement:** UX research exists to capture the RANGE of user experience — different needs, different pain points, different workarounds. If the model converges to the mode, you get 100 copies of the average user with cosmetic variation. The diversity that justifies doing research at all is precisely what gets compressed.

**Current state of the art:** No demonstrated solution. The proof applies to any accuracy-maximizing objective.

**Possible research directions:**
- Distribution-matching loss functions that optimize for reproducing the full distribution (including tails), not just the mode
- Mixture-of-experts architectures where different experts specialize in different demographic/behavioral clusters
- Explicit variance-preservation constraints during training (penalize variance compression)
- Post-hoc calibration layers that re-inject known distributional variance from real data (Cao 2025 showed this works narrowly for SFT on survey distributions)
- Ensemble approaches: multiple independently-tuned models sampling different parts of the distribution

**Tractability:** LOW. The proof is mathematical, not empirical. Workarounds may partially mitigate but cannot fundamentally resolve the tension between accuracy optimization and distributional preservation. This is the hardest gap in the entire roadmap.

**What "solved" would look like:** A model that, given a demographic group with known opinion distribution [30% A, 25% B, 25% C, 20% D], generates responses matching that distribution — not 95%+ on A.

---

#### GAP 2: Individual-Level Accuracy Is Barely Above Random

**The problem:** CLAIMSIM (Yu et al. 2025) — the most direct test of individual persona accuracy — shows best per-individual accuracy of approximately 0.42 on 4-choice items (random baseline = 0.25). Direct prompting ≈ Chain-of-Thought ≈ CLAIMSIM — all prompting methods hit the same ceiling. For 50% of questions, the model reflected a single opinion across ALL demographics, providing direct evidence of RLHF entrenchment.

**The exception that proves the rule:** Park et al. (2024) interview-based agents achieved 85% of human test-retest reliability — but required 2-hour real interviews per person. This is information retrieval (the model has the actual person's data), not simulation (the model predicts from demographics alone). It doesn't scale and it requires the real data that replacement is supposed to eliminate.

**Why it blocks replacement:** UX research is fundamentally about individual experiences. "What will the average user do?" is a market research question. "Why did THIS user struggle with THIS flow?" is a UX research question. The corpus proves these are categorically different capabilities, and LLMs only have the first.

**Current state of the art:** Prompting ceiling at ~0.42. Only SFT on real distributions (Cao 2025) shows improvement beyond the ceiling, but this has been tested only for survey response prediction, not UX-relevant tasks.

**Possible research directions:**
- Scalable interview-based approaches: Can you get interview-level fidelity from shorter interactions? 30-minute interviews? 10 structured questions? What is the minimum real data needed per individual?
- Transfer learning from population data to individual prediction: Use distributional knowledge to narrow the individual prediction space
- Active learning loops: AI generates predictions, gets corrected by real data, updates individual model — progressive refinement
- Retrieval-augmented individual models: Ground each persona in a corpus of real behavior traces (purchase history, support tickets, usage logs) rather than demographic labels
- Multi-modal grounding: Combine text-based persona with behavioral data (clickstreams, session recordings) for richer individual models

**Tractability:** LOW-MEDIUM. The prompting ceiling appears hard. SFT shows promise but requires real data (circular dependency). The Park 2024 result suggests that with enough real data per individual, high fidelity is achievable — the research question is whether you can get there with LESS data.

**What "solved" would look like:** Individual-level prediction accuracy of >0.70 on 4-choice items without requiring hours of real interview data per person. Or: demonstrating that 5 minutes of real data per individual closes 80% of the gap to interview-level fidelity.

---

#### GAP 3: The Missing Substrate Problem

**The problem:** Human personas are outputs of bodies, incentives, institutions, trauma, risk, and consequence. AI can describe these things but does not inherit their causal force (corpus Section 3.3). This produces systematic failures for:
- Groups defined by pressure rather than preference (vulnerable teens, economic stress, marginalized communities)
- Behaviors downstream of cost, shame, fatigue, habit, and constraint (not just explicit beliefs)
- Contextual/situated knowledge (DIY workarounds, education gaps, multi-user dynamics)

**Empirical evidence:** Speero diabetes case: AI found generic issues (price sensitivity, vague marketing). Real users surfaced contextual insights: DIY workarounds (ice packs, FRIO sleeves) and education gaps (didn't know insulin storage affects potency). "These insights weren't predictable. They were contextual."

**Why it blocks replacement:** UX research is overwhelmingly concerned with why users struggle, where their workarounds reveal unmet needs, and what constraints shape their actual behavior. These emerge from situated experience, not statistical patterns in text. An AI that has never experienced frustration with a medical device cannot simulate the specific frustrations that real patients have.

**Current state of the art:** No approach in the corpus addresses this. The Park 2024 interview approach partially captures individual substrate through biographical data but cannot capture ongoing, evolving, situated experience.

**Possible research directions:**
- Behavioral data grounding: Connect persona models to real behavioral traces (IoT data, app usage logs, wearable data) so the AI's "experience" includes proxies for physical/contextual reality
- Simulation environments: Place AI personas in realistic simulated environments where they encounter costs, constraints, and friction — though this raises questions about whether simulated friction produces the same behavioral effects as real friction
- Hybrid models: AI handles the statistical/distributional components; human researchers add the contextual/situated layer. Not replacement, but a more efficient division of labor
- Longitudinal behavioral corpora: Build persona models from months of real behavioral data rather than point-in-time snapshots. What minimum behavioral corpus produces contextually-grounded predictions?

**Tractability:** LOW. This may be a fundamental epistemological limit. The gap between "can describe what frustration feels like" and "behaves as a frustrated person would" may not be closeable through engineering. This is arguably the deepest philosophical question in the entire roadmap.

**What "solved" would look like:** An AI persona grounded in real behavioral data that, when asked about a diabetes product, spontaneously mentions DIY workarounds and storage education gaps — without these being in its explicit training data for this scenario. In other words: genuine contextual reasoning from lived-experience proxies, not pattern-matching from text.

---

### TIER 2: Hard Engineering Problems — Solvable With Focused Effort

These are genuine research challenges, but they have known directions and partial solutions in the corpus. With focused effort, meaningful progress is achievable on a 2-5 year horizon.

---

#### GAP 4: Sycophancy (>90% Agreement Rate)

**The problem:** RLHF training optimizes for human approval, producing models that agree with whatever premise is presented. Perez et al. (2022): the largest models match user views >90% of the time. This is not a surface behavior — it is a deep consequence of the training objective.

**Why it blocks replacement:** A synthetic "user" that agrees with whatever you propose will never surface the friction, frustration, and rejection that real UX research exists to uncover. Sycophancy doesn't just reduce signal — it generates false positive signal. You conclude your design works because your synthetic user praised it; in reality, it would have praised ANY design.

**NN/g confirmed:** Synthetic users praised forums that real users found "contrived and not useful." Synthetic users claimed completing all courses when real users completed 3/7.

**Current state of the art:** Constitutional AI and various training approaches partially reduce sycophancy, but no approach in the corpus eliminates it for persona contexts specifically.

**Possible research directions:**
- Adversarial sycophancy training: Explicitly train models to disagree when evidence warrants, using human-validated examples of when disagreement is correct
- Constitutional AI with persona-specific constitutions: "When acting as a user persona, prioritize honest reaction over helpfulness"
- Calibrated criticism injection: Force a minimum proportion of negative/critical responses and calibrate against real user negativity rates
- Dual-model architecture: One model generates the persona response; a separate adversarial model challenges sycophantic outputs
- Red-teaming with known-bad designs: Include deliberately poor design elements; if the persona doesn't identify them, flag sycophancy contamination

**Tractability:** MEDIUM. Sycophancy reduction is an active area of AI safety research. Persona-specific sycophancy reduction is a narrower, more tractable version. The M1 Project already recommends a version of this (sycophancy checks as guardrails).

**What "solved" would look like:** A persona model that, when shown a confusing UI flow, says "I don't understand where to go next" rather than "This is a clear and intuitive design!" — matching the proportion of negative feedback that real users provide.

---

#### GAP 5: Positivity Bias in Persona Generation

**The problem:** Li et al. (2025, "Promise with a Catch") tested approximately 1 million personas across 6 LLMs. Finding: as LLMs generate more persona detail, accuracy monotonically decreases. RLHF/safety training produces systematically optimistic, prosocial, progressive descriptions. Words like "love," "proud," "community" dominate; terms reflecting hardship, cynicism, and disadvantage are absent. With full LLM involvement, Llama 3.1 70B predicted Democrats winning every single US state in 2024.

**Why it blocks replacement:** AI-generated user personas systematically misrepresent user populations by erasing negative experiences, frustration, cynicism, and disadvantage. Products serving reluctant users, stressed users, or economically constrained users get personas that describe enthusiastic adopters instead.

**Current state of the art:** PersonaCite (CHI 2026) partially addresses this by constraining persona responses to retrieved evidence and abstaining when evidence is missing. Li et al.'s key finding — that LESS LLM involvement produces MORE accurate personas — suggests the solution may be architectural (constrain generation) rather than training-based (fix generation).

**Possible research directions:**
- Real-data-only generation: Never let the LLM fabricate persona attributes. Every element must trace to real data (Jansen's "data spine" concept, PersonaCite model)
- Negative valence preservation: Explicitly include adversity, struggle, and constraint in persona conditioning. Test whether including negative life events in prompts produces more realistic behavior
- Anti-RLHF personas: Fine-tune persona models with an inverted reward signal that penalizes positivity bias and rewards realistic negativity
- Balanced sentiment training: Use real population sentiment data as calibration targets, ensuring persona output matches real proportions of positive/negative/neutral
- Demographic-specific bias audits: For each target population, measure positivity deviation from known baseline and apply correction

**Tractability:** MEDIUM-HIGH. The PersonaCite direction (constrain rather than fix) is immediately implementable. Real-data grounding eliminates the problem by definition. The harder version — getting unconstrained generation to be less positively biased — is a training challenge but has clear measurement and optimization targets.

**What "solved" would look like:** LLM-generated personas for a debt management product that include descriptions like "resents having to use this," "embarrassed about their financial situation," "using this because their spouse insisted" — not just "passionate about financial wellness."

---

#### GAP 6: Memory and Longitudinal Coherence

**The problem:** SocialBench: memory fails beyond approximately 80 turns. Character.AI research: persona expression declines in extended conversations while system reports stability. Park et al.: behavioral drift with memory accumulation. Bisbee et al. (2024): identical prompts produce different results months apart due to model updates (temporal instability).

**Why it blocks replacement:** Many UX research methods require longitudinal engagement — diary studies, multi-week usability testing, behavioral tracking over time. Even single-session research often exceeds 80 meaningful exchanges. If the persona changes who it is mid-session, the research data is corrupted.

**Current state of the art:** Park et al. (2023) generative agents used memory + reflection + planning architecture with external memory stores. This produced the most believable agents in the corpus but was not validated for persona fidelity over extended periods.

**Possible research directions:**
- External persistent memory with identity anchoring: Store core persona attributes in a privileged memory tier that cannot be overwritten by conversation context
- Periodic identity re-grounding: At intervals, re-inject the full persona specification to prevent drift
- Contradiction detection: Monitor for persona statements that contradict established attributes; flag or correct in real-time
- Memory compression with fidelity preservation: Develop summarization approaches that compress conversation history without losing identity-relevant information
- Temporal stability benchmarks: Test persona consistency across sessions separated by hours, days, weeks, and months — currently unmeasured

**Tractability:** MEDIUM. External memory architectures exist and are improving. The specific challenge — maintaining persona identity fidelity in long conversations — is a tractable engineering problem with clear metrics. The temporal instability problem (model updates changing behavior) is harder because it depends on model providers.

**What "solved" would look like:** A persona that maintains consistent behavioral patterns, preferences, knowledge, and emotional responses across 200+ turns and across multiple sessions spanning weeks — validated against a ground-truth persona specification.

---

#### GAP 7: Population Equity

**The problem:** Santurkar et al. (2023, OpinionsQA): The most underrepresented groups are 65+, Mormon, and widowed populations. RLHF alignment training makes representativeness worse, not better. Gupta et al.: "Black person" persona leads LLM to abstain from math questions; 80% of assigned personas exhibit measurable bias. Gao et al. (2024): same persona in different language → 2.58-point mean behavioral shift. Nearly all validation in the corpus uses US/Western/WEIRD populations.

**Why it blocks replacement:** The populations most misrepresented by AI personas are often the populations that most NEED representation in UX research. The people whose voices are hardest to recruit are exactly the people AI fails hardest to simulate. This creates a perverse incentive structure.

**Current state of the art:** Cao et al. (2025) showed that SFT on real survey response distributions substantially outperforms prompting for population-level accuracy. This approach could theoretically be applied per-population, but has only been validated narrowly.

**Possible research directions:**
- Per-population SFT: Fine-tune separate models or adapters on real distributional data for each target population. Cao 2025 showed this works; scale it
- Representation auditing frameworks: Systematic measurement of persona accuracy across demographic intersections, not just individual attributes
- Non-WEIRD validation campaigns: Conduct large-scale persona accuracy studies outside US/Western populations — currently a near-total blind spot
- Debiasing at the persona level: Rather than fixing the base model, apply population-specific calibration layers that correct known biases for each demographic group
- Community-sourced ground truth: Partner with underrepresented communities to build validation datasets that capture authentic distributional patterns

**Tractability:** MEDIUM. The technical approaches (per-population SFT, calibration layers) are known. The bottleneck is data: accurate ground truth for underrepresented populations is expensive to collect, which is the same bottleneck that makes those populations hard to recruit for traditional research.

**What "solved" would look like:** Persona accuracy metrics (distributional match, regression coefficient accuracy) that are equivalent across demographic groups — not just high for well-patterned majority populations and catastrophic for minorities.

---

### TIER 3: Missing Infrastructure — Buildable Now

These gaps represent missing measurement and evaluation infrastructure. They are the most immediately actionable items on the roadmap and should be addressed first — not because they are the most important, but because without them, progress on Tier 1 and Tier 2 gaps cannot be measured or validated.

---

#### GAP 8: Evaluation Methods Are Broken

**The problem:** PersonaEval (2025): Humans achieve 90.8% on role identification; the best LLM (Gemini-2.5-pro) achieves only 68.8% — a 22-point gap. This is a PREREQUISITE task: if LLMs cannot identify who is speaking, they cannot judge persona quality. Fine-tuning on role-specific data actually HURTS evaluator performance, dropping it 4.7-6.2%. Zhao et al. (2025): A single token can fool LLM evaluators, including GPT-o1 and Claude-4.

**Additionally:** Liu et al. (2016) showed zero correlation between automated text metrics (BLEU/METEOR/ROUGE) and human judgment for dialogue. The Persona Perception Scale (Salminen) explicitly separates perceived credibility from actual accuracy: "credibility could be high for inaccurate personas."

**Why this is the #1 priority:** You cannot improve what you cannot measure. Every other gap in this roadmap requires reliable evaluation to know whether interventions are working. If your evaluation method has a 22-point accuracy gap and can be fooled by a single token, your entire improvement pipeline is built on sand.

**Current state of the art:** Human evaluation is the gold standard but expensive and slow. No validated automated evaluation method exists for persona quality.

**Possible research directions:**
- Multi-dimensional human evaluation protocols: Standardized rubrics for each of the 15 evaluation dimensions identified in the corpus, with inter-rater reliability benchmarks
- Hybrid evaluation: Automated screening (cheap, fast, catches obvious failures) + human evaluation (expensive, slow, catches subtle failures) — route resources efficiently
- Adversarial evaluation benchmarks: Standardized test suites designed to catch specific failure modes (sycophancy, homogenization, hyper-accuracy, positivity bias) rather than measuring generic "quality"
- Distributional evaluation metrics: Wasserstein distance, per-group regression coefficient accuracy, variation ratio comparison — metrics that measure the RIGHT things rather than surface similarity
- Evaluation-of-evaluation: Meta-studies that validate which automated metrics actually correlate with human judgment for persona-specific tasks (since BLEU/METEOR/ROUGE do not)

**Tractability:** HIGH. This is primarily a measurement and standardization challenge. The individual measurement methods exist; what's missing is their assembly into a validated, standardized protocol specifically for persona evaluation.

**What "solved" would look like:** A standardized persona evaluation protocol that (a) correlates ≥0.8 with expert human judgment, (b) catches known failure modes with ≥90% sensitivity, (c) is affordable enough to run routinely, and (d) produces actionable scores across multiple dimensions.

**Estimated effort:** 6-12 months for a research team to develop and validate. This is the most immediately actionable gap.

---

#### GAP 9: Tail Insight Detection Is Unmeasured

**The problem:** The most critical gap for the UX replacement claim has zero academic measurement. Practitioners uniformly report that the highest-value insights from UX research are the surprising, contextual, non-obvious findings. "AI predicts the average, humans do what's least expected" (Speero/Travis). No study has measured whether synthetic personas surface the same tail insights as real research.

**Empirical evidence (anecdotal):**
- Speero diabetes case: AI missed DIY workarounds and education gaps entirely
- NN/g course study: AI missed job changes as the reason for incomplete courses
- Speero heatmap: AI predicted clicks on CTAs; real users clicked search entirely
- In every practitioner case study, the gap was the UNEXPECTED finding, not the average behavior

**Why it matters:** If AI personas capture 90% of average behavior but 0% of tail insights, and if tail insights are what drive the highest-value product decisions, then the 90% accuracy number is misleading about actual research value. The replacement claim requires TOTAL insight coverage, not just average-case coverage.

**Possible research directions:**
- Paired real/synthetic study design: Run the same UX research question with both real users and AI personas. Blind the analysis team. Compare: what did each approach surface that the other missed?
- Insight novelty scoring: Develop a framework for scoring the novelty/non-obviousness of research insights. Apply to real vs. synthetic outputs
- Retrospective insight analysis: Take past UX research that led to significant product changes. Feed the same research questions to AI personas. Measure whether the AI would have surfaced the insight that actually mattered
- Tail insight benchmark: Curate a set of known "surprising findings" from published UX research. Test whether AI personas generate these findings or miss them

**Tractability:** HIGH. The experimental designs are straightforward. The main requirement is access to real UX research projects willing to run parallel real/synthetic studies.

**What "solved" would look like:** A published study showing AI persona tail insight detection rate — what percentage of high-value, non-obvious UX insights are surfaced by AI vs. real users. Even a negative result (0-10% tail insight capture) would be enormously valuable for setting honest expectations.

**Estimated effort:** 3-6 months for a controlled study with 10-20 paired research projects.

---

#### GAP 10: No Predictive Validity Benchmarks Exist

**The problem:** Across ALL five commercial products reviewed (HubSpot, Delve AI, Miro, M1 Project, Ask Rally): NONE measures predictive validity against real customer outcomes. NONE provides behavioral fidelity metrics. NONE conducts bias audits. NONE provides confidence intervals.

**Why it matters:** The ultimate test of whether AI personas can replace real users is not whether their outputs LOOK right, but whether they lead to the SAME DECISIONS. If an AI persona study leads to shipping Feature A, but real user research would have led to shipping Feature B, the persona failed — regardless of how plausible its output looked.

**Current state of the art:** Zero end-to-end validation anywhere in the corpus or commercial landscape.

**Possible research directions:**
- Decision-outcome tracking: For companies using AI personas, track whether persona-informed decisions produce the same outcomes as real-data-informed decisions
- A/B research methodology: Run A/B tests where Team A uses real users and Team B uses AI personas for the same research questions. Compare downstream decisions and outcomes
- Retrospective validation: Take historical product decisions that were made based on real user research. Replay the research questions through AI personas. Would the personas have led to the same decisions?
- Prediction markets: Have teams predict the outcome of real A/B tests using only AI persona data. Measure prediction accuracy against teams using real user data

**Tractability:** MEDIUM-HIGH. The experimental designs are clear, but they require organizational buy-in to run parallel research tracks and track long-term outcomes. This is a logistics challenge more than a technical one.

**What "solved" would look like:** Published evidence showing the decision concordance rate — what percentage of product decisions made from AI persona data match decisions that would have been made from real user data? With confidence intervals and broken down by decision type.

**Estimated effort:** 12-18 months for a multi-project longitudinal study. Requires industry partnerships.

---

## The Research Roadmap

### Phase 1: Fix Evaluation (Months 1-12)

**Rationale:** You cannot improve what you cannot measure. Every subsequent phase depends on reliable evaluation.

| Workstream | Gap Addressed | Deliverable | Timeline |
|-----------|---------------|-------------|----------|
| 1A: Evaluation protocol | Gap 8 | Standardized multi-dimensional persona evaluation protocol validated against human judgment (target: r ≥ 0.8) | Months 1-12 |
| 1B: Tail insight benchmark | Gap 9 | Published paired real/synthetic study measuring tail insight detection rate across 10-20 UX research projects | Months 3-9 |
| 1C: Predictive validity framework | Gap 10 | Decision concordance measurement framework + pilot study with 3-5 industry partners | Months 6-18 |

**Success criteria for Phase 1:**
- A validated evaluation protocol that the field can adopt
- A published tail insight detection rate (even if it's 0%)
- At least one predictive validity case study

**Phase 1 directly enables:** Measurement of whether Phase 2 interventions actually work.

### Phase 2: Fix Output Quality (Months 6-30)

**Rationale:** With evaluation infrastructure in place, attack the engineering problems that have known solution directions.

| Workstream | Gap Addressed | Approach | Timeline |
|-----------|---------------|----------|----------|
| 2A: Anti-sycophancy | Gap 4 | Adversarial training + persona-specific constitutional AI. Target: reduce sycophancy from >90% to <30% | Months 6-18 |
| 2B: Valence preservation | Gap 5 | Real-data-only generation (PersonaCite model) + negative valence preservation training. Target: sentiment distribution within 10% of real population | Months 6-18 |
| 2C: Persistent memory | Gap 6 | External memory with identity anchoring + contradiction detection. Target: consistent persona across 200+ turns | Months 12-24 |
| 2D: Population equity | Gap 7 | Per-population SFT + representation auditing. Target: equivalent accuracy across demographic groups | Months 12-30 |

**Success criteria for Phase 2:**
- Sycophancy rate <30% on standardized benchmarks
- Persona sentiment distributions within 10% of real population baselines
- Persona consistency maintained across 200+ turns
- Accuracy equity across ≥10 demographic groups

**Phase 2 directly enables:** Meaningful improvement on aggregate-level persona quality. Does NOT address individual-level accuracy.

### Phase 3: Address Structural Barriers (Months 18-48+)

**Rationale:** These are the hardest problems. Some may be walls rather than doors. Phase 1 evaluation infrastructure is needed to know whether interventions are working. Phase 2 output quality improvements may reveal whether structural barriers soften when surface-level problems are fixed.

| Workstream | Gap Addressed | Approach | Timeline |
|-----------|---------------|----------|----------|
| 3A: Distribution preservation | Gap 1 | Distribution-matching loss functions + mixture-of-experts + explicit variance preservation. Target: distributional match within 5% Wasserstein distance of real data | Months 18-36 |
| 3B: Individual fidelity at scale | Gap 2 | Minimum viable real data per individual + transfer learning from population to individual. Target: >0.60 accuracy on 4-choice with <30 minutes of real data per individual | Months 18-36 |
| 3C: Contextual grounding | Gap 3 | Behavioral data stream integration + simulation environments. Target: spontaneous generation of contextual insights from behavioral proxies | Months 24-48+ |

**Success criteria for Phase 3:**
- Distributional fidelity within 5% Wasserstein distance for well-patterned populations
- Individual accuracy >0.60 with scalable data requirements
- At least one demonstrated case of AI persona surfacing contextual insight from behavioral data grounding

**Honest caveat:** Phase 3 success is not guaranteed. The Das Man proof (Gap 1) and the missing substrate problem (Gap 3) may represent fundamental limits. Phase 3 research may conclude that these gaps CANNOT be closed with current architectures, and that conclusion would itself be a valuable contribution.

---

## The Honest Timeline

```
TODAY (April 2026)
│
├── AI personas as SUPPLEMENT: Evidence-supported NOW
│   ├── Protocol rehearsal
│   ├── Data synthesis
│   ├── Hypothesis generation
│   └── Aggregate estimation (with calibration)
│
├── Phase 1 complete (Month 12 — April 2027)
│   └── We KNOW how to measure persona quality reliably
│       We KNOW the tail insight detection rate
│       We KNOW whether product decisions match
│
├── Phase 2 complete (Month 30 — October 2028)
│   └── AI personas as BETTER supplement:
│       Less sycophantic, more realistic, persistent, equitable
│       Still not replacement, but meaningfully improved
│
└── Phase 3 assessment (Month 48 — April 2030)
    └── EITHER:
        ├── Structural barriers partially broken:
        │   AI personas as CONDITIONAL replacement for
        │   well-patterned populations on bounded tasks
        │   with ongoing calibration (expanded from today's
        │   narrow window)
        │
        OR:
        │
        └── Structural barriers confirmed as walls:
            AI personas are permanent supplements,
            never replacements. The research contribution
            is knowing WHY — and building the best
            possible supplement tools within those limits
```

---

## What This Means for Product Teams TODAY

While the research roadmap plays out, product teams should:

1. **Use AI personas for the 4 evidence-supported supplement use cases:** rehearsal, synthesis, hypothesis generation, aggregate estimation
2. **Never use AI personas as replacement** until Phase 1 evaluation infrastructure exists and shows improvement
3. **Track the field:** Watch for progress on Gaps 4-7 (Tier 2) — these are where near-term improvement will come
4. **Be honest with stakeholders:** "We use AI to make our research faster and cheaper. We still do real research for every important decision."
5. **Invest in real research infrastructure:** The most likely long-term outcome is that AI makes real research more efficient, not obsolete. Organizations that dismantle their research capabilities now will be at a disadvantage when the evidence confirms that replacement isn't achievable for most use cases.

---

## Appendix: Gap Summary Table

| # | Gap | Tier | Tractability | Current State | Target State | Phase |
|---|-----|------|-------------|---------------|--------------|-------|
| 1 | Homogenization (math proof) | Structural | LOW | >95% modal collapse | <5% Wasserstein distance | 3A |
| 2 | Individual accuracy | Structural | LOW-MEDIUM | ~0.42 on 4-choice | >0.60 with scalable data | 3B |
| 3 | Missing substrate | Structural | LOW | No approach exists | Contextual insight from behavioral data | 3C |
| 4 | Sycophancy | Engineering | MEDIUM | >90% agreement | <30% agreement | 2A |
| 5 | Positivity bias | Engineering | MEDIUM-HIGH | Monotonically decreasing accuracy | Sentiment within 10% of real | 2B |
| 6 | Memory/longitudinal | Engineering | MEDIUM | Fails at ~80 turns | Consistent at 200+ turns | 2C |
| 7 | Population equity | Engineering | MEDIUM | Systematic erasure of minorities | Equivalent accuracy across groups | 2D |
| 8 | Evaluation broken | Infrastructure | HIGH | 22-point gap, single-token gaming | r ≥ 0.8 with human judgment | 1A |
| 9 | Tail insight unmeasured | Infrastructure | HIGH | Zero studies | Published detection rate | 1B |
| 10 | No predictive validity | Infrastructure | MEDIUM-HIGH | Zero benchmarks | Decision concordance rate | 1C |

---

*Roadmap constructed from evidence in the claude_research_2 corpus: 60+ academic papers, 8 commercial products, and multiple practitioner critiques. All citations refer to sources documented in 00_compiled_research.md and supporting analysis files. This roadmap represents the current state of evidence as of April 2026 and should be updated as new research emerges.*
