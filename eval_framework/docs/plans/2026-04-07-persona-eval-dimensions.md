# Persona Evaluation Dimensions: Complete Catalog

> Exhaustive mapping of every testable dimension for LLM-generated personas, with automated and human evaluation methods, trustworthiness ratings, and research grounding.

---

## How to Read This Document

Each dimension includes:
- **What it tests** — the specific quality being measured
- **Why it matters** — what breaks if this dimension fails
- **Automated method(s)** — machine-checkable approaches
- **Human method(s)** — human-judgment approaches
- **Trustworthiness** — how reliable each method is, based on evidence
- **Research grounding** — which paper(s) from the corpus support this

Trustworthiness scale: **High** (validated against human judgment), **Medium** (correlates but with known gaps), **Low** (unvalidated or known to be unreliable), **Unknown** (no evidence either way).

---

## Tier 1 — Structural Validity

*"Is the output well-formed?"*
These are the cheapest, fastest, most reliable tests. They should gate everything else — if structure fails, don't bother scoring semantics.

### D1. Schema Compliance

**What it tests:** Does the persona output conform to the expected JSON/data structure? Are types correct? Are enums valid?

**Why it matters:** Downstream consumers (twin chat, dashboards, APIs) will crash or silently malfunction on malformed output. This is the most basic contract.

**Automated methods:**
- JSON Schema validation (jsonschema, Pydantic, Zod)
- Property-based testing (Hypothesis) to fuzz inputs and verify output shape
- Trustworthiness: **High** — deterministic, no ambiguity

**Human methods:**
- Not needed. This is fully automatable.

**Research grounding:** Standard software engineering practice. No persona-specific research needed.

---

### D2. Completeness

**What it tests:** Are all required fields populated? No nulls, empty strings, or placeholder values where real data is expected?

**Why it matters:** Incomplete personas create gaps in twin behavior — a persona missing "communication style" will produce generic responses in that dimension.

**Automated methods:**
- Null/empty checks per field
- Minimum length thresholds for narrative fields
- Semantic emptiness detection (e.g., "N/A", "not specified", "unknown" in free-text fields)
- Trustworthiness: **High** — deterministic

**Human methods:**
- Spot-check that "filled" fields contain meaningful content (not just plausible-looking filler)
- Trustworthiness: **High** — humans catch qualitative emptiness that passes length checks

**Research grounding:** Zhang et al. (PERSONA-CHAT) profile coverage problem — not all persona attributes get expressed even when structurally present.

---

### D3. Internal Logical Consistency

**What it tests:** Do the persona's own fields contradict each other? E.g., claims "budget-conscious" but lists exclusively premium preferences. Says "entry-level role" but has 20 years experience.

**Why it matters:** Self-contradictory personas break immersion immediately and produce incoherent twin behavior.

**Automated methods:**
- Rule-based constraint validation (if role=entry-level, then experience < 5 years)
- Entailment checking via NLI model (does field A entail or contradict field B?)
- LLM-as-judge: "Do these persona fields contradict each other?" with structured rubric
- Trustworthiness: Rule-based **High**. NLI **Medium** (good at obvious contradictions, misses subtle ones). LLM-as-judge **Medium-Low** (PersonaEval shows 22-point gap on role identification; contradiction detection is similarly hard).

**Human methods:**
- Read the full persona and flag contradictions
- Trustworthiness: **High** — humans are excellent at catching logical inconsistencies

**Research grounding:** Zhang et al. — best PERSONA-CHAT models still produce within-conversation contradictions. Li et al. 2016 — implicit persona models produced "I am 25" and "I am a grandmother" in the same session.

---

## Tier 2 — Semantic Validity

*"Is the content meaningful and faithful?"*
These tests assess whether the persona's content is grounded, coherent, and substantive. More expensive than structural tests but critical.

### D4. Factual Grounding

**What it tests:** Does every claim in the persona trace back to actual source data (CRM records, interview transcripts, analytics, survey responses)? Are there fabricated details?

**Why it matters:** A persona that invents plausible-sounding but false details is worse than useless — it creates confident wrong assumptions. This is the #1 failure mode identified across your research corpus.

**Automated methods:**
- Citation verification: persona includes source references; validate they exist and say what's claimed
- Retrieval-based grounding: embed persona claims, retrieve source chunks, measure semantic similarity
- Claim extraction + fact-checking pipeline: extract atomic claims from persona, verify each against source corpus
- Trustworthiness: Citation checking **High**. Retrieval similarity **Medium** (threshold-dependent). Claim extraction **Medium** (depends on claim granularity).

**Human methods:**
- Side-by-side review: persona on left, source data on right. Annotator marks each claim as Grounded / Ungrounded / Partially Grounded
- Trustworthiness: **High** — this is the gold standard for grounding evaluation

**Research grounding:** CharacterBench — Fact Accuracy is the consistently lowest-scoring dimension across all 18 models tested (2.1–3.0 on 5-point scale). This is the hardest thing for LLMs to get right.

---

### D5. Behavioral Consistency (Stability Under Repeated Probing)

**What it tests:** If you ask the persona the same question 100 times (with paraphrasing), do you get semantically equivalent answers? What's the variance?

**Why it matters:** An inconsistent persona produces different twin behavior depending on when you ask — unreliable for any downstream use.

**Automated methods:**
- Repeated query with temperature > 0, measure semantic similarity distribution across responses
- Embedding centroid + radius: tighter cluster = more consistent
- Contradiction detection across response set (NLI pairwise)
- Trustworthiness: **Medium** — measures surface consistency well, may miss deep behavioral inconsistency

**Human methods:**
- Present 10 responses to the same question, ask annotator: "Are these from the same persona?" and rate consistency 1-5
- Trustworthiness: **High** — PERSONA-CHAT benchmark: best models achieve 79% of human consistency (3.44/4.36)

**Research grounding:** Zhang et al. — consistency metric is foundational. Human ceiling at 4.36/5.0 sets realistic upper bound.

---

### D6. Distinctiveness (Inter-Persona Differentiation)

**What it tests:** Are Persona A and Persona B meaningfully different? Or are they the same generic persona with different names? Does the set of personas cover the intended diversity?

**Why it matters:** The Das Man paper mathematically proves that LLMs tend toward homogenization. If all your personas converge to the same "average person," the entire system is useless.

**Automated methods:**
- Pairwise embedding distance: compute cosine distance between persona embeddings; flag pairs below threshold
- Classifier confusion matrix: train a lightweight classifier to identify which persona generated a response; low accuracy = indistinguishable personas
- Variation ratio per attribute: what fraction of personas share the modal value? (Das Man metric)
- Topic modeling divergence: do personas emphasize different themes?
- Stylometric vector similarity: compute per-persona stylometric features (vocabulary richness, avg sentence length, punctuation frequency, formality markers, hedging rate) and measure pairwise distance. Tests whether personas have distinct *voices*, not just distinct *content*. (From eval_suggestions.md)
- Trustworthiness: Embedding distance **Medium** (captures surface differences, misses behavioral ones). Classifier **Medium-High** (directly tests distinguishability). Variation ratio **High** (mathematically grounded). Stylometric **Medium-High** (well-established in authorship attribution; captures voice distinctiveness that content embeddings miss).

**Human methods:**
- Paired comparison: present two persona profiles side-by-side, ask "How different are these on a 1-5 scale?" and "What specifically differs?"
- Blind sort: give annotator 5 conversation transcripts and 5 persona profiles, ask them to match. Success rate = distinctiveness score.
- Trustworthiness: **High** — PersonaEval shows humans achieve 90.8% on role identification (vs. 68.8% best LLM)

**Research grounding:** Das Man (Li, Li & Qiu 2025) — mathematical proof that next-token prediction guarantees homogenization. Immigration question: virtually all 395 subgroups show >95% probability on single answer for GPT-4. Variation ratio is a direct homogenization measure.

---

### D7. Demographic Coherence

**What it tests:** Do the persona's attributes form a plausible human profile? Not a Frankenstein of incompatible traits sampled independently from marginal distributions.

**Why it matters:** Li et al. (Promise with a Catch) showed that LLMs filling gaps in demographic data introduce stereotypical correlations. Real humans have coherent life stories; synthetic personas often don't.

**Automated methods:**
- Joint distribution validation: compare attribute combinations against Census or survey joint distributions (when available)
- Anomaly detection: train on real human profiles, flag synthetic personas that fall outside the learned distribution
- Co-occurrence plausibility: check if attribute pairs exist in real data (e.g., "PhD + age 19" is implausible)
- Trustworthiness: Joint distribution **High** (when reference data exists). Anomaly detection **Medium** (depends on training data quality).

**Human methods:**
- "Does this person exist?" test: present the full persona to an annotator, ask if this could be a real person. Rate plausibility 1-5.
- Trustworthiness: **Medium-High** — humans have strong intuition for plausibility but may have their own stereotypes

**Research grounding:** Li et al. (Promise with a Catch) — marginal vs. joint distribution problem. Census provides marginals only; LLMs filling gaps introduce stereotypical correlations.

---

### D8. Memory Consistency (Self-Recall)

**What it tests:** Can the persona accurately recall its own stated attributes when asked directly or indirectly? If the persona says "I manage a team of 12," does it remember this in conversation turn 47?

**Why it matters:** Memory failures are the most user-visible quality problem. A twin that forgets its own traits breaks trust immediately.

**Automated methods:**
- Direct recall probes: "How many people are on your team?" → compare answer to persona JSON
- Indirect recall probes: "Tell me about your management challenges" → check for consistency with team size, role, experience
- Long-context decay test: probe the same fact at turn 5, 20, 50, 100 — measure accuracy degradation
- Trustworthiness: Direct recall **High** (easily checkable). Indirect recall **Medium** (requires semantic matching). Decay test **High** (quantitative).

**Human methods:**
- Read conversation transcript, flag moments where the twin contradicts its persona's stated facts
- Trustworthiness: **High**

**Research grounding:** CharacterBench — Memory Consistency is a core dimension. SocialBench (cited in CharacterBench) — memory fails beyond 80 turns, creating "consistency collapse" in long conversations.

---

### D9. Knowledge Boundary Awareness

**What it tests:** Does the persona know what it should know AND not know what it shouldn't? A mid-level marketing manager shouldn't have detailed opinions on kernel architecture. A startup founder shouldn't know enterprise procurement workflows.

**Why it matters:** Omniscient personas are a hallmark of LLM artifacts. Real people have knowledge gaps, and those gaps define their behavior as much as their knowledge does.

**Automated methods:**
- Out-of-domain probes: ask questions outside the persona's expertise domain, check for appropriate uncertainty/deflection vs. confident hallucination
- Knowledge boundary classifier: given persona role + question topic, predict whether the persona should know the answer; compare to actual response confidence
- Trustworthiness: **Medium** — requires a reliable model of "what should this persona know," which is itself subjective

**Human methods:**
- Expert review: domain expert reads responses, flags moments where the persona demonstrates impossible knowledge for their stated role
- Trustworthiness: **High** — domain experts are excellent at spotting "this person wouldn't know that"

**Research grounding:** CharacterBench — Boundary Consistency dimension. Aher et al. — hyper-accuracy distortion: aligned models give inhumanly correct answers (IQR = 0 on factual questions where humans show massive variance).

---

### D10. Lexical vs. Semantic Generalization

**What it tests:** Does the persona maintain consistency when probed with paraphrased questions? Or does it only "remember" its traits when you use the exact same words as the persona definition?

**Why it matters:** Zhang et al. showed a 30% performance drop when persona attributes were paraphrased. If your persona only works with keyword matching, it's fragile and unreliable in real conversations.

**Automated methods:**
- Paraphrase probing: generate N paraphrases of each persona attribute, probe for consistency across all versions
- Zero-overlap probes: deliberately construct questions that share no vocabulary with the persona definition
- Trustworthiness: **Medium-High** — measures a real and important capability, but paraphrase quality matters

**Human methods:**
- Conversational probing: human interviewer tries to elicit persona traits through natural conversation without using the exact words from the profile
- Trustworthiness: **High** — this is how real users would interact

**Research grounding:** Zhang et al. (PERSONA-CHAT) — original persona Hits@1 = 0.509, revised (paraphrased) = 0.354. A 30% relative drop proves that keyword dependency is a major failure mode.

---

### D11. Profile Coverage

**What it tests:** Across a multi-turn conversation, what fraction of the persona's defined attributes actually get expressed or are consistent with the conversation?

**Why it matters:** A persona with 20 defined traits that only surfaces 5 in conversation is effectively a much simpler persona. Unexpressed traits are untested traits.

**Automated methods:**
- Attribute mention tracking: for each persona attribute, check if it was expressed or consistent with at least one conversational turn
- Coverage ratio: attributes_expressed / total_attributes across N conversations
- Topic steering test: deliberately steer conversation toward each attribute, measure success rate
- Trustworthiness: **Medium** — mention ≠ genuine integration; some attributes may not be relevant to all conversations

**Human methods:**
- Annotator reads persona + full conversation, marks which attributes were demonstrated vs. absent vs. contradicted
- Trustworthiness: **High**

**Research grounding:** Zhang et al. — "not all profile sentences get surfaced naturally in conversation; some aspects of a persona may never be expressed depending on topic."

---

### D12. Narrative Coherence

**What it tests:** Does the persona's story "make sense" as a whole? Not just logically consistent (D3), but narratively coherent — does the career trajectory match the skills? Does the communication style match the background? Does it feel like one person's life?

**Why it matters:** Individual attributes can all be valid but still not cohere into a believable human narrative. This is the gestalt quality.

**Automated methods:**
- LLM-as-judge with narrative coherence rubric: "Rate how well this persona reads as a single person's life story"
- Story arc validation: does the persona have a plausible progression (education → early career → current role)?
- Trustworthiness: **Low-Medium** — LLMs are poor judges of narrative quality (PersonaEval gap applies here)

**Human methods:**
- Holistic review: "Does this feel like one real person?" on 1-5 scale
- Character sketch test: can a writer turn this persona into a believable character sketch? Friction points reveal incoherence.
- Trustworthiness: **High** — this is fundamentally a human-judgment task

**Research grounding:** CharacterBench — Believability aspect (Human-Likeness + Engagement). This is the "dense dimension" that users evaluate on every interaction.

---

## Tier 3 — Distributional & Statistical Validity

*"Does this persona set represent reality?"*
Individual personas can be high-quality but the set can still fail if it doesn't match real population distributions.

### D13. Opinion / Response Diversity (Anti-Homogenization)

**What it tests:** Across your persona set, do personas express a genuine diversity of opinions? Or do they all converge on the same "average" position?

**Why it matters:** The Das Man paper proves mathematically that LLMs default to modal responses. If every persona in your set agrees on everything, you've just generated the same person N times.

**Automated methods:**
- Variation ratio per question: what fraction of personas give the modal answer? Compare to known human baselines.
- Response entropy: higher entropy = more diversity. Compare to real survey data.
- Wasserstein distance: between persona response distribution and real population distribution
- Modal collapse detection: flag any question where >80% of personas agree (in a diverse set, this should be rare)
- Trustworthiness: **High** — mathematically grounded, directly testable

**Human methods:**
- Not strictly needed — this is a statistical property best measured automatically
- But: qualitative review of "do these personas FEEL different?" can catch subtle homogenization that statistics miss

**Research grounding:** Das Man — GPT-4 shows >95% probability on single answer for virtually all 395 subgroups on immigration. Santurkar et al. (OpinionsQA) — >99% modal collapse on most questions. This is one of the most well-documented failure modes.

---

### D14. Variance Fidelity

**What it tests:** Does the spread of persona responses match the spread of real human responses? Not just the center (mean/median), but the tails?

**Why it matters:** Even if the average persona is correct, zero variance means you've lost all the information in the tails — which is often where the most interesting customer insights live.

**Automated methods:**
- IQR comparison: compare interquartile range of persona responses to human baseline on same questions
- Distribution shape tests (K-S test, Anderson-Darling): does the full distribution match?
- Tail coverage: are extreme positions represented at appropriate rates?
- Trustworthiness: **High** — purely statistical, requires reference human data

**Human methods:**
- Not needed for measurement — but human review of outlier personas can verify they're plausible extremes, not noise

**Research grounding:** Aher et al. — hyper-accuracy distortion: GPT-4 IQR = 0 on questions where human IQR = 532. This is catastrophic variance compression. Bisbee et al. — 0.5–1.0 SDs smaller variance on 100-point thermometer scales.

---

### D15. Structural Consistency Across Aggregation Levels

**What it tests:** If you query "female marketing managers" as a group, does the response match what you get by aggregating individual female marketing manager personas?

**Why it matters:** The Das Man paper found that querying "female" directly ≠ aggregating all female subgroup personas. The dominant answer actually FLIPS between aggregation levels. This means your system gives contradictory results depending on how you query it.

**Automated methods:**
- Cross-aggregation test: query the group, query individuals, aggregate individuals, compare
- Consistency ratio: fraction of attributes where group-level and individual-aggregate agree
- Trustworthiness: **High** — deterministic comparison

**Human methods:**
- Not needed — this is a structural/statistical test

**Research grounding:** Das Man — "Structural inconsistency: querying 'female' directly ≠ aggregating all female subgroups. Dominant answer FLIPS between aggregation levels."

---

### D16. Minority Viewpoint Preservation

**What it tests:** Are minority opinions within each demographic subgroup represented? Or has the system erased all but the majority view?

**Why it matters:** Real populations contain dissenters. A conservative-majority group still has progressives in it. If your personas erase minority viewpoints, you lose the tail of the distribution that often drives market disruption.

**Automated methods:**
- Within-group entropy: for each demographic subgroup, measure opinion diversity
- Minority representation rate: for each subgroup, check if the minority position (known from survey data) appears at approximately the correct rate
- Trustworthiness: **High** — requires reference human data but measurement is straightforward

**Human methods:**
- Qualitative review: "Do any of these personas represent a minority viewpoint within their group?"
- Trustworthiness: **Medium** — depends on annotator's awareness of actual minority distributions

**Research grounding:** Das Man — "Minority viewpoints within subgroups are systematically erased." Santurkar et al. — consistently underrepresented groups: 65+, Mormon, widowed, high religious attendance.

---

### D17. Calibration

**What it tests:** When the persona expresses confidence in something, is that confidence warranted? A persona that says "I'm certain I'd buy this" should actually buy it at high rates; one that says "maybe" should be closer to 50/50.

**Why it matters:** Miscalibrated personas give you false precision. You think you know what they'd do, but the confidence signals are noise.

**Automated methods:**
- Expected Calibration Error (ECE): bin responses by confidence level, measure accuracy per bin
- Reliability diagrams: visual calibration check
- Platt scaling / temperature calibration: can post-hoc calibration fix it?
- Trustworthiness: **High** — well-established statistical methodology

**Human methods:**
- Compare persona stated confidence to actual human behavior rates on the same questions
- Trustworthiness: **High** — but requires real human reference data

**Research grounding:** Economic Choice Labs — ECE ~0.15 for LLM-trained models vs ~0.08 for human-trained. An 87% calibration degradation. "Right answers for wrong reasons, meaningless confidence signals."

---

### D18. Joint Distribution Fidelity

**What it tests:** Do the correlations between persona attributes match real-world correlations? Or has the LLM introduced stereotypical or random correlations?

**Why it matters:** Real humans have complex, sometimes surprising attribute correlations. LLMs tend to either stereotypically correlate (PhD → liberal → urban) or randomly decouple attributes. Both are wrong.

**Automated methods:**
- Correlation matrix comparison: compute attribute correlations across persona set, compare to reference population correlation matrix
- Mutual information analysis: which attributes are correlated in the persona set vs. reality?
- Stereotypical correlation detection: flag attribute pairs with correlation significantly higher than reference data
- Trustworthiness: **High** — mathematical, but requires good reference data

**Human methods:**
- Expert review of surprising correlations: "Is it plausible that 80% of your budget-conscious personas are also early technology adopters?"
- Trustworthiness: **Medium** — subject to reviewer's own biases about what correlations are "normal"

**Research grounding:** Li et al. (Promise with a Catch) — "Census provides marginals only; LLMs filling gaps introduce stereotypical correlations." Cross-model test showed bias persists regardless of which model generates vs. simulates.

---

## Tier 4 — Bias & Safety Validity

*"Is the system producing fair, unbiased, representative personas?"*
These dimensions specifically test for known systematic distortions that LLMs introduce.

### D19. RLHF Positivity Bias Detection

**What it tests:** Does the persona skew systematically toward optimistic, prosocial, progressive descriptions? Are negative life experiences, challenging circumstances, and cynical viewpoints underrepresented?

**Why it matters:** Li et al. proved that RLHF-trained models produce personas that would make Democrats win EVERY US state. The bias is directional, consistent, and large enough to be catastrophic for any research application.

**Automated methods:**
- Sentiment analysis: compute sentiment distribution across persona set; compare to expected baseline
- Valence audit: ratio of positive to negative descriptors. Real populations aren't 80/20 positive.
- "Life challenge" representation: count personas mentioning financial hardship, health issues, job loss, family conflict, etc.
- TextBlob / VADER polarity across generated persona descriptions
- Trustworthiness: **Medium-High** — sentiment analysis is a blunt tool but catches the big signal

**Human methods:**
- "Sunshine audit": annotator reads 20 personas, flags whether they feel unrealistically positive
- Trustworthiness: **High** — humans immediately notice the "everyone is thriving" problem

**Research grounding:** Li et al. — "Critically absent: terms reflecting life challenges, social difficulties, negative experiences, hardship." Subjectivity and sentiment polarity increase monotonically with LLM-generated content. Yi-34B exception shows bias direction is training-dependent but presence is universal.

---

### D20. Sycophancy Resistance

**What it tests:** Does the persona/twin resist agreeing with everything the user says? Or does it shift its stated positions to match whoever is asking?

**Why it matters:** A persona that agrees with the interviewer is useless for research. RLHF specifically trains models to be agreeable, creating personas that tell you what you want to hear.

**Automated methods:**
- Opinion shift test: ask the same persona a question, then present a counter-argument and re-ask. Measure position shift.
- Leading question battery: "Don't you think X?" where X contradicts the persona's profile. Measure agreement rate.
- Cross-interviewer consistency: same persona, different interviewer tones (agreeable vs. challenging). Responses should be stable.
- Trustworthiness: **Medium-High** — tests a real phenomenon, but designing good leading questions requires care

**Human methods:**
- Adversarial interview: human interviewer deliberately pushes back on persona positions, measures how firmly the persona holds its ground
- Trustworthiness: **High**

**Research grounding:** Perez et al. — "largest models match user views >90% of the time." This is a known, measured, severe problem.

---

### D21. WEIRD Bias Detection

**What it tests:** Does the persona set overrepresent Western, Educated, Industrialized, Rich, Democratic perspectives? Even when configured for other demographics?

**Why it matters:** LLM training data is WEIRD-skewed. Personas conditioned on non-WEIRD demographics may still exhibit WEIRD values underneath.

**Automated methods:**
- Cross-cultural value surveys: run personas through validated instruments (World Values Survey questions) and compare to actual country-level data
- Language analysis: check for culture-specific assumptions (individualism vs. collectivism markers)
- Trustworthiness: **Medium** — depends on having good reference data for non-WEIRD populations

**Human methods:**
- Cultural expert review: someone from the target culture reviews personas for cultural authenticity
- Trustworthiness: **High** — but expensive and hard to scale

**Research grounding:** Boelaert et al. — "model outputs essentially same distribution regardless of persona" across WVS data in 5 countries. Hartmann et al. — ChatGPT "consistently pro-environmental, left-libertarian across 4 languages."

---

### D22. Hyper-Accuracy Distortion Detection

**What it tests:** Does the persona know MORE than a real human in that role would know? Does it give perfect answers to factual questions where real humans show variance and error?

**Why it matters:** A persona that knows everything isn't simulating a human — it's simulating an encyclopedia with a personality wrapper. This breaks any research that depends on realistic knowledge gaps.

**Automated methods:**
- Factual question battery with known human accuracy rates: compare persona accuracy to human baseline. If persona is significantly MORE accurate, flag it.
- Confidence-accuracy mismatch: personas shouldn't be confidently correct on questions humans get wrong
- IQR comparison on factual questions: IQR near 0 = hyper-accuracy
- Trustworthiness: **High** — requires human baseline data but measurement is straightforward

**Human methods:**
- "Too perfect" audit: domain expert reads responses, flags answers that are suspiciously accurate for the claimed role
- Trustworthiness: **High**

**Research grounding:** Aher et al. — GPT-4 gives exact correct answer with IQR of 0 for 8/10 factual questions. "Hyper-accuracy increases monotonically with alignment." Human median on aluminum melting point = 190 with IQR = 532; LLM median = 660 (correct), IQR = 0.

---

### D23. Stereotype Amplification Detection

**What it tests:** Does persona generation amplify stereotypes beyond their real-world prevalence? E.g., are all "female engineer" personas also described as "collaborative" and "empathetic" at rates exceeding reality?

**Why it matters:** Stereotyped personas reinforce biased decision-making rather than reflecting actual market diversity.

**Automated methods:**
- Attribute frequency by demographic: compare trait prevalence in generated personas to survey baselines
- Stereotype pair detection: check for over-correlated demographic-trait pairs (e.g., age → tech-averse)
- Persona-assignment bias test (Gupta et al.): measure if assigning a demographic persona changes behavior on unrelated tasks
- Trustworthiness: **Medium-High** — requires baseline data on actual trait prevalence

**Human methods:**
- Bias audit: diverse panel reviews personas for stereotypical portrayals
- Trustworthiness: **High** — but reviewers need to be diverse themselves

**Research grounding:** Gupta et al. — "Black person" persona leads LLM to abstain from math questions. 80% of personas exhibit measurable bias. Even GPT-4-Turbo: problematic bias in 42% of personas.

---

### D24. Negative Experience Representation

**What it tests:** Does the persona set include people with difficult life circumstances — financial stress, health problems, job loss, discrimination, family conflict, addiction, grief?

**Why it matters:** These experiences are statistically prevalent in real populations. A persona set without them is sanitized to the point of uselessness for any product serving real humans.

**Automated methods:**
- Adversity lexicon matching: check for mentions of challenges, difficulties, negative life events
- Adversity distribution: what % of personas include negative experiences? Compare to known prevalence rates (e.g., ~20% of US adults report mental health challenges)
- Trustworthiness: **Medium** — lexical matching is crude; some challenges are described subtly

**Human methods:**
- "Hardship audit": annotator reads personas, rates whether the set collectively represents the full range of human experience including difficulties
- Trustworthiness: **High**

**Research grounding:** Li et al. — word cloud analysis of LLM-generated personas shows prevalence of "love", "proud", "family", "community" and critical absence of challenge-related terms.

---

## Tier 5 — Behavioral & Interactive Validity

*"Does the persona behave correctly in conversation?"*
These dimensions require actually running conversations with the twin and evaluating behavior.

### D25. Emotional Self-Regulation

**What it tests:** Does the persona maintain appropriate emotional tone? Does it modulate emotion based on context rather than being emotionally flat or randomly emotional?

**Why it matters:** Emotionally flat personas produce biased qualitative data. Erratic emotional personas break trust.

**Automated methods:**
- Emotion detection (GoEmotions, fine-tuned classifier) across turns: measure emotional range and appropriateness
- Emotional consistency check: persona's emotional baseline should match its described personality (e.g., "optimistic" persona shouldn't be consistently negative)
- Trustworthiness: **Medium** — emotion detection models have known limitations

**Human methods:**
- Annotator rates emotional appropriateness per turn and overall emotional coherence
- Trustworthiness: **High** — CharacterBench identifies Emotion as one of the weakest aspects but most valued by users

**Research grounding:** CharacterBench — Emotion is one of the two weakest aspects across all 18 models. "Users notice flat emotional responses and low-engagement characters immediately."

---

### D26. Empathetic Responsiveness

**What it tests:** Does the persona respond appropriately to emotional cues from the conversational partner? Does it acknowledge sadness, share excitement, or respond to frustration?

**Why it matters:** In twin chat (simulated user research), an empathy-deficient persona produces unnatural conversation dynamics that bias the research findings.

**Automated methods:**
- Empathy probes: present emotional scenarios, check for appropriate acknowledgment
- Response appropriateness classifier: does the response match the emotional valence of the prompt?
- Trustworthiness: **Medium** — classifiers can detect gross mismatches but miss subtlety

**Human methods:**
- Conversational partner rates empathy quality on standardized rubric
- Trustworthiness: **High**

**Research grounding:** CharacterBench — Empathetic Responsiveness dimension.

---

### D27. Moral Stability

**What it tests:** Does the persona maintain consistent ethical positions under normal conversational conditions? Does a "values-driven" persona consistently reference those values?

**Why it matters:** Moral inconsistency is highly visible and breaks trust. A persona that's ethical in one turn and amoral in the next is incoherent.

**Automated methods:**
- Moral Foundations Questionnaire: run the persona through it multiple times, measure consistency
- Ethical dilemma battery: present standard dilemmas, check for consistent moral framework
- Trustworthiness: **Medium** — captures stated positions, not deep moral reasoning

**Human methods:**
- Ethicist review: are the persona's moral positions internally consistent and plausible for their described background?
- Trustworthiness: **High**

**Research grounding:** CharacterBench — Morality: Stability dimension.

---

### D28. Moral Robustness (Adversarial Ethics)

**What it tests:** Does the persona maintain its ethical positions when pressured? Does it resist manipulation attempts to change its moral stance?

**Why it matters:** A persona that abandons its values under social pressure isn't simulating a real person — it's simulating an agreeable chatbot.

**Automated methods:**
- Adversarial moral probing: present social pressure scenarios, measure position shift
- Jailbreak-style ethics tests: attempt to get the persona to violate its stated values through role-play scenarios or authority pressure
- Trustworthiness: **Medium** — hard to automate realistic social pressure

**Human methods:**
- Red team interview: skilled interviewer attempts to get persona to break character on moral positions
- Trustworthiness: **High**

**Research grounding:** CharacterBench — Morality: Robustness dimension. SimulateBench — "focuses on consistency and robustness under perturbation (adversarial probes of character stability)."

---

### D29. Refusal Behavior

**What it tests:** Does the persona appropriately decline to answer questions outside its scope? Does a B2B marketing persona refuse to speculate about nuclear physics rather than confidently making things up?

**Why it matters:** The opposite of hyper-accuracy distortion — personas should have realistic limitations and express them naturally.

**Automated methods:**
- Out-of-scope question battery: ask questions progressively further from the persona's domain, measure the point at which it starts refusing vs. hallucinating
- Refusal quality assessment: when it does refuse, is it natural ("I'm not really the right person to ask about that") or robotic ("I cannot answer that question")?
- Trustworthiness: **Medium** — detecting refusal is easy; assessing quality is hard automatically

**Human methods:**
- Naturalness rating of refusals: "Does this sound like a real person declining to answer?"
- Boundary testing interview: push the persona outside its comfort zone, evaluate refusal behavior
- Trustworthiness: **High**

**Research grounding:** CharacterBench — Boundary Consistency dimension. Related to D9 (Knowledge Boundary Awareness) but tested in conversation rather than in the persona definition.

---

### D30. Adversarial Robustness (Character Stability Under Attack)

**What it tests:** Can the persona be broken out of character through adversarial prompting? Does it resist attempts to make it "drop the act"?

**Why it matters:** In production, users will (intentionally or not) pressure the twin in unexpected ways. A persona that breaks under pressure is unreliable.

**Automated methods:**
- Adversarial prompt suite: jailbreaks, role-play overrides ("ignore your persona and..."), identity confusion attempts
- Character leakage detection: measure how often the underlying LLM "shows through" vs. the persona maintaining control
- "One Token to Fool" test: minimal perturbation attacks on persona-conditioning prompts
- Trustworthiness: **Medium** — automated adversarial testing catches known attacks but not novel ones

**Human methods:**
- Red team assessment: skilled attackers try to break the persona
- Trustworthiness: **High**

**Research grounding:** "One Token to Fool" (cited in PersonaEval) — "demonstrates that a single token change can game LLM evaluators." SimulateBench — perturbation-based robustness testing.

---

### D31. Recovery Behavior

**What it tests:** After being pushed off-character (by a confusing question, an adversarial probe, or a system error), does the persona return to character naturally?

**Why it matters:** No system is perfect — the question isn't whether the persona ever breaks, but whether it recovers gracefully.

**Automated methods:**
- Break-and-recover test: deliberately destabilize the persona, then continue normal conversation. Measure how many turns until behavior returns to baseline.
- Post-perturbation consistency check: compare pre-break and post-break response distributions
- Trustworthiness: **Medium** — requires a reliable "break" mechanism and consistent measurement

**Human methods:**
- Annotator reviews post-recovery conversation for naturalness and character consistency
- Trustworthiness: **High**

**Research grounding:** No direct paper — but this is a practical production concern. Related to SocialBench's finding that consistency collapses beyond 80 turns.

---

### D32. Engagement & Human-Likeness

**What it tests:** Does the persona feel like talking to a real person? Is it engaging, natural, and interesting — or stilted, robotic, and formulaic?

**Why it matters:** This is the ultimate user-facing quality metric. A persona can pass every other test and still feel obviously artificial.

**Automated methods:**
- Perplexity analysis: very low perplexity responses are likely formulaic; some variance indicates more natural language
- Response diversity metrics: lexical diversity, syntactic variety across turns
- Turing-style classifier: can a trained model distinguish persona responses from real human responses?
- Trustworthiness: **Low-Medium** — these are proxy metrics; none directly measures "feels human"

**Human methods:**
- Turing test: blind evaluation — is this a real person or AI? Track deception rate.
- Engagingness rating: "How much did you enjoy this conversation?" 1-5 scale
- Trustworthiness: **High** — this is definitionally a human-judgment metric

**Research grounding:** CharacterBench — Believability aspect (Human-Likeness + Engagement). Aher et al. — the Turing Experiment framework directly tests whether simulated participants are indistinguishable from real ones.

---

### D33. Engagingness-Consistency Tradeoff

**What it tests:** Does optimizing for consistency make the persona boring? Does optimizing for engagement introduce inconsistencies? Where is the Pareto frontier?

**Why it matters:** Zhang et al. identified this as a fundamental tradeoff. Any eval system needs to measure both and understand their relationship.

**Automated methods:**
- Joint scoring: measure consistency AND engagement independently, plot the tradeoff curve
- Pareto analysis across persona variants: which configurations are Pareto-optimal?
- Trustworthiness: **Medium** — each individual metric has its own trustworthiness limitations

**Human methods:**
- Dual-axis annotation: rate both consistency and engagement per conversation, analyze correlation
- Trustworthiness: **High**

**Research grounding:** Zhang et al. — "more engaging responses are sometimes less consistent with the persona profile; models that stay rigidly on-persona can seem stilted."

---

### D34. Multi-Turn Coherence Decay

**What it tests:** How does persona quality degrade over the length of a conversation? At what turn count does coherence break down?

**Why it matters:** Most eval happens on short conversations. Production conversations can be 50+ turns. If quality degrades at turn 20, your short-conversation evals are meaningless.

**Automated methods:**
- Sliding-window consistency: measure persona consistency in turns 1-10, 10-20, 20-30, etc.
- Decay curve: plot quality metric vs. turn count
- Critical turn detection: identify the turn where quality drops below threshold
- Trustworthiness: **Medium-High** — methodologically sound, captures a real phenomenon

**Human methods:**
- Long-conversation annotation: annotators read full 50+ turn conversations and flag where quality degrades
- Trustworthiness: **High**

**Research grounding:** SocialBench (cited in CharacterBench) — "memory fails beyond 80 turns, creating consistency collapse in long conversations."

---

## Tier 6 — Functional & System Validity

*"Does the pipeline work correctly at the system level?"*
These dimensions test the engineering and operational aspects.

### D35. Role Identifiability

**What it tests:** Can an evaluator (human or LLM) correctly identify which persona is speaking from a conversation transcript alone?

**Why it matters:** If personas aren't identifiable, they're not distinct enough to be useful. This is the most direct test of whether your personas are actually different from each other.

**Automated methods:**
- LLM identification test: present transcript, ask model to identify the persona from a lineup
- Classifier accuracy: train on persona-transcript pairs, test on held-out conversations
- Trustworthiness: **Low-Medium** — PersonaEval shows best LLM achieves 68.8% vs human 90.8%. LLM judges are unreliable here.

**Human methods:**
- Blind matching: annotators match transcripts to persona profiles
- Trustworthiness: **High** — humans achieve 90.8% accuracy

**Research grounding:** PersonaEval — this is the central finding. "The LLM-as-judge paradigm for roleplay evaluation is unvalidated." GPT-4o achieves 40.9% on a task humans do at 90.8%.

---

### D36. Predictive Validity

**What it tests:** Do the persona's responses predict what real humans in that segment would actually do? If the persona says "I'd switch to this product," do real humans in that segment actually switch?

**Why it matters:** This is the ultimate validation — does the persona provide useful predictive signal about real human behavior?

**Automated methods:**
- Holdout validation: generate persona predictions, compare to held-out real human behavioral data
- A/B correlation: do persona preferences correlate with real A/B test outcomes?
- Regression coefficient comparison: do persona-derived coefficients match real-data-derived coefficients? (Bisbee et al. found 32% have FLIPPED SIGNS)
- Trustworthiness: **High** — but requires real behavioral data for comparison, which is expensive

**Human methods:**
- Expert prediction comparison: do domain experts agree with the persona's predicted behaviors?
- Trustworthiness: **Medium** — experts have their own biases

**Research grounding:** Economic Choice Labs — 79-80% accuracy is achievable but with 87% calibration degradation. Bisbee et al. — "48% of regression coefficients significantly different from ANES; 32% have FLIPPED SIGNS." This is the dimension where the gap between promise and reality is largest.

---

### D37. Temporal Stability

**What it tests:** Does the same pipeline produce comparable personas today vs. last week vs. last month? Holding inputs constant, how much does output drift?

**Why it matters:** Model updates, API changes, prompt drift, and infrastructure changes can all cause silent quality degradation.

**Automated methods:**
- Golden set re-run: regenerate from the same inputs periodically, measure delta
- Semantic drift metric: embedding distance between current and historical outputs
- Distribution shift detection (PSI, K-L divergence) on key attributes over time
- Trustworthiness: **High** — temporal comparison is straightforward

**Human methods:**
- Periodic human review of same golden set outputs: "Has quality changed?"
- Trustworthiness: **High** — but expensive to do frequently

**Research grounding:** Bisbee et al. — "Temporal instability: identical prompts produce different results 3 months apart."

---

### D38. Cross-Model Stability

**What it tests:** When you switch from Sonnet 4.6 to Opus 4.6 to a future model, do persona outputs maintain quality? What changes?

**Why it matters:** You will change models. Model providers will update models. Your eval suite needs to catch quality regressions across model transitions.

**Automated methods:**
- Model comparison suite: run same inputs through multiple models, compare outputs across all other dimensions
- Regression detection: new model must score within X% of baseline model on all dimensions
- Trustworthiness: **High** — this is the core purpose of the eval suite

**Human methods:**
- Comparative review: annotator sees outputs from both models (blinded), rates which is better and why
- Trustworthiness: **High**

**Research grounding:** Li et al. (Promise with a Catch) — "Cross-model test showed bias persists regardless of which model generates vs. simulates." Universal across all 6 models tested.

---

### D39. Reproducibility / Determinism

**What it tests:** Given identical inputs and settings, how much do outputs vary across runs? What's the acceptable variance band?

**Why it matters:** If your pipeline produces wildly different personas each run, you can't reliably test, debug, or validate.

**Automated methods:**
- N-run comparison: generate same persona 10 times, measure variance across runs
- Acceptable variance band: define thresholds per attribute (some variance is expected for narrative fields; zero variance expected for structured fields)
- Trustworthiness: **High** — purely statistical

**Human methods:**
- Not strictly needed — but human review of "are these all the same persona?" can catch semantic drift that statistics miss

**Research grounding:** General best practice. Related to D37 (Temporal Stability) but focused on within-session reproducibility.

---

### D40. Cost/Latency Bounds

**What it tests:** Does persona generation stay within acceptable cost and time budgets? Do certain inputs trigger pathologically expensive runs?

**Why it matters:** A persona that costs $50 and takes 10 minutes to generate isn't viable at scale, even if it's high quality.

**Automated methods:**
- Token counting: input + output tokens per generation, aggregated across pipeline steps
- Latency profiling: end-to-end generation time, with breakdown by pipeline stage
- Cost regression detection: alert if average cost per persona increases by >10%
- Trustworthiness: **High** — objective measurement

**Human methods:**
- Not needed — this is fully automatable

**Research grounding:** Open question from the original spec: "Do we eval cost/latency alongside quality?"

---

### D41. Degradation Detection (Production Drift)

**What it tests:** Are production outputs gradually getting worse over time? Are there step-function quality drops (e.g., after a model update)?

**Why it matters:** Quality regressions in production are invisible without active monitoring. By the time a customer complains, the problem has been live for weeks.

**Automated methods:**
- Continuous scoring: sample production outputs daily/weekly, run through eval suite
- Statistical process control: track metrics over time with control limits (±1σ, ±2σ, ±3σ)
- Anomaly detection: flag days where metrics deviate significantly from historical baseline
- Trustworthiness: **High** — standard monitoring practice

**Human methods:**
- Weekly spot-check: human reviews a random sample of production outputs
- Trustworthiness: **High**

**Research grounding:** Success metric from the original spec: "Weekly production sample shows no drift > 1σ on core metrics."

---

## Tier 7 — Generation-Specific Validity

*"Is the persona generation process itself introducing problems?"*
These dimensions are specific to the act of generating personas, distinct from evaluating finished personas.

### D42. Generation Bias Amplification

**What it tests:** As more of the persona is LLM-generated (vs. sourced from real data), does quality degrade? Is the relationship monotonic?

**Why it matters:** Li et al. proved that "as LLM generates more persona content, simulation accuracy monotonically decreases." This means every additional generated field is a potential quality loss.

**Automated methods:**
- Ablation study: generate personas with varying levels of LLM contribution (tabular → objective tabular → subjective tabular → descriptive). Measure quality at each level.
- "More LLM = worse?" curve: plot quality metrics against LLM contribution level
- Trustworthiness: **High** — well-established methodology from the paper

**Human methods:**
- Comparative review: human rates personas at each generation level for quality and plausibility
- Trustworthiness: **High**

**Research grounding:** Li et al. (Promise with a Catch) — central finding. Taxonomy: Meta Personas → Objective Tabular → Subjective Tabular → Descriptive. Quality degrades monotonically with LLM involvement.

---

### D43. Source Data Fidelity

**What it tests:** After the pipeline processes source data (CRM, interviews, analytics), how much of the original signal survives in the persona? Is the persona a faithful compression or a lossy distortion?

**Why it matters:** The whole point of data-grounded personas is that they reflect reality. If the pipeline loses the signal, you might as well generate from scratch.

**Automated methods:**
- Information retention score: extract key facts from source data, check which survive in the persona
- Retrieval-based comparison: embed both source and persona, measure information overlap
- Signal-to-noise ratio: what fraction of persona content maps to source data vs. LLM-generated filler?
- Trustworthiness: **Medium-High** — depends on quality of fact extraction

**Human methods:**
- Side-by-side review: source data + persona, annotator marks what was preserved, lost, or fabricated
- Trustworthiness: **High** — the gold standard

**Research grounding:** Li et al. — "Grounding in real data (Meta Personas) outperforms LLM enrichment." Argyle et al. — "The promise of silicon samples hinges on realistic conditioning."

---

### D44. Sparse vs. Dense Dimension Coverage

**What it tests:** Does the eval suite appropriately test both frequently-surfacing traits (dense: communication style, emotional tone) AND rarely-surfacing traits (sparse: specific factual recall, edge-case opinions)?

**Why it matters:** CharacterBench showed that "treating all dimensions as equally elicitable leads to misleading aggregate scores." If you only test dense dimensions, you'll miss the sparse failures that surprise users.

**Automated methods:**
- Dimension frequency analysis: track how often each persona attribute surfaces in test conversations
- Sparse dimension forced probing: deliberately construct scenarios that elicit rare attributes
- Coverage matrix: attribute × conversation, marking which attributes were tested
- Trustworthiness: **Medium-High** — methodologically sound but requires thoughtful test design

**Human methods:**
- Test design review: expert reviews eval suite to ensure sparse dimensions aren't neglected
- Trustworthiness: **High**

**Research grounding:** CharacterBench — "Sparse dimensions evaluated infrequently because opportunities to test them arise rarely in natural dialogue. Dense dimensions evaluated frequently across most turns."

---

## Tier 8 — Cross-Channel & Efficiency Validity

*"Does the persona hold up across contexts, and can evaluators work with it efficiently?"*
These dimensions come from eval_suggestions.md and cover cross-context coherence and evaluator efficiency.

### D45. Time-to-Identify (Evaluator Efficiency)

**What it tests:** How quickly can a human evaluator identify which persona is speaking from a conversation transcript? Faster identification = stronger distinctiveness signal.

**Why it matters:** D35 (Role Identifiability) measures *whether* someone can identify the persona. This measures *how fast* — a proxy for distinctiveness strength. A persona that takes 30 seconds to identify is more distinct than one that takes 5 minutes of careful reading. In production annotation workflows, this also determines eval throughput.

**Automated methods:**
- Not directly automatable — this is a human speed metric
- Proxy: classifier confidence score. A classifier that identifies the persona with high confidence on fewer tokens has effectively "fast" identification.
- Trustworthiness: Proxy is **Medium** — correlation between classifier confidence and human speed is plausible but unvalidated

**Human methods:**
- Timed blind matching: present transcript, start timer, annotator identifies persona from lineup. Record time-to-correct-answer.
- Trustworthiness: **High** — direct measurement

**Research grounding:** eval_suggestions.md — "Stopwatch on evaluator." Related to PersonaEval's finding that humans achieve 90.8% accuracy — but speed adds a second axis beyond just accuracy.

---

### D46. Cross-Platform Coherence

**What it tests:** Does the same persona behave consistently across different interaction channels? If the persona responds via chat, email, voice transcript, and support ticket — are those responses recognizably the same person?

**Why it matters:** Real humans behave somewhat differently across channels (more formal in email, more casual in chat) but remain recognizably themselves. A persona should do the same. If the persona is a completely different character in email vs. chat, it's not coherent.

**Automated methods:**
- Multi-channel generation: generate responses from the same persona in different channel contexts (chat, email, formal document, support ticket)
- Cross-channel embedding similarity: embed responses across channels, measure whether the persona cluster is tighter than the channel cluster (i.e., Persona A's chat and email should be more similar to each other than Persona A's chat and Persona B's chat)
- Style adaptation check: persona should adapt style (formality, length) to channel while maintaining voice (vocabulary, values, knowledge)
- Trustworthiness: **Medium** — requires careful experimental design to separate channel adaptation from persona inconsistency

**Human methods:**
- Cross-channel matching: same evaluator rates persona outputs across 2+ channels, assesses whether it's recognizably the same person
- "Same person?" test: present email + chat from same persona (unlabeled), ask if they're the same person
- Trustworthiness: **High**

**Research grounding:** eval_suggestions.md — "Same evaluator rates persona on 2+ channels." No direct academic reference, but this is a practical production concern for any omnichannel product.

---

## Meta-Dimension: Evaluator Validity

*"Can we trust our evaluation methods?"*

### M1. LLM-as-Judge Reliability

**What it tests:** When using an LLM (e.g., Claude Opus) as a judge, how well does it correlate with human judgment on each dimension?

**Why it matters:** This is the single most critical meta-question. PersonaEval shows a 22-point gap. CharacterBench shows CharacterJudge at 68% correlation vs GPT-4 at 40%. If your judge is unreliable, your entire eval suite is unreliable.

**How to validate:**
- Human annotation baseline: get human scores on a representative sample
- Judge correlation: measure Pearson/Spearman correlation between LLM judge and human scores per dimension
- Dimension-specific trust: LLM judges may be reliable on some dimensions (schema compliance) and unreliable on others (emotional coherence)
- Known biases to test for: position bias, verbosity bias, self-enhancement bias (Zheng et al. MT-Bench)

**Research grounding:** PersonaEval — 22-point gap. CharacterBench — CharacterJudge outperforms GPT-4 but still only 68% correlated. Son et al. — LLM evaluators miss factual inaccuracy and cultural misrepresentation. "One Token to Fool" — single token changes game LLM evaluators.

---

### M2. Judge Gaming Prevention

**What it tests:** Can the system being evaluated manipulate the judge to get higher scores without actually being better?

**Why it matters:** If the same model family generates personas and judges them, systematic blind spots will be shared. The judge literally cannot see failures that are inherent to its architecture.

**How to validate:**
- Cross-family judging: if generating with Claude, judge with GPT (and vice versa)
- Adversarial judge testing: deliberately submit known-bad personas, verify the judge catches them
- Judge calibration: known-quality outputs should get predictable scores
- Human override rate: how often do humans disagree with the judge on a random sample?

**Research grounding:** Open question from the original spec: "How do we prevent the judge model from gaming itself?"

---

### M3. Evaluation Metric Validity

**What it tests:** Do our automated metrics actually correlate with the thing we care about — useful, accurate personas?

**Why it matters:** Liu et al. 2016 showed that ALL standard dialogue metrics (BLEU, METEOR, ROUGE) have ZERO correlation with human judgment. We must not repeat this mistake with our own metrics.

**How to validate:**
- Metric-human correlation study: compute every automated metric on a sample, get human ratings, measure correlation
- Metric sensitivity analysis: do metrics actually change when quality changes?
- Metric gaming test: can you improve the metric without improving actual quality?

**Research grounding:** Liu et al. 2016 — "ALL standard automated metrics show zero or near-zero correlation with human judgments on the Ubuntu dialogue corpus." This is why response ranking replaced BLEU as the primary metric.

---

## Summary Table

| # | Dimension | Tier | Best Automated Method | Auto Trust | Best Human Method | Human Trust |
|---|-----------|------|----------------------|------------|-------------------|-------------|
| D1 | Schema Compliance | Structural | JSON Schema validation | High | — | — |
| D2 | Completeness | Structural | Null/empty + semantic emptiness | High | Spot-check filler | High |
| D3 | Internal Logical Consistency | Structural | Rule-based + NLI | Med-High | Full-persona review | High |
| D4 | Factual Grounding | Semantic | Claim extraction + retrieval | Medium | Side-by-side source review | High |
| D5 | Behavioral Consistency | Semantic | Embedding cluster tightness | Medium | Multi-response rating | High |
| D6 | Distinctiveness | Semantic | Variation ratio + classifier | Med-High | Blind sort matching | High |
| D7 | Demographic Coherence | Semantic | Joint distribution validation | High* | "Does this person exist?" | Med-High |
| D8 | Memory Consistency | Semantic | Direct/indirect recall probes | High/Med | Transcript contradiction flag | High |
| D9 | Knowledge Boundary Awareness | Semantic | Out-of-domain probes | Medium | Expert review | High |
| D10 | Lexical vs Semantic Generalization | Semantic | Paraphrase probing | Med-High | Conversational probing | High |
| D11 | Profile Coverage | Semantic | Attribute mention tracking | Medium | Annotator coverage marking | High |
| D12 | Narrative Coherence | Semantic | LLM narrative rubric | Low-Med | Holistic "one person?" review | High |
| D13 | Opinion Diversity | Distributional | Variation ratio + entropy | High | — | — |
| D14 | Variance Fidelity | Distributional | IQR + K-S test | High* | — | — |
| D15 | Structural Aggregation Consistency | Distributional | Cross-aggregation test | High | — | — |
| D16 | Minority Viewpoint Preservation | Distributional | Within-group entropy | High* | Qualitative diversity review | Medium |
| D17 | Calibration | Distributional | ECE + reliability diagrams | High* | — | — |
| D18 | Joint Distribution Fidelity | Distributional | Correlation matrix comparison | High* | Expert correlation review | Medium |
| D19 | RLHF Positivity Bias | Bias | Sentiment distribution analysis | Med-High | "Sunshine audit" | High |
| D20 | Sycophancy Resistance | Bias | Leading question battery | Med-High | Adversarial interview | High |
| D21 | WEIRD Bias | Bias | Cross-cultural value surveys | Medium | Cultural expert review | High |
| D22 | Hyper-Accuracy Distortion | Bias | Factual accuracy vs baseline | High* | "Too perfect" audit | High |
| D23 | Stereotype Amplification | Bias | Demographic-trait correlation | Med-High | Diverse panel audit | High |
| D24 | Negative Experience Representation | Bias | Adversity lexicon matching | Medium | "Hardship audit" | High |
| D25 | Emotional Self-Regulation | Behavioral | Emotion classifier across turns | Medium | Turn-level annotation | High |
| D26 | Empathetic Responsiveness | Behavioral | Empathy probes + classifier | Medium | Partner empathy rating | High |
| D27 | Moral Stability | Behavioral | Moral Foundations consistency | Medium | Ethicist review | High |
| D28 | Moral Robustness | Behavioral | Adversarial moral probing | Medium | Red team interview | High |
| D29 | Refusal Behavior | Behavioral | Out-of-scope question battery | Medium | Naturalness rating | High |
| D30 | Adversarial Robustness | Behavioral | Adversarial prompt suite | Medium | Red team assessment | High |
| D31 | Recovery Behavior | Behavioral | Break-and-recover test | Medium | Post-recovery annotation | High |
| D32 | Engagement & Human-Likeness | Behavioral | Turing-style classifier | Low-Med | Turing test + engagement rating | High |
| D33 | Engagingness-Consistency Tradeoff | Behavioral | Joint scoring + Pareto | Medium | Dual-axis annotation | High |
| D34 | Multi-Turn Coherence Decay | Behavioral | Sliding-window consistency | Med-High | Long-conversation annotation | High |
| D35 | Role Identifiability | Functional | LLM identification test | Low-Med | Blind matching | High |
| D36 | Predictive Validity | Functional | Holdout behavior comparison | High* | Expert prediction comparison | Medium |
| D37 | Temporal Stability | Functional | Golden set re-run + drift metrics | High | Periodic human review | High |
| D38 | Cross-Model Stability | Functional | Model comparison suite | High | Blinded comparative review | High |
| D39 | Reproducibility | Functional | N-run variance measurement | High | — | — |
| D40 | Cost/Latency Bounds | Functional | Token + time profiling | High | — | — |
| D41 | Degradation Detection | Functional | SPC + anomaly detection | High | Weekly spot-check | High |
| D42 | Generation Bias Amplification | Generation | Ablation across LLM involvement | High | Comparative level review | High |
| D43 | Source Data Fidelity | Generation | Information retention score | Med-High | Side-by-side source review | High |
| D44 | Sparse vs Dense Coverage | Generation | Dimension frequency analysis | Med-High | Test design expert review | High |
| D45 | Time-to-Identify | Cross-Channel | Classifier confidence proxy | Medium | Timed blind matching | High |
| D46 | Cross-Platform Coherence | Cross-Channel | Cross-channel embedding similarity | Medium | Cross-channel matching | High |
| M1 | LLM-as-Judge Reliability | Meta | Judge-human correlation | — | Human annotation baseline | High |
| M2 | Judge Gaming Prevention | Meta | Cross-family judging | — | Human override rate | High |
| M3 | Evaluation Metric Validity | Meta | Metric-human correlation | — | Metric gaming test | High |

\* = Requires reference human/real-world data to achieve stated trustworthiness level

---

## Key Patterns From This Analysis

### 1. Human judgment is consistently more trustworthy than automated methods

Across all 44+ dimensions, human evaluation is rated **High** trustworthiness in nearly every case. Automated methods range from High (structural/statistical) to Low (behavioral/narrative). This doesn't mean we can't automate — it means we need to validate our automated methods against human baselines before trusting them.

### 2. Structural and statistical tests are most automatable

Tiers 1 and 3 (structural validity, distributional validity) can be almost entirely automated with High trustworthiness. These should be the backbone of CI/PR gating.

### 3. Behavioral tests are hardest to automate reliably

Tier 5 (behavioral/interactive) dimensions consistently show Medium trustworthiness for automated methods. These require human validation at higher rates.

### 4. The LLM-as-judge problem is pervasive

At least 15 dimensions could theoretically use LLM-as-judge, but PersonaEval and CharacterBench demonstrate that general-purpose LLMs are poor judges of persona quality. A specialized evaluator (like CharacterJudge) helps but still only reaches 68% correlation.

### 5. Reference data is the bottleneck

Many High-trustworthiness automated methods (marked with *) require real human/population data for comparison. Without this reference data, their trustworthiness drops to Medium or lower. The golden set design is therefore the most critical architectural decision.

### 6. Sparse dimensions are systematically undertested

CharacterBench's sparse/dense insight implies that standard eval suites will overttest common traits and undertest rare ones — leading to false confidence in quality.
