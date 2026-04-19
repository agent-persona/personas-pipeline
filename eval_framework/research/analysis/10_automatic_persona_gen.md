# Jansen et al. — Automatic Persona Generation (APG)

**Sources:** 
- Jung, Salminen, Kwak, An & Jansen (2018). "Automatic Persona Generation (APG): A Rationale and Demonstration." CHIIR'18
- Salminen, Jansen, An, Kwak & Jung (2019). "Automatic Persona Generation for Online Content Creators." Springer, Personas: User Focused Design, Ch. 8

---

## Key Contributions

1. **APG as a deployed system, not just a method.** Full pipeline: Configuration → Collection → Persona Generation → Interaction. Beta-deployed with Al Jazeera, Qatar Airways, and SMEs by 2018.

2. **Democratization of personas.** Manual persona creation cost $80,000-$120,000 and took months. APG generates from millions of interactions in days.

3. **Combining empathy with analytics.** Personas are relatable but stale/expensive; analytics are granular but cognitively overwhelming. APG claims the intersection.

4. **Privacy-preserving by design.** Uses aggregated platform analytics (demographic buckets like [Female, 25-34, USA]), not individual-level data.

---

## Methods: The 6-Step Pipeline

1. **Data collection** via platform APIs (YouTube Analytics, Facebook Insights, Google Analytics)
2. **Data preparation** — matrix **V** (g × c): user groups × content items, with interaction metrics
3. **Pattern identification via NMF** (Non-negative Matrix Factorization) — discovers distinct behavioral patterns even within single demographic groups
4. **Impactful group identification** — selects demographic group with largest coefficient per NMF cluster
5. **Skeletal persona creation** — three attributes: gender, age, country from most impactful group
6. **Enrichment** — name (Census-based), photo (stock library), topics of interest, quotes (filtered social media comments), audience size (Facebook Marketing API), textual description (NLG templates)

### Data Sources
YouTube Analytics, Facebook Insights, Facebook Ads, Google Analytics

---

## Five Guiding Principles

1. **Consistency**: Independent elements must create coherent portrayal
2. **Relevance**: Information shown must be immediately useful
3. **Non-offensiveness**: Prevent harmful elements (balanced with authenticity)
4. **Authenticity**: Don't manipulate representations to deviate from data
5. **Context**: Make clear the persona represents a distribution/group

---

## Claims vs. Evidence

| Claim | Evidence |
|---|---|
| Faster (days vs. months) | System deployed — plausible but no timing study |
| Cheaper ($0 vs. $80-120K) | Asserted from practitioner discussions, not formally measured |
| More current (auto-updating) | Architecture supports periodic re-collection — demonstrated |
| Privacy-preserving | Aggregated data by design — true by construction |
| More accurate (behavioral grounding) | NMF on real data — mathematically grounded but no accuracy metric defined |
| Scalable | Platform-agnostic given V matrix — demonstrated for 3 platforms |

---

## Error Analysis / Caveats

1. **Aggregation problem**: Persona picks ONE demographic identity, but the underlying NMF cluster contains a distribution. "Samantha, 25 from New York" may share behavioral similarity with "a 42-year-old man from Doha."
2. **Quote selection bias**: User studies revealed quotes disproportionately influence perception. Hateful comments made a persona "perceived as a monster."
3. **Cross-platform identity mapping**: Currently single-platform — no entity resolution across YouTube + Twitter
4. **Image-demographic mismatch**: Ethnicity inferred probabilistically from country via Census
5. **Template-based descriptions**: Not true NLG — needs "generative text algorithms"
6. **No formal evaluation**: CHIIR paper is a 4-page demo. Book chapter is a "research agenda." No head-to-head comparison with manually-created personas.

---

## The Data Spine Concept

APG's "data spine" is the matrix **V** — the user-group-by-content interaction matrix. The persona is decomposed from real behavioral patterns via NMF, then dressed with humanizing attributes. Every element (name, image, topics, quotes) should be traceable to underlying data. Motto: **"Giving faces to data."**

---

## DFS Level 2 Reference A: Chapman & Milham (2006) — "The Personas' New Clothes"

**The single most important pre-LLM persona critique.**

### Core Arguments
1. **Population coverage is unknowable.** No way to assess what proportion of users a persona represents.
2. **Curse of dimensionality.** With 21 attributes at 0.5 base rate: (0.5)^21 = 0.000048% of population (~134 people in US).
3. **Non-falsifiability.** "Personas are admittedly fictional... no data can disprove a fictional construction."
4. **Validation only asserted, never demonstrated.** Only procedural claims found in literature.

### Empirical Follow-up: Chapman et al. (2008)
- Tested across **6 real survey datasets** (N=268 to N=10,307) with 10,000 random persona descriptions per dataset
- At 99th percentile, combining 9+ attributes → **0% of respondents matched** in 5/6 datasets
- Pearson's r for observed vs. predicted prevalence: 0.394-0.713

**Devastating implication:** Personas with typical detail level (15-25 attributes) almost certainly describe an empty population.

---

## DFS Level 2 Reference B: Salminen et al. (2020) — Persona Perception Scale (PPS)

**First validated psychometric instrument for measuring persona quality from end-user perspective.**

### 8 Constructs
| Construct | Sample Item |
|---|---|
| Credibility | "The persona seems like a real person" |
| Consistency | "The persona information seems consistent" |
| Completeness | "The persona profile seems complete" |
| Clarity | "Information is easy to understand" |
| Likability | "I find this persona likable" |
| Empathy | "I feel like I understand this persona" |
| Similarity | "This persona feels similar to me" |
| Willingness to Use | "This persona would improve my decisions" |

### Validation (n=412)
- EFA: 6-factor solution, 72.3% variance explained
- CFA: X²/df = 2.581, CFI = 0.943, RMSEA = 0.060
- All Cronbach's α > 0.779; factor loadings 0.616-0.919
- Invariant across gender and experience level

### Critical Insight
**Separates perceived credibility from actual accuracy.** "Credibility could be high for inaccurate personas if presented believably." Actual accuracy requires "hard metrics and quantitative analysis" — a "conceptually separate construct."

---

## DFS Level 3: Pruitt & Grudin (2003) — "Personas: Practice and Theory"

**The Microsoft approach that became the industry standard.**

- Start with large-sample market segmentation
- Enrich with field studies, focus groups, interviews
- "Foundation Documents" with copious footnotes linking characteristics to supporting data
- "Sanity check" site visits with matching users
- 5,000-person Persona User Panel for Windows personas

### The Adoption Paradox (from Friess 2012, Matthews et al. 2012)
- Designers valued the *process* of creating personas (engagement with data) but rarely invoked them during actual decision-making
- Practitioners used personas for **communication**, not design decisions
- ~1/3 of experienced practitioners found personas "abstract, impersonal, misleading, and distracting"
- **If the value lies in the creation process, automating it may remove the mechanism that makes personas useful**

---

## Relevance to Persona Accuracy Framework

### What "Grounding Quality" Meant Pre-LLM

| Level | Method | What It Measures |
|---|---|---|
| Process validation | Foundation documents, affinity sessions | "We followed rigorous process" (weakest) |
| Statistical grounding | Prevalence analysis (Chapman 2008) | Population coverage of attributes |
| Perceptual validation | PPS (Salminen 2020) | Do users find personas credible/useful? |
| Behavioral validation | Prediction studies (An et al. 2017) | Do personas predict user behavior? |

### The Dual Construct of Persona "Accuracy"
Pre-LLM era established that accuracy is at minimum dual:
- **Data fidelity**: Does the persona faithfully represent real user segments? (NMF, prevalence analysis)
- **Perceptual quality**: Do end users find it credible, consistent, empathy-inducing? (PPS)

LLM-generated personas add a third:
- **Behavioral fidelity**: Can the persona respond to novel queries consistently with its segment?

The pre-LLM literature provides measurement foundations for the first two but has nothing for the third — the distinctively new challenge of the LLM era.
