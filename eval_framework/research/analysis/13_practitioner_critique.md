# Practitioner Critiques of Synthetic User Research

## Primary: Speero — "Why I'm Not Sold on Synthetic User Research"
**Author:** Emma Travis (14 years UX) | August 2025

### Core Argument
"AI predicts the average, while humans do what's least expected." Synthetic research systematically misses contextual, emotional, contradictory, and irrational behavior.

### Evidence: Two Case Studies
1. **Diabetes product**: AI found reasonable issues (price sensitivity, vague marketing). Real users surfaced: DIY workarounds (ice packs/FRIO sleeves) and education gaps (didn't know insulin storage affects potency). "These insights weren't predictable. They were contextual."
2. **Predictive vs real heatmap**: AI predicted clicks on shopping cart, CTAs, bestsellers. Real data showed clicks clustered almost entirely around site search. "Same page, completely different story."

### The "Average" Problem
AI generates statistically probable responses. The actual insights that change product strategy are not predictable from prior text.

## NN/g Studies (Nielsen Norman Group)
- **Online learning**: Synthetic users claimed completing all courses. Real users: 3 of 7 (cited job changes)
- **Forums**: Synthetic users praised them. Real users found them "contrived and not useful"
- **Sycophancy**: AI generates overly favorable responses (Sharma et al. 2023)
- **Jeff Sauro/MeasuringU**: ChatGPT was too GOOD at tree testing — superhuman, not human-like

## UXtweak 182-Study Review (2026)
- "Lack of realistic variability is the most universal and ubiquitous bias"
- Emotional responses show "unrealistically encyclopedic" awareness with flattened affect
- "Journey bias" — overwhelming convergence on training-data-dominant answers
- Theory of Mind reasoning inferior to humans
- All remediation approaches (few-shot, CoT, RAG, fine-tuning) showed only modest gains
- Best-performing approaches required inputting expected results beforehand (defeating purpose)
- "Misleading believability" is the core problem

## Li et al. Million-Persona Study
- LLM personas predicted Democratic victories across ALL 2024 US states
- Systematic biases: environmental > economic, liberal arts > STEM, artistic > mainstream

## PersonaCite (CHI 2026)
- Grounds synthetic personas in voice-of-customer artifacts via RAG
- Constrains responses to retrieved evidence, abstains when evidence missing
- "Persona Provenance Cards" — emerging "grounded persona" approach

## Counter-Argument: Zuhlke (March 2026)
- Benchmark comparison is wrong: surveys have 81-85% test-retest reliability, 31% fraud rate
- "Real" data quality is degrading — LLMs vs surveys may be closer than assumed
- Industry shows 80-90% match rates between simulated consumers and surveys
- BUT: actual behavioral outcomes remain unvalidated

## What Evaluation Framework Should Address
1. **Tail insight detection**: Does synthetic research surface same non-obvious, high-value insights as real research?
2. **Variability calibration**: Does synthetic exhibit same distributional spread? (Variance ratios, IQR comparisons)
3. **Sycophancy measurement**: Rate of uncritically positive responses vs critical/negative
4. **Contextual knowledge surface rate**: % of situated, practice-based insights synthetic captures
5. **Demographic equity audit**: Accuracy stratified by demographic group
6. **Grounding/provenance transparency**: Source attribution per claim (PersonaCite model)
7. **Confidence intervals**: Uncertainty quantification, not point estimates
8. **"Zuhlke benchmark"**: Compare synthetic not against perfect standard but against actual survey quality
