# Consolidated Misalignment Taxonomy — 30 Types Across 6 Dimensions

Compiled from all DFS sources. This is the most comprehensive taxonomy of AI persona misalignment we found.

## I. Semantic/Factual (4 types)
1. Trait omission — attributes ignored
2. Trait hallucination — attributes invented or contradicted
3. Semantic contradiction — identity contradicted by expressed beliefs
4. Surface fidelity masking deep failure — correct averages, wrong sentences (r=0.40 for IC)

## II. Cultural/Contextual (5 types)
5. Low-resource cultural blindness — 87% human vs 37-62% LLM (Bangladesh)
6. Religious minority erasure — Buddhist personas worst everywhere
7. Political asymmetry — models favor different parties
8. Historical narrative flattening — can't represent competing histories
9. WEIRD population default — defaults to USA/Canada/Australia/W.Europe

## III. Emotional/Social (5 types)
10. Empathy gap — ~1 Likert point deficit
11. Pollyanna Principle — +0.39 positive bias
12. Sycophancy — approves questionable features
13. Emotional depth loss — missing nuance, surprise
14. Emotional flattening — RLHF avoids jealousy, reduces negatives

## IV. Behavioral (5 types)
15. Oversimplification — missing competing priorities
16. Undifferentiated needs — cares about everything equally
17. No behavioral data — can't actually use products
18. Coherence traps — smooths away meaningful contradictions
19. Persona incoherence — different questions, different person

## V. Demographic Bias (5 types)
20. Socially desirable persona bias — attractive/White performs better
21. Implicit reasoning bias — rejects stereotypes explicitly, applies them implicitly
22. Gender bias — male > female accuracy
23. Disability bias — up to 64% performance drop; 58% refusal rate
24. Training data skew — English, affluent, tech-literate overrepresented

## VI. Structural/Methodological (6 types)
25. Variance compression — lower SD than humans
26. Ordering/labeling artifacts — 43 models → randomness when controlled
27. Temporal instability — different results months apart
28. Prompt brittleness — minor wording → major output changes
29. Scaling paradox — bigger can be worse
30. Task-dependent fidelity — structured maintains, free-form degrades

## Usage for Our Experiment
Each of these 30 types is a testable hypothesis. We should design our evaluation to detect as many as possible, prioritizing the most consequential for our use case.
