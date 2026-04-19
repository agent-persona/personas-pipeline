# SCOPE — Socially-Grounded Persona Framework (Venkit et al., 2026)

**URL:** https://arxiv.org/abs/2601.07110
**Venue:** arXiv, January 2026 (Salesforce Research)

## Research Question
Does demographic-only conditioning produce stereotyped behavior, and can sociopsychological enrichment improve fidelity?

## Framework: 8 Sociopsychological Facets
**4 Conditioning facets (inputs):**
1. Demographics
2. Sociodemographic Behavior
3. Personality Traits
4. Identity Narratives

**4 Evaluation facets (outputs to measure):**
5. Values/Motivations
6. Behavioral Patterns
7. Professional Identity
8. Creativity

## Setup
- 124 U.S. participants completed 141-item protocol
- Three evaluation dimensions: Behavioral Correlation, Exact-Match Accuracy, Demographic Accentuation Bias

## Key Results
- Demographic-only: r=0.624, 35.1% accuracy, **Bias%=101.23** (DOUBLES demographic signal)
- Full SCOPE: r=0.667, 39.7% accuracy
- Non-demographic personas (traits + identity): r=0.658, **Bias%=-56.35** (REDUCES demographic signal)
- **Demographics alone explain only ~1.5% of response variance**
- AI-generated summaries underperform structured human-grounded facets

## Key Takeaway for Our Experiment
The Demographic Accentuation Bias metric is novel and important — it measures whether persona conditioning AMPLIFIES stereotypes. The conditioning/evaluation facet separation is a clean experimental design. Demographics alone are nearly useless for persona accuracy — we need richer grounding.
