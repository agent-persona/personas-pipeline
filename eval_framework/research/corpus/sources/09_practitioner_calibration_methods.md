# Practitioner Calibration Methods — Compiled from 6 Sources

## Sources
- M1 Project (Synthetic Users + SaaS Buyer Personas)
- Sybill (ICP Guide 2026)
- GrowthAhoy (Build ICP with AI)
- Vaultmark (AI ICP Persona Lab)
- AriseGTM (Behavioral Validation Guide)
- Signoi/Personalysis (Empirical Validation Study)

## Key Empirical Finding: Signoi/Personalysis
The most rigorous empirical validation found in practitioner space:
- 12 UK population clusters, 60 AI personas (5 per segment)
- Compared synthetic ratings against 50 real concept testing studies (avg 400 respondents)
- **Result: r=0.88 (R²=0.77)** with persona-based training
- Without personas (zero-shot): r~0.45 (R²~0.20)
- This is the strongest evidence that grounding matters

## AriseGTM Validation Score
**Validation Score = (Behavioral Evidence Strength × Consistency) / Recency Decay**
- Each component: 0.0-1.0
- > 0.7: validated
- 0.5-0.7: moderately validated
- < 0.5: weak, do not use for major decisions

## Critical Data Points
- 40-60% divergence between stated needs and actual behavior (AriseGTM)
- Buyers state 8.3 "must-have" features, use only 3.1 in 90 days (Gartner 2024)
- Stated evaluation timeline: 4.2 months vs actual 7.8 months (Forrester 2025)
- ~80% of average persona attributes are unvalidated hypotheses

## Key Takeaway for Our Experiment
The Signoi result is the strongest pro-persona evidence but is narrow (UK consumer concept testing). AriseGTM's framework is the most rigorous practitioner approach — we should adapt their Validation Score for our evaluation. The stated-vs-actual gap (40-60%) is the most damning number for interview-only personas.
