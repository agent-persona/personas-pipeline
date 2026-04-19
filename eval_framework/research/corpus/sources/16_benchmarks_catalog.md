# Persona Evaluation Benchmarks Catalog (2024-2026)

## Production-Ready Benchmarks

### PersonaGym (EMNLP Findings 2025)
- **URL:** https://arxiv.org/abs/2407.18416
- **Scale:** 200 personas, 150 environments, 10,000 questions
- **Dimensions:** Expected Action, Linguistic Habits, Persona Consistency, Toxicity Control, Action Justification
- **Key metric:** PersonaScore (1-5 composite, 76.1% Spearman with human)
- **Finding:** Model size ≠ persona capability

### RPEval (arXiv 2025)
- **URL:** https://arxiv.org/abs/2505.13157 | **Code:** https://github.com/yelboudouri/RPEval
- **Scale:** 9,018 scenarios, 3,061 characters
- **Dimensions:** Emotional understanding, decision-making, moral alignment, in-character consistency
- **Finding:** GPT-4o 5.81% on in-character consistency (catastrophic epistemic boundary failure)

### CharacterBench (AAAI 2025)
- **URL:** https://arxiv.org/abs/2412.11912
- **Scale:** 22,859 annotated samples, 3,956 characters, bilingual CN/EN
- **Dimensions:** Memory, Knowledge, Persona, Emotion, Morality, Believability (11 sub-dimensions)
- **Innovation:** Sparse vs dense dimension distinction; CharacterJudge fine-tuned evaluator

### MRBench (arXiv 2026)
- **URL:** https://arxiv.org/abs/2603.19313
- **Scale:** 800 instances from 16 novels, bilingual
- **Dimensions:** Memory-Anchoring, Memory-Selecting, Memory-Bounding, Memory-Enacting
- **Innovation:** 4-stage decomposition identifies WHERE failures occur

### PERSIST (arXiv 2025)
- **URL:** https://arxiv.org/abs/2508.04826
- **Scale:** 2M+ measurements, 25 models, 250 permutations
- **Focus:** Stability/reliability of persona measurements
- **Finding:** 20% measurement shift from question reordering; CoT increases instability

### InCharacter (ACL 2024)
- **URL:** https://arxiv.org/abs/2310.17976 | **Code:** https://github.com/Neph0s/InCharacter
- **Scale:** 32 characters, 14 psychological scales
- **Innovation:** Interview-based psychological assessment of persona fidelity
- **Finding:** 80.7% dimension accuracy with Expert Rating

### SCOPE (arXiv 2026)
- **URL:** https://arxiv.org/abs/2601.07110
- **Scale:** 124 human participants, 141-item protocol
- **Innovation:** 8 sociopsychological facets; Demographic Accentuation Bias metric
- **Finding:** Demographics explain only 1.5% of variance; demographic-only doubles demographic signal

## Which Benchmarks to Use for Our Experiment
1. **PersonaGym** — ready-made evaluation framework, adoptable metrics
2. **InCharacter** — validated psychometric approach, best for personality-based personas
3. **PERSIST methodology** — essential for measuring our evaluation's own reliability
4. **SCOPE framework** — for understanding conditioning vs evaluation facet separation
5. **MREval decomposition** — for diagnostic granularity on failure stages
