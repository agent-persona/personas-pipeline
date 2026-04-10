# Experiment 6.17 — Persona Naming Distinctiveness: Findings

## Summary

**Clustering coefficient**: `0.75` (STRONG signal)

**Hypothesis**: Generic naming indicates low lexical creativity and synthetic-origin bias.

---

## Baseline Persona Scores

| File | Name | Archetype Score | Specificity Score | Flags |
|------|------|----------------|-------------------|-------|
| persona_00.json | Alex the API-First DevOps Engineer | 0.75 | 0.17 | ROLE_DESCRIPTOR_TEMPLATE, NEUTRAL_FIRST_NAME, ROLE_ECHO(api,devops,engineer) |
| persona_01.json | Maya the Freelance Brand Designer | 0.75 | 0.17 | ROLE_DESCRIPTOR_TEMPLATE, NEUTRAL_FIRST_NAME, ROLE_ECHO(brand,designer,freelance) |

### Baseline Notes

- **Alex the API-First DevOps Engineer**: matches "<Name> the <Role>" template (descriptor: "API-First DevOps Engineer"); "alex" is in the neutral-name corpus; role-echo words: ['api', 'devops', 'engineer']
- **Maya the Freelance Brand Designer**: matches "<Name> the <Role>" template (descriptor: "Freelance Brand Designer"); "maya" is in the neutral-name corpus; role-echo words: ['brand', 'designer', 'freelance']

---

## Synthetic Name Candidates (alternative naming approaches)

These names were synthesized using strategies designed to reduce archetype collapse.

| Name | Strategy | Cluster | Archetype Score | Specificity Score | Flags |
|------|----------|---------|----------------|-------------------|-------|
| Priya Raghunathan | South Asian name, no role descriptor, no template | DevOps / API engineer | 0.0 | 0.98 | — |
| Kofi Mensah-Boateng | West African compound surname, culturally specific | DevOps / API engineer | 0.0 | 0.98 | — |
| Ren | Single nickname, gender-ambiguous, East Asian origin, zero role signal | Freelance designer | 0.0 | 1.0 | — |
| Valentina Cruz | Latina name, no role echo, high cultural specificity | Freelance designer | 0.0 | 0.98 | — |
| Dmitri Volkov | Eastern European, strongly non-neutral demographic signal | DevOps / API engineer | 0.0 | 0.98 | — |

**Synthetic candidate clustering coefficient**: `0.0` — compare with baseline `0.75` (lower is better / more distinctive)

---

## Signal Interpretation

**Signal level**: STRONG

The majority of baseline persona names collapse into recognisable AI-generated archetypes.
The `<First> the <Role-Descriptor>` template is the dominant driver.
**Recommendation**: ADOPT — implement a naming diversity constraint in the persona generator.
Require names drawn from a culturally diverse name corpus and prohibit the `the <Role>` suffix pattern.

---

## Methodology

Each name is scored on four binary/continuous components (averaged to produce `archetype_score`):

1. **ROLE_DESCRIPTOR_TEMPLATE** (0 or 1) — matches `<Name> the <Descriptor>` pattern
2. **NEUTRAL_FIRST_NAME** (0 or 1) — first name appears in a 48-entry LLM-safe name corpus
3. **ROLE_ECHO** (0–1) — fraction of job-title words echoed in the name (2+ = 1.0)
4. **ALLITERATIVE_ARCHETYPE** (0 or 1) — alliterative two-word pattern (e.g. Marketing Mary)

`clustering_coefficient` = mean archetype_score across baseline personas.

Thresholds: STRONG > 0.6 | MODERATE 0.4–0.6 | WEAK 0.2–0.4 | NOISE < 0.2

`specificity_score` = max(0, 1 − archetype_score − 0.1 × length_penalty)
where length_penalty = min(1, (word_count − 1) / 5). Penalises long label-like names.
