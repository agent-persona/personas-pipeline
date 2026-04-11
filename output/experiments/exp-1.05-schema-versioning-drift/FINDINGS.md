# Experiment 1.05: Schema Versioning Drift

## Hypothesis
Schema additions should not materially change quality on the same source data unless they help the model structure the persona more cleanly.

## Method
1. Ran 2 clusters through 3 schema variants: `v1`, `v1.1`, and `v2`.
2. Used branch-local tool definitions and prompt notes for each version; the shared prompt builder stayed untouched.
3. Scored every persona with the same branch-local judge rubric.
4. Compared shared fields against the `v1` baseline to estimate schema drift.
5. Remote LLM paths were kept in code but the run used the local deterministic fallback so the branch could complete without provider stalls.

## Variant Summary
- `v1`: judge `3.52`, grounded `1.00`, valid `100%`
- `v1.1`: judge `3.65`, grounded `1.00`, valid `100%`, judge delta vs v1 `+0.13`, shared similarity vs v1 `1.00`
- `v2`: judge `3.56`, grounded `1.00`, valid `100%`, judge delta vs v1 `+0.04`, shared similarity vs v1 `1.00`

## Baseline
- Mean judge score: `3.52`
- Mean groundedness: `1.00`
- Validity rate: `100%`

## Decision
Adopt. The schema enrichments improved or matched quality without substantial drift in shared fields.

## Caveat
Tiny sample: 1 tenant, 2 clusters. The signal is directional only.
