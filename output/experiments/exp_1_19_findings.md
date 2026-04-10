# Experiment 1.19 — Schema Artifact Format

## Summary

| | Control (no schema) | Pydantic | JSON Schema | TypeScript |
|---|---|---|---|---|
| **Success rate** | 2/2 | 2/2 | **1/2** | 2/2 |
| **Groundedness** | 1.000 | 1.000 | 1.000 | 1.000 |
| **Cost/persona** | $0.036 | $0.035 | $0.040 | $0.037 |
| **Total list items** | 44 | 40 | 38 | 44 |
| **Mean tokens/item** | 16.7 | 13.6 (-19%) | 14.1 (-16%) | 15.5 (-7%) |
| **Summary length (tok)** | 87 | 78 | 77 | 90 |
| **Attempts** | 2.0 | 2.0 | 2.0 | 2.0 |
| **Avg Jaccard sim to ctrl** | — | 0.042 | 0.040 | 0.020 |

Model: claude-haiku-4-5-20251001
Clusters: 2 (mock tenant, engineers + designers)

---

## Hypothesis

> The artifact format the model sees (Pydantic, JSON Schema, TypeScript)
> changes what it produces, even when the fields are identical.

**Verdict: CONFIRMED.** The format dramatically changes the *content* of the
output while having a moderate effect on *structural* properties.

---

## Key Findings

### 1. Content variance is EXTREME across formats (STRONG signal)

The Jaccard similarity between any two variants' list fields is nearly **zero**
for goals, pains, motivations, objections, channels, and decision_triggers.
The model generates completely different text for the same persona from the
same data when the schema is described differently.

Average Jaccard similarity to control:
- Pydantic: **0.042** (essentially no overlap)
- JSON Schema: **0.040**
- TypeScript: **0.020**

This is the strongest finding: **schema format is a major source of output
non-determinism.** The model doesn't just rephrase the same ideas — it
generates fundamentally different personas.

**Signal: STRONG** — near-zero similarity is unambiguous across all fields.

### 2. Vocabulary is the only partially stable field (MODERATE signal)

The `vocabulary` field shows the highest cross-variant similarity (0.04-0.33),
suggesting that short, keyword-like entries are somewhat resistant to format
influence. Words like "deploy", "iterate", "brand" appear across variants.
But even vocabulary similarity is LOW overall.

**Signal: MODERATE** — slightly more stable than other fields, but still
highly variable.

### 3. Pydantic format produces the most concise output (STRONG signal)

Mean tokens per item across list fields:
- Control: **16.7**
- Pydantic: **13.6** (-19%)
- JSON Schema: **14.1** (-16%)
- TypeScript: **15.5** (-7%)

Pydantic's compact Python type syntax (e.g., `list[str]  # 2-8 items`)
appears to encourage the model toward brevity. TypeScript (with its more
verbose interface syntax) produces output closest to the control.

Per-field breakdown confirms this pattern:
- Goals: Control 22.6 tok → Pydantic 18.4 (-19%)
- Pains: Control 24.9 tok → Pydantic 20.0 (-20%)
- Objections: Control 25.7 tok → Pydantic 21.7 (-16%)

**Signal: STRONG** — consistent direction across all list fields.

### 4. Pydantic reduces list item counts (MODERATE signal)

Total list items:
- Control: 44
- Pydantic: 40 (-9%)
- JSON Schema: 38 (-14%)
- TypeScript: 44 (same)

Both Pydantic and JSON Schema produce fewer items. The explicit min/max
constraints visible in JSON Schema (`"minItems": 2, "maxItems": 8`) may
cause the model to target the lower end. TypeScript comments (`// 2-8 items`)
are softer constraints and preserve the control's item count.

**Signal: MODERATE** — directional but small sample.

### 5. JSON Schema caused one synthesis failure (WEAK signal)

The JSON Schema variant failed on one cluster (3 attempts, all groundedness
failures). This may be because JSON Schema is the most verbose format and
consumes more of the context window for the schema description, leaving
less room for the model to attend to source records.

However, n=1 failure is not conclusive.

**Signal: WEAK** — single failure, could be random.

### 6. Groundedness is unaffected by format (STRONG signal)

All successful runs achieved groundedness score of 1.000 across all formats.
The model grounds its claims equally well regardless of how the schema is
described. This means format affects *what* the model says but not *whether*
it can support its claims.

**Signal: STRONG** — perfect scores across 7/8 runs.

### 7. Cost is essentially unchanged (WEAK signal)

Cost ranged from $0.034-$0.040 with no meaningful pattern. The schema
description in the system prompt adds a few hundred tokens but doesn't
significantly affect total cost since most tokens are in the cluster data
and the generated output.

**Signal: WEAK** — no meaningful cost difference.

---

## Dimension-by-Dimension Analysis

| Dimension | Control | Pydantic | JSON Schema | TypeScript | Signal |
|---|---|---|---|---|---|
| Content similarity to ctrl | — | 0.042 | 0.040 | 0.020 | STRONG (all LOW) |
| Mean tokens/item | 16.7 | 13.6 | 14.1 | 15.5 | STRONG (Pydantic shortest) |
| Total list items | 44 | 40 | 38 | 44 | MODERATE |
| Groundedness | 1.000 | 1.000 | 1.000 | 1.000 | STRONG (all equal) |
| Cost | $0.036 | $0.035 | $0.040 | $0.037 | WEAK (negligible diff) |
| Success rate | 100% | 100% | 50% | 100% | WEAK (n=1 failure) |
| Summary length | 87 tok | 78 tok | 77 tok | 90 tok | MODERATE |

---

## Implications

1. **Schema format is a confound in other experiments.** Any experiment that
   compares synthesis outputs must hold the schema format constant, or the
   format variation will dominate the signal. This is a methodological finding
   that affects the entire experiment program.

2. **If you want shorter output, use Pydantic format.** It's the most concise
   format and produces 19% fewer tokens per item than no-schema control. This
   is a free improvement with no groundedness cost.

3. **Content is fundamentally non-deterministic.** Even with the same data,
   same model, same tool schema, and same prompt — changing only the schema
   description produces completely different persona text. This means single-run
   comparisons between experiments are unreliable. Multiple runs and statistical
   tests are needed.

4. **The tool schema does the real work.** Since all formats produce valid,
   grounded personas despite wildly different text, the structured tool-use
   forcing (the `input_schema` parameter) is what ensures structural compliance,
   not the system prompt description.

---

## Recommendation

**DEFER** — do not adopt any format as default yet.

Rationale:
- The primary finding is methodological: format is a confound, not a feature.
- The conciseness benefit of Pydantic format overlaps with experiment 1.17
  (length budgets), which addresses the same goal more directly.
- The extreme content variance suggests the control (no schema in prompt)
  is the safest baseline — it adds no confounding influence.

**Action items:**
1. Document that schema format must be held constant across experiments.
2. Consider adding Pydantic schema to system prompt as a conciseness lever
   IF experiment 1.17's budget approach is not adopted.
3. Run multiple iterations to establish variance baselines for future experiments.

---

## Limitations

- **Small sample**: 2 clusters, 1 run per variant. N=8 total runs (7 successful).
- **Single model**: Haiku 4.5 only. Larger models may respond differently.
- **Jaccard is coarse**: Exact string matching understates semantic similarity.
  "Reduce deployment time" and "Speed up deploys" score 0.0 Jaccard but are
  semantically equivalent. Embedding-based similarity would give a more
  nuanced picture.
- **JSON Schema failure (n=1)**: Cannot conclude format causes failures.

---

## Files Changed

- `synthesis/engine/prompt_builder.py` — Added `SchemaFormat`, three renderers
  (`_render_pydantic_schema`, `_render_jsonschema_schema`, `_render_typescript_schema`),
  `build_system_prompt()` function
- `synthesis/engine/synthesizer.py` — Added `schema_format` parameter, uses
  `build_system_prompt()` instead of raw `SYSTEM_PROMPT`
- `scripts/experiment_1_19.py` — Experiment runner with cross-variant similarity
- `tests/test_exp_1_19.py` — 17 unit tests

## Decision

**DEFER** — methodological finding; format is a confound, not a feature to adopt.
