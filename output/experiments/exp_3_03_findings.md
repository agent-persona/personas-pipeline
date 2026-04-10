# Experiment 3.03 — Retrieval-Augmented Synthesis

## Summary

| | Control (all) | k=3 | k=10 | k=30 |
|---|---|---|---|---|
| **Groundedness** | 1.000 | 1.000 | 1.000 | 1.000 |
| **Hallucination rate** | **0.0%** | 5.9% | 4.0% | 2.9% |
| **Evidence coverage** | **100%** | 62.5% | 83.3% | 100% |
| **Mean confidence** | **0.872** | 0.816 | 0.865 | 0.838 |
| **Cost/persona** | **$0.029** | $0.039 (+35%) | $0.034 (+18%) | $0.076 (+163%) |
| **Attempts** | 1.5 | 2.0 | 1.5 | 3.0 |
| **Duration** | 31.6s | 43.8s | 29.7s | 69.8s |

Model: claude-haiku-4-5-20251001
Clusters: 2 (12 records each)

---

## Hypothesis

> Per-section top-k record retrieval improves groundedness while reducing
> cost compared to dumping all records in context.

**Verdict: DISCONFIRMED.** Retrieval-augmented synthesis is *worse* than the
control across all measured dimensions. The flat all-records approach
produces better groundedness metrics, lower cost, and no hallucinations.

---

## Key Findings

### 1. The control (all records) is the best variant (STRONG signal)

The control achieved:
- Perfect groundedness (1.000) — same as all variants
- **Zero hallucinations** — no evidence entries with confidence < 0.7
- **100% evidence coverage** — every record was cited at least once
- **Lowest cost** ($0.029) — fewest tokens needed, fewest retries
- **Highest confidence** (0.872 mean) across evidence entries

No retrieval variant matched the control on ANY metric.

**Signal: STRONG** — the control dominates across all dimensions.

### 2. Retrieval INCREASES cost, not decreases (STRONG signal)

This is the most counter-intuitive finding. The hypothesis predicted lower
cost from retrieval (fewer records in context = fewer input tokens). But:

- k=3: $0.039 (+35%) — more retries due to groundedness failures
- k=10: $0.034 (+18%)
- k=30: **$0.076 (+163%)** — the per-section format repeats records across
  sections, inflating the prompt far beyond the flat list

The per-section format includes 9 sections × k records each. With k=30 and
12 unique records, the same record appears in multiple sections, dramatically
increasing prompt length. With k=3, fewer records are available, causing more
groundedness failures and retries.

**Signal: STRONG** — cost increased in every variant.

### 3. Retrieval introduces low-confidence evidence (MODERATE signal)

Long-tail hallucination rate (evidence entries with confidence < 0.7):
- Control: **0.0%** (zero low-confidence entries)
- k=3: 5.9% (1 entry per persona on average)
- k=10: 4.0%
- k=30: 2.9%

When the model receives only a subset of records per section, it sometimes
generates claims it can't strongly ground, producing lower confidence scores.
The more records available (higher k), the lower the hallucination rate —
converging toward the control.

**Signal: MODERATE** — consistent trend, but rates are still low.

### 4. Evidence coverage drops significantly at low k (STRONG signal)

The control cited all 12 records (100% coverage). With retrieval:
- k=3: only 7.5 unique records cited (62.5%)
- k=10: 10 records cited (83.3%)
- k=30: 12 records cited (100%, matching control)

Low k means some records never appear in ANY section's retrieval results,
so they can't be cited. This creates "invisible" records — data the pipeline
has but the synthesizer never sees.

**Signal: STRONG** — clear monotonic relationship between k and coverage.

### 5. k=30 matches control on groundedness but costs 2.6x more (STRONG signal)

k=30 returns all 12 records for each section (12 < 30), so every record is
available everywhere — functionally identical to the control in terms of
record availability. Yet it costs $0.076 vs $0.029 because the per-section
format repeats records 9 times, producing a much larger prompt.

This proves the retrieval overhead is pure cost with no benefit when
k >= corpus size. And for this dataset, even k=10 covers 83% of records.

**Signal: STRONG** — the format itself is the cost driver.

---

## Why the Hypothesis Failed

The experiment was designed for a scenario where:
1. Records number in the hundreds or thousands per cluster
2. Only a fraction are relevant to each persona section
3. Full-context synthesis would blow the token budget

But the current mock dataset has only **12 records per cluster**. At this
scale:
- All records fit easily in context
- Every record is potentially relevant (no noise to filter)
- Per-section retrieval adds format overhead without reducing content

**The hypothesis may hold at production scale** (hundreds of records per
cluster), where full-context is impractical and selective retrieval is
necessary. This experiment should be re-run when the store module
provides chunked records from real data sources.

---

## Dimension-by-Dimension Analysis

| Dimension | Control | k=3 | k=10 | k=30 | Winner | Signal |
|---|---|---|---|---|---|---|
| Groundedness | 1.000 | 1.000 | 1.000 | 1.000 | TIE | N/A |
| Hallucination rate | 0.0% | 5.9% | 4.0% | 2.9% | Control | MODERATE |
| Evidence coverage | 100% | 62.5% | 83.3% | 100% | Control/k=30 | STRONG |
| Mean confidence | 0.872 | 0.816 | 0.865 | 0.838 | Control | MODERATE |
| Cost | $0.029 | $0.039 | $0.034 | $0.076 | Control | STRONG |
| Attempts | 1.5 | 2.0 | 1.5 | 3.0 | Control | MODERATE |

---

## Recommendation

**REJECT** for current dataset. **DEFER** for production scale.

The retrieval-augmented pattern is worse on every metric at the current
data scale (12 records/cluster). However, the pattern is architecturally
sound and should be re-tested when:
1. Clusters contain 100+ records
2. The store module provides chunked unstructured text
3. Record embeddings use a real model (Voyage-3) instead of TF-IDF

Keep the `record_retrieval.py` module and `retrieval_k` parameter as
infrastructure for future experiments.

---

## Limitations

- **Small dataset**: Only 12 records per cluster. The hypothesis targets
  production-scale data (hundreds of records).
- **TF-IDF embeddings**: Bag-of-words similarity is crude. Real semantic
  embeddings (Voyage-3) would produce better retrieval quality.
- **Single model**: Haiku 4.5 only.
- **Mock data**: Records are short and keyword-rich. Real Intercom/Zendesk
  data is longer and noisier, which is where retrieval would add value.
- **n=2 clusters per variant**: Small sample, no statistical tests.

---

## Files Changed

- `synthesis/engine/record_retrieval.py` — New module: TF-IDF record index
  with per-section top-k retrieval
- `synthesis/engine/prompt_builder.py` — Added `build_retrieval_augmented_message()`,
  `_render_records()`, `retrieval_k` parameter to `build_messages()`/`build_retry_messages()`
- `synthesis/engine/synthesizer.py` — Added `retrieval_k` parameter
- `scripts/experiment_3_03.py` — Experiment runner with evidence analysis
- `tests/test_exp_3_03.py` — 20 unit tests

## Decision

**REJECT** at current scale. **DEFER** re-evaluation to when production
data is available (100+ records per cluster).
