# exp-2.05 — Few-Shot Exemplars

## Hypothesis

Injecting N hand-curated example personas into the synthesis system prompt
has **two competing effects**:

- **Schema/structural lift**: exemplars act as an implicit schema anchor,
  potentially improving field completeness and vocabulary distinctiveness.
- **Cloning pressure**: as N grows, the model may copy exemplar content into
  its output rather than grounding to the tenant's actual records.

Primary question: does `groundedness_rate` decline as N increases? (The
`EXAMPLE_rec_*` tracer plus a tracer-token vocabulary are the instruments.)

Sweep: `N ∈ {0, 1, 3, 5}`. Exemplars are drawn from five distant verticals
(clinical research, tax accounting, K-12 teaching, independent restaurant,
urban planning) against a B2B SaaS tenant — cloning would be obvious.

## Control

`N = 0` is byte-identical to the pre-experiment `SYSTEM_PROMPT` (unit-test
guaranteed in `prompt_builder.build_system_prompt`). The N=0 slice runs in
the same pipeline session as N=1/3/5, against the same clusters, at the same
default temperature. This holds segmentation stochasticity fixed across
conditions — a tighter control than a separate-run baseline.

Tenant: mock B2B SaaS. Two runs: initial 2-cluster run (designer + engineer),
then expanded 8-cluster run (+ product managers, sales reps, marketing
managers, customer success, executives, project coordinators).
Model: `claude-haiku-4-5-20251001`.

## Metric

- **Primary**: `groundedness_rate` — mean of
  `synthesis.engine.groundedness.GroundednessReport.score` across the run.
- **Guardrail**: `schema_validity` (expected pegged at 1.00 due to retry loop).
- **Cloning bundle** (local proxies, not shared metrics):
  - Jaccard overlap of vocabulary / goal-tokens vs injected exemplars.
  - Tracer-token count: 19 hand-picked distinctive phrases like `IRB`,
    `EBITDA`, `lesson plan`, `mise en place`, `zoning variance`.
  - Quote LCS ratio (difflib) vs all exemplar quotes.
  - Count of `EXAMPLE_rec_*` record-IDs in synthesized personas (unconfounded
    because `source_evidence` is omitted from the prose-rendered prompt).
- **Cost**: `cost_per_persona` by N.

## Result

### Run 1: 2 clusters (initial)

2 clusters × 4 N-values = 8 syntheses, total synthesis cost **$0.2717**.

| N | schema_validity | groundedness_rate | cost_per_persona $ |
|---|---|---|---|
| 0 | 1.00 | 1.00 | 0.0492 |
| 1 | 1.00 | 1.00 | 0.0286 |
| 3 | 1.00 | 1.00 | 0.0283 |
| 5 | 1.00 | 0.97 | 0.0297 |

With only 2 data points per N, results were inconclusive. A cost inversion
(N=0 most expensive due to retries) was observed but could not be
distinguished from noise. Decision was to re-run on a larger tenant.

### Run 2: 8 clusters (expanded mock tenant)

Mock tenant expanded from 2 user types (engineers + designers) to 8 (added
product managers, sales reps, marketing managers, customer success,
executives, project coordinators). 128 records → 8 clusters. 8 clusters × 4
N-values = 32 syntheses, total synthesis cost **$1.10**.

| N | schema_validity | groundedness_rate | cost_per_persona $ |
|---|---|---|---|
| 0 | 1.00 | 1.00 | 0.0364 |
| 1 | 1.00 | 0.97 | 0.0277 |
| 3 | 1.00 | 0.99 | 0.0383 |
| 5 | 1.00 | 0.99 | 0.0356 |

Full per-cluster tables in `structural_comparison.log`.

### Cloning-bundle summary (run 2, 32 cells)

- **Tracer tokens**: 0 hits at N>0 across all 32 cells. Two "Title I" hits
  appeared at N=0 (random coincidence from SaaS cluster data, not exemplar
  cloning — confirms the baseline floor).
- **`EXAMPLE_rec_*` leakage**: 0 at every N in every cluster. The tracer was
  never emitted.
- **Vocabulary Jaccard**: 0.00 at every N vs every injected exemplar.
- **Goal-token Jaccard**: ≤ 0.08 at N>0, indistinguishable from the
  random-token coincidence floor.
- **Quote LCS ratio**: 0.05–0.15 across all N, flat. No paraphrase-copy
  signal.

### Cost inversion did not replicate

The run-1 cost inversion (N=0 most expensive) did not hold at 8 clusters.
N=0 ($0.036/persona) and N=5 ($0.036/persona) are essentially equal. The
run-1 finding was noise from retry-count variance at n=2.

### Groundedness is flat, not declining

Groundedness at N=0 is 1.00, dipping to 0.97 at N=1 (4 of 8 clusters scored
0.94 instead of 1.00) and recovering to 0.99 at N=3 and N=5. The dips are
single-violation misses scattered across clusters with no monotone trend.
There is no evidence that increasing N degrades groundedness.

## Decision

**Reject — exemplars have no measurable effect on this pipeline.**

With 8 clusters (32 synthesis calls), the results are clear:

- **No cloning detected** by any proxy at any N. The tracer battery works
  (unit-tested for prompt-level invisibility; 0 at N=0 confirms the
  coincidence floor), and the exemplar verticals are distant enough that
  cloning would be unambiguous. The model simply doesn't copy.
- **No groundedness improvement or degradation**. Groundedness is flat across
  N, with minor stochastic dips that don't correlate with the treatment.
- **No cost benefit**. The run-1 cost inversion was noise; at n=8 costs are
  equal across N.
- **Schema validity pegged at 1.00** at every N, as predicted (the retry loop
  already enforces this).

The tool-use forcing + retry loop already provides the structural anchor that
exemplars would theoretically offer. Adding exemplars to the system prompt
adds token cost without measurable benefit.

**What is adoptable**: the `build_system_prompt(few_shot_count)` API and the
`EXAMPLE_rec_*` tracer pattern are both byte-parity-safe at N=0 and give
future experiments a free cloning-detection instrument. These can stay in the
codebase at zero cost to the default path.

## Notes and caveats

- **N=0 byte-parity** is unit-asserted and is the load-bearing control. Any
  future edit to `SYSTEM_PROMPT` or `build_system_prompt` must preserve
  `build_system_prompt(0) == SYSTEM_PROMPT`.
- **`source_evidence` is deliberately omitted from the prose rendering**.
  This keeps the `EXAMPLE_rec_*` tracer unconfounded: any appearance in
  output is unambiguous cloning. Unit-tested (check 4).
- **Twin chat runs only against the N=0 persona** to bound the sweep cost to
  the synthesis stage. Downstream twin-demo contract preserved.
- **Segmentation is not cross-run deterministic** on cluster IDs (finding
  from exp-4.10). Within a single run, sorting by `cluster_id` in
  `stage_synthesize` gives stable ordering. Cross-run comparisons require
  re-identifying clusters by content.
- **Cache hygiene**: `_GOLDEN_EXAMPLES_CACHE` is in-process, so a live
  fixture edit will not invalidate a long-running process. Not a concern for
  normal `python scripts/run_full_pipeline.py` invocations.

## Files

- `synthesis/synthesis/fixtures/golden_examples.json` — the 5-persona
  fixture. Minimum-viable array lengths. All fake record_ids prefixed
  `EXAMPLE_rec_`.
- `synthesis/synthesis/engine/prompt_builder.py` — `build_system_prompt()`,
  `_load_golden_examples()`, `_render_exemplar_prose()`.
- `synthesis/synthesis/engine/synthesizer.py` — `few_shot_count` kwarg
  threaded through `synthesize()`.
- `scripts/run_full_pipeline.py` — `EXEMPLAR_SWEEP = [0, 1, 3, 5]`,
  sweep-in-stage pattern.
- `output/experiments/exp-2.05-few-shot-exemplars/analyze.py` — read-only
  post-hoc analyzer.
- `output/experiments/exp-2.05-few-shot-exemplars/structural_comparison.log`
  — per-cluster metric table from this run.
