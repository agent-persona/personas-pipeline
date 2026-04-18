# Benchmark plans

Three head-to-head benchmarks against the strongest OSS comparators
identified in `BENCHMARK_CANDIDATES.md`. Each follows our standard
experiment shape (hypothesis → control → metric → decision) so it slots
directly into `output/experiments/`.

---

## Benchmark 1 — Full pipeline vs. `joongishin/persona-generation-workflow`

**Experiment slot:** `exp-7.01-oss-bench-fullpipe`

**Hypothesis.** personas-pipeline produces personas with materially
higher groundedness and comparable-or-better judge scores than
LLM-summarizing++ (the strongest workflow in Shin et al., DIS'24), on
the same input records, at comparable model cost.

**Comparator.** [joongishin/persona-generation-workflow](https://github.com/joongishin/persona-generation-workflow),
`LLM-summarizing++` variant (pre-cluster by human, LLM summarizes with
designer-specified qualities).

**Control.** Our `tenant_acme_corp` golden set (37 records, 2 clusters).
Adapt it to their CSV schema; keep record content byte-identical.

**Procedure.**
1. Run both systems on the adapted input. Use the same model (Haiku,
   `temperature=0.0` per exp-2.06).
2. Normalize outputs to `synthesis.models.PersonaV1` shape (their
   output lacks `source_evidence` by design — record as "0.0
   groundedness" for honesty).
3. Score both with:
   - `evaluation/metrics.py` → schema validity, groundedness,
     distinctiveness, judge score, cost per persona.
   - DeepEval cross-check: Role Adherence, Hallucination, Faithfulness.

**Metrics & decision.**
- Primary: mean groundedness, mean judge score, cost/persona.
- Ship the comparison if our groundedness ≥ 0.9 while our judge score
  is within 0.5 pts of theirs. If we trail on judge score by >0.5 pts,
  log as negative result and investigate.

**Expected effort.** 1–2 days. Main cost: CSV adapter for their input
format.

---

## Benchmark 2 — Twin runtime vs. `microsoft/TinyTroupe`

**Experiment slot:** `exp-7.02-oss-bench-twin`

**Hypothesis.** Our twin runtime enforces persona boundary, refusal,
and meta-question behavior more consistently than a TinyTroupe
`TinyPerson` loaded with the same persona spec.

**Comparator.** [microsoft/TinyTroupe](https://github.com/microsoft/TinyTroupe),
`TinyPerson` with persona fields mapped from our `PersonaV1`.

**Control.** Three production personas from `output/persona_*.json`.
The boundary/refusal/meta-question prompt suites already used in
exp-4.05, exp-4.16, exp-4.20.

**Procedure.**
1. Load each PersonaV1 into our twin and into a `TinyPerson` (mapping
   `goals`, `pains`, `sample_quotes`, `vocabulary` → TinyTroupe fields).
2. Run each suite, 10 prompts × 3 runs per persona × 2 systems.
3. Blind-judge the 360 transcripts with our judge rubric
   (`evaluation/judges.py`).

**Metrics & decision.**
- Primary: boundary-respect rate, refusal-correctness rate,
  meta-question handling rate, consistency (intra-run variance).
- Ship if our rates are ≥ +10 pp on boundary and refusal. Treat any
  loss as an honest negative; file follow-up experiment.

**Expected effort.** 2–3 days. TinyTroupe field mapping is the main
risk (their spec is denser than ours).

---

## Benchmark 3 — Synthesis coverage vs. `tencent-ailab/persona-hub`

**Experiment slot:** `exp-7.03-oss-bench-coverage`

**Hypothesis.** For a specific tenant's records, our grounded synthesis
produces a persona set with higher coverage of that tenant's
population than a demographically-matched sample drawn from PersonaHub
(1B personas, ungrounded).

**Comparator.** [tencent-ailab/persona-hub](https://github.com/tencent-ailab/persona-hub).
Sample N personas from their public dump matching the tenant's stated
demographic (e.g., Discord community owners for `tenant_acme_corp`).

**Control.** `tenant_acme_corp` records. Our current 2-persona output
vs. N=2 and N=5 PersonaHub samples.

**Procedure.**
1. Reuse the exp-6.02 coverage framework: for each source record,
   measure nearest-persona similarity. Aggregate to coverage%.
2. Also score distinctiveness (inter-persona divergence) on both sets.

**Metrics & decision.**
- Primary: coverage%, distinctiveness, unique-claim count per persona.
- Expected result: we win on coverage (grounded), PersonaHub wins on
  raw diversity. Frames grounding as the load-bearing differentiator.

**Expected effort.** 1 day (PersonaHub is a static dataset — no runtime
integration needed).

---

## Benchmark 4 — Segmentation 1:1 vs. three OSS clusterers

**Experiment slot:** `exp-7.04-oss-bench-segment`

**Why this benchmark.** Benchmarks 1–3 compare full pipelines or end
stages. This one isolates the segmentation module so we can tell
whether our clustering choices (greedy Jaccard on behavioral sets with
`similarity_threshold=0.15`) are pulling their weight vs. the obvious
modern alternatives. Two of the comparators are topic modelers rather
than persona systems, on purpose — topic modeling is what a senior
engineer would actually reach for if they were told "cluster these
chat logs into segments."

**Hypothesis.** For behavioral records (Discord messages, LinkedIn
comments, analytics events), our behavior-set clusterer produces
downstream personas with higher coverage and comparable
distinctiveness than clusters derived from topic-modeling baselines.
We expect to lose ground when input records are text-rich and
topic-heavy (e.g., long-form subscriber surveys) — that case is worth
seeing.

**Comparators.** Three. Picked to span the realistic solution space.

| # | Repo | Method | Why this one |
|---|---|---|---|
| 1 | [joongishin/persona-generation-workflow](https://github.com/joongishin/persona-generation-workflow) | sklearn k-means on LLM-selected user fields | Only persona-native OSS repo with a segmentation step (DIS'24) |
| 2 | [MaartenGr/BERTopic](https://github.com/MaartenGr/BERTopic) | sentence-transformers → UMAP → HDBSCAN → c-TF-IDF | ~7.5k★, de-facto modern text-topic baseline for review / chat clustering |
| 3 | [ddangelov/Top2Vec](https://github.com/ddangelov/Top2Vec) | joint document + topic + word embedding clustering | Alternative modern topic modeler; different inductive bias than BERTopic |

**Control.** Two inputs on purpose — behavioral vs. text-rich —
because we expect them to rank the methods differently.
- `tenant_acme_corp` golden set (37 behavioral records).
- A text-rich control: ≥300 Discord messages or survey responses from
  a real crawler run, to stress topic-modeling methods fairly.

**Procedure.**
1. Run each of the four segmenters on each input. Fix seeds where
   possible; record all knobs.
2. Normalize every segmenter's output to the same cluster-labels-per-
   record form.
3. **Intrinsic metrics** (segmentation quality, for reference):
   - Coverage % — fraction of records assigned a non-noise cluster
     (matches the exp-6.02 coverage definition).
   - Cluster count and size distribution.
   - Stability — run each segmenter 5× with permuted record order;
     measure label-agreement (Adjusted Rand Index against the first
     run). This is the exp-6.02 "stable persona ID" check, extended
     to comparators.
4. **Downstream metrics** (the one that matters): feed each
   segmenter's clusters into the **same** `synthesis/` stage (identical
   prompt, model, seed) and score the resulting personas with our
   evaluation harness — groundedness, distinctiveness, judge score,
   cost per persona.

**Metrics & decision.**
- Primary: downstream persona groundedness and judge score on the
  behavioral input. Secondary: stability ARI.
- Ship if we lead downstream judge score by ≥ 0.3 pts on behavioral
  input AND our stability ARI ≥ 0.8. If a topic modeler wins on the
  text-rich input, that's a real finding — publish it and consider
  shipping an `engine/clusterer.py` swap-in for that regime (the
  module's call signature is already designed for drop-in
  replacement, per `segmentation/README.md`).

**Expected effort.** 3 days. Bulk of the cost is (a) the adapter for
persona-generation-workflow's expected CSV schema and (b) curating
the text-rich control input.

---

## Sequencing

Run **Benchmark 1 first** — it's the headline claim and cheapest. If
it lands, **Benchmarks 2, 3, 4** can run in parallel and feed a single
`exp-7.xx-oss-bench/SUMMARY.md` writeup. Benchmark 4's downstream
synthesis step reuses the same model backend and prompt as Benchmark 1,
so the two share setup cost.

All benchmark outputs go under `output/experiments/exp-7.xx/`
following the standard FINDINGS.md template.
