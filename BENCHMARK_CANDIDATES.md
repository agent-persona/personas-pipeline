# Benchmark candidates for personas-pipeline

Goal: identify open-source repos we can benchmark our work against. Our
pipeline covers five concerns: **crawl → segment → synthesize (grounded)
→ twin runtime → evaluate**. No single OSS project does all five, so
benchmarking has to be split by component.

## TL;DR — recommended benchmark set

| Role in benchmark | Repo | Why |
|---|---|---|
| **Full-pipeline apples-to-apples** | [joongishin/persona-generation-workflow](https://github.com/joongishin/persona-generation-workflow) | Closest shape to ours: survey data → k-means clusters → LLM persona summarization → roleplay runtime. DIS'24 paper. Missing: formal eval, evidence-binding. |
| **Twin-runtime strong baseline** | [microsoft/TinyTroupe](https://github.com/microsoft/TinyTroupe) | Microsoft-backed, active, ships statistical validation (t-test, KS-test) against real survey data. No grounding, but the best-resourced twin-sim project. |
| **Synthesis diversity / scale baseline** | [tencent-ailab/persona-hub](https://github.com/tencent-ailab/persona-hub) | De-facto citation for synthetic-persona-at-scale (1B personas, NeurIPS). Useful to benchmark distinctiveness/diversity, even though it is ungrounded. |
| **Evaluation-harness reference** | [confident-ai/deepeval](https://github.com/confident-ai/deepeval) | Not a persona system, but ships Role Adherence, Hallucination, Faithfulness, G-Eval. Good yardstick for our `evaluation/` package. |
| **External benchmark harness** | [bowen-upenn/PersonaMem](https://github.com/bowen-upenn/PersonaMem) (COLM 2025) | Persona-grounded personalization benchmark. Plug our twin in as a scored system. |

## Primary recommendation

If we can only pick **one** primary comparator for a full-pipeline
benchmark, it's **joongishin/persona-generation-workflow**:

- Same shape: structured user records → cluster → LLM summarize → chat.
- Backed by a DIS'24 empirical study, so claims are measurable.
- Gives us four variants (LLM-auto, LLM-grouping, LLM-summarizing,
  LLM-summarizing++) — a natural ablation ladder.
- Our differentiators vs. it are crisp and measurable: (a) evidence
  binding / `source_evidence` record IDs, (b) the `evaluation/` harness
  with groundedness/distinctiveness/drift, (c) the logged
  hypothesis-driven experiments under `output/experiments/`.

Pair it with **TinyTroupe** for the twin-runtime axis (role adherence,
drift, refusal — TinyTroupe has validation infra but no grounding).

## What was ruled out

- **vincentkoc/synthetic-user-research** — single-notebook demo, no
  pipeline, no eval. Too thin to benchmark against.
- **orange0629/llm-personas** — persona-measurement dataset, not a
  generation pipeline.
- **marc-shade/ai-persona-lab** — Ollama group-chat playground, no
  grounding or eval.
- **google-research-datasets/Synthetic-Persona-Chat** — dataset only.
- **DeepPersona** — paper + landing page, no released pipeline we can
  run against our golden set.

## Proposed benchmark protocol (sketch)

1. Run `joongishin/persona-generation-workflow` (LLM-summarizing++ variant)
   against our `tenant_acme_corp` golden input, adapted to its CSV format.
2. Score both outputs with our `evaluation/` metrics (schema validity,
   groundedness, distinctiveness, judge score) and with DeepEval's Role
   Adherence + Hallucination for cross-validation.
3. For the twin axis: feed same persona JSON into our twin and into a
   TinyTroupe `TinyPerson`; run the boundary/refusal/meta-question
   suites (exp-4.05, 4.16, 4.20) against both.
4. Publish the comparison as `output/experiments/exp-7.xx-oss-bench/`.

This gives a head-to-head on our load-bearing claims (grounding, eval
harness, cost) without pretending the other repos were trying to solve
the same problem.
