# exp-7.xx — OSS benchmark suite: SUMMARY

Head-to-head benchmarks of `personas-pipeline` against the five
strongest open-source comparators identified in `BENCHMARK_CANDIDATES.md`.
All runs on the `tenant_acme_corp` golden set (38 records / 8 users / 2
clusters). Matched model and temperature across every head-to-head
(`claude-haiku-4-5-20251001`, `temperature=0.0`). Full per-experiment
results under `output/experiments/exp-7.0{1..4}*/`.

## One-line per experiment

| Exp | Compared | One-line result |
|---|---|---|
| [7.01](../exp-7.01-oss-bench-fullpipe/FINDINGS.md) | **ours** vs `persona-generation-workflow` (DIS'24, LLM-summarizing++) | narrative judge ties (4.5/5 both); schema fidelity heavily favors ours (21.5 vs 0 `source_evidence` rows); pgw-port is ~20× cheaper for equal narrative quality |
| [7.02](../exp-7.02-oss-bench-twin/FINDINGS.md) | **ours** vs TinyTroupe-style port | *(pending: results will populate after exp-7.02 run completes)* |
| [7.03](../exp-7.03-oss-bench-coverage/FINDINGS.md) | **ours** personas vs 100 sampled from `persona-hub` | paired per-record win-rate for ours: 58%; mean-max-sim gap +0.012 (small but real; 34.2% coverage@0.20 vs 31.6% for persona-hub with 50× more personas) |
| [7.04](../exp-7.04-oss-bench-segment/intrinsic_FINDINGS.md) intrinsic | **ours-jaccard** vs BERTopic vs Top2Vec vs kmeans-emb | on 8-user corpus, ours/bertopic/kmeans-emb all 100% coverage / 2 clusters / cross-method ARI 1.0 (shuffled); BERTopic order-sensitive (ARI 0.774 vs ours on unshuffled order, see downstream); Top2Vec fails on tiny corpus |
| [7.04](../exp-7.04-oss-bench-segment/downstream_FINDINGS.md) downstream | same clusters → same synthesis | when synthesis succeeds, all methods ground=1.0; 3/6 runs failed identically across methods due to harness adapter bug (not a segmentation signal) |

## The "ours has everything" table — now with benchmark numbers

| Capability | personas-pipeline | strongest OSS comparator | Our position |
|---|---|---|---|
| Ingest behavioral records | crawler + mock connectors | persona-generation-workflow (CSV only) | strictly broader |
| Behavioral segmentation | greedy Jaccard on behavior sets | BERTopic / kmeans-emb / persona-generation-workflow k-means | **matches on golden set (ARI 1.0 shuffled)**; order-insensitive where BERTopic isn't |
| Grounded synthesis (`source_evidence`) | PersonaV1 with 21.5 evidence rows avg | *no OSS comparator ships this* | **qualitatively unique**; pgw-port/TinyTroupe/persona-hub all 0 |
| Twin runtime (boundary / refusal / meta) | TwinChat with explicit system-prompt protections | *(pending exp-7.02)* | — |
| Evaluation harness | groundedness / judge rubric / golden set | DeepEval (generic) | **persona-specific** metrics the others lack |
| Logged experiments | `output/experiments/` now includes exp-7.01–7.04 | none | — |

## Session spend

Actual LLM cost tracked in per-experiment `ledger.jsonl` and printed
at the end of each run:

- exp-7.04 intrinsic: $0.00 (no LLM)
- exp-7.03: $0.00 (no LLM)
- exp-7.04 downstream: $0.2092
- exp-7.01: ~$0.01 (pgw synthesis 2x + judge 4x)
- exp-7.02: *(pending — capped at ~$3.00)*

Running under the $15 session ceiling.

## What these benchmarks do and don't prove

**Prove:**
- Our `source_evidence` binding is qualitatively unique among the OSS
  comparators (exp-7.01 schema fidelity table).
- On *narrative* persona judging, a single-pass LLM summarizer matches
  our retry+groundedness pipeline — meaning our cost premium buys
  auditability, not prose quality.
- For small behavioral corpora, simple behavior-set Jaccard clustering
  matches modern topic modelers (exp-7.04 intrinsic, cross-method ARI 1.0).
- Grounded personas cover a tenant's records better than 50× more
  ungrounded samples from a generic persona pool (exp-7.03).

**Don't prove:**
- How our pipeline scales to larger corpora (38 records is small).
- How our twin handles a full adversarial prompt suite (exp-7.02 uses 10
  prompts × 2 personas — enough for signal, not for significance).
- Segmentation quality differences on datasets where methods actually
  diverge (this tenant is too clean to discriminate them on intrinsic).

See each experiment's `FINDINGS.md` for honest per-run caveats.
