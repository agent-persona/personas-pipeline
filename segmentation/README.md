# segmentation

**Stage 2.** Take raw behavioral records (from `crawler/`) and group them into
clusters that each represent one behavioral segment. Emit a cluster summary
dict in the exact shape that `synthesis/` consumes.

## What lives here

```
segmentation/
├── segmentation/
│   ├── pipeline.py           # segment() — the public entry point
│   ├── engine/
│   │   ├── featurizer.py     # Aggregate records by user_id -> UserFeatures
│   │   ├── clusterer.py      # Greedy Jaccard clustering (drop-in replaceable)
│   │   └── summarizer.py     # UserFeatures -> synthesis-shaped dict
│   └── models/
│       ├── record.py         # RawRecord — mirrors crawler.Record
│       └── features.py       # UserFeatures — internal aggregation type
└── pyproject.toml
```

## I/O contract

Input:
- `records: list[RawRecord]` — typically the crawler's output round-tripped
  through `model_dump()`.
- `tenant_industry`, `tenant_product`, `existing_persona_names` — propagated
  into the cluster summary for synthesis context.
- `similarity_threshold` (default `0.4`) — Jaccard cutoff for a user to join
  a cluster.
- `min_cluster_size` (default `2`) — smaller clusters are dropped as noise.

Output: `list[dict]`. Each dict is a cluster summary that validates against
`synthesis.models.ClusterData`. The keys are:

```
{
  "cluster_id":       str,
  "tenant": {
    "tenant_id":                str,
    "industry":                 str | None,
    "product_description":      str | None,
    "existing_persona_names":   list[str],
  },
  "summary": {
    "cluster_size":                    int,
    "top_behaviors":                   list[str],
    "top_pages":                       list[str],
    "conversion_rate":                 float | None,
    "avg_session_duration_seconds":    float | None,
    "top_referrers":                   list[str],
    "extra":                           dict,
  },
  "sample_records":  list[{record_id, source, timestamp, payload}],
  "enrichment":      { firmographic, intent_signals, technographic, extra },
}
```

This dict is returned **as a plain dict**, not a Pydantic model, so
segmentation has no runtime dependency on synthesis. Either side validates
independently.

## How to run standalone

```python
from segmentation.pipeline import segment
from segmentation.models.record import RawRecord

records = [RawRecord(...), ...]
clusters = segment(
    records,
    tenant_industry="B2B SaaS",
    tenant_product="Project management tool",
    similarity_threshold=0.15,
    min_cluster_size=2,
)
```

## Knobs you can turn

- **Similarity threshold** (`pipeline.py::segment`): raising it produces
  more, tighter clusters; lowering it produces fewer, broader ones.
- **Clusterer implementation** (`engine/clusterer.py`): `cluster_users()` is
  intentionally trivial (greedy Jaccard). The call signature is stable so
  HDBSCAN, KMeans, or an embedding-based clusterer can drop in behind it.
- **Featurizer** (`engine/featurizer.py`): today we aggregate `behaviors`,
  `pages`, `sources` as sets keyed by `user_id`. Add new feature fields to
  `UserFeatures` and wire them through if your experiment needs embeddings,
  recency weights, etc.
- **Sample size** (`engine/summarizer.py::DEFAULT_SAMPLE_SIZE`): how many
  records get forwarded to synthesis per cluster. Larger = better grounding,
  higher synthesis cost.

## Scientific backing

This module is where **Problem space 6 — Population distinctiveness &
coverage** was run. The clustering defaults (greedy Jaccard, the similarity
threshold, `DEFAULT_SAMPLE_SIZE`) were picked after head-to-head runs on:

- **Distinctiveness floor.** Pairwise cluster embedding distance with a
  reject threshold, measuring whether the downstream persona set improves.
  Coverage-gap analysis is recorded under `output/experiments/` (6.02
  report).
- **Coverage gaps.** What % of the population is unrepresented when records
  fall into no cluster.
- **Cluster count sweep.** Varying threshold / forcing target cluster counts
  {3, 5, 7, 10, 15} to find the knee on distinctiveness vs usefulness.
- **Stability across reruns.** Segmentation run 5× with permuted record
  order to check whether the same clusters reappear (the "stable persona
  ID" question).
- **Long-tail viability.** Forcing a single tiny cluster through to
  synthesis and comparing quality to the largest cluster.

Problem space 3 — Groundedness also exercises this module: the sparse-data
ablation downsamples records *before* clustering, so cluster shape itself is
a variable.

## Coordination note

Space 6 iterations also touch `synthesis/` (contrast prompting, fan-out).
Schema/pipeline changes in `synthesis/` move on a separate branch to keep
the two spaces from stepping on each other.

## Tests

Parent repo has `segmentation/segmentation/tests/` with featurizer, clusterer,
summarizer, and pipeline tests. Copy them over if you want a regression net
under your experiment.
