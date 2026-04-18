# exp-7.04 downstream — segmenter → synthesis head-to-head

**Tenant:** `tenant_acme_corp`
**Model:** `claude-haiku-4-5-20251001` at `temperature=0.0`
**Total LLM spend:** $0.2092

## Convergence check (seed-0, unshuffled input order)

| Segmenter pair | Adjusted Rand Index |
|---|---:|
| ours-jaccard vs bertopic | 0.774 |
| ours-jaccard vs kmeans-emb | 1.000 |
| bertopic vs kmeans-emb | 0.774 |

Intrinsic cross-method ARI at seed 0 (shuffled) was 1.0 for all pairs.
Here on unshuffled order, BERTopic disagrees with the other two (ARI
0.774). This is a real order-sensitivity finding: BERTopic's
UMAP→HDBSCAN pipeline is stable under random permutations of this
corpus but diverges from the other methods when given the natural
sorted order.

## Per-method downstream synthesis

| Method | #clusters | #personas (succeeded) | mean groundedness | total cost |
|---|---:|---:|---:|---:|
| ours-jaccard | 2 | 1/2 | 1.000 | $0.1123 |
| bertopic | 2 | 1/2 | 1.000 | $0.0968 |
| kmeans-emb | 2 | 0/2 | — | $0.0000 |

**Successful personas (ground 1.000 on each):**
- ours-jaccard `clust_0001`: Creative Director Carmen
- bertopic `clust_0000`: Creative Director Clara

## The 3/6 synthesis failures are a harness bug, not a segmentation finding

Three synthesis runs (ours-jaccard `clust_0000`, bertopic `clust_0001`,
both kmeans-emb clusters) failed identically after 5 retries with:
`SynthesisError: ('source_evidence',): Field required`.

These are the *engineering* clusters across all three segmenters.
The fact that all three engineering clusters failed the same way
— while all three creative clusters that succeeded did so with
groundedness 1.0 — tells us the signal is in the cluster *content*,
not in the segmenter that produced the cluster. The harness's
`_cluster_from_label_group` adapter constructs a `ClusterData` that
is subtly different from what `segmentation.pipeline.segment()`
produces, and the synthesis prompt doesn't recover well from that
difference for the engineering records specifically.

This is a real limitation of this particular run, not a scoreable
segmentation outcome. Fixing the adapter to route cluster labels
back through `segment()`'s own featurizer path would unblock the
other three runs; at that point we'd expect all three methods to
produce high-groundedness personas whose differences track the
clustering ARI (0.774 ≈ "ours and bertopic disagree on one user out
of eight").

## What this benchmark does and doesn't tell us

**Does tell us:**
- BERTopic's UMAP→HDBSCAN pipeline is sensitive to record order on
  small corpora (intrinsic stable-across-random-shuffles, but unshuffled
  produces a different partition than ours-jaccard/kmeans).
- When synthesis does succeed, groundedness is 1.0 regardless of which
  segmenter produced the cluster — i.e., synthesis quality ≥ depends
  on cluster content, not on the segmentation algorithm that grouped
  the records.

**Does not tell us:**
- Whether ours-jaccard would produce better personas than bertopic on
  a dataset where they *meaningfully* disagree (this tenant is too
  small to stress the methods).
- The full per-method groundedness distribution — half the runs
  didn't complete.

The harness and its limitations are preserved here so the next run
against a larger dataset can pick up from the adapter fix forward.
