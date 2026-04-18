# exp-7.04 intrinsic — segmentation head-to-head (no LLM)

**Tenant:** `tenant_acme_corp`  
**Input:** 8 users / 38 records  
**Seeds:** [0, 1, 2, 3, 4] (input order permuted per seed for stability ARI)  

## Results

| Method | Coverage% | #clusters | Stability ARI (mean over 4 perms vs seed 0) | Error |
|---|---:|---:|---:|---|
| ours-jaccard | 100.0% | 2 | 1.000 |  |
| bertopic | 100.0% | 2 | 1.000 |  |
| top2vec | — | — | — | ValueError: need at least one array to concatenate |
| kmeans-emb | 100.0% | 2 | 1.000 |  |

## Cross-method agreement on seed-0 labelings

ARI = 1.0 means two methods put every user in the same group.

| Pair | ARI |
|---|---:|
| ours-jaccard vs bertopic | 1.000 |
| ours-jaccard vs kmeans-emb | 1.000 |
| bertopic vs kmeans-emb | 1.000 |

## Honest caveats

- The golden tenant has only ~8 aggregated users. Topic-modeling methods (BERTopic, Top2Vec) are designed for corpora of hundreds+; small-corpus behavior is informative but not representative.
- Stability ARI is intra-method (same method, permuted input). The cross-method section above is a separate measure — ARI between different methods' seed-0 assignments.
- No LLM was called in this benchmark. Cost: $0.