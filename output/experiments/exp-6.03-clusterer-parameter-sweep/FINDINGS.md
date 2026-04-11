# Experiment 6.03: Clusterer Parameter Sweep

## Hypothesis
A parameter knee exists on similarity threshold times min_cluster_size that maximizes persona usefulness.

## Method
1. Swept the full grid of 20 clustering configs on the golden tenant.
2. Computed cluster count, noise rate, size spread, and centroid compactness for every config.
3. Ran synthesis and judge scoring on five representative threshold values at `min_cluster_size=2`.

- Tenant: `tenant_acme_corp`
- Source records: `38`
- Provider: `anthropic`

## Grid Summary
- Best compactness config: `t=0.6`, `m=1`
- Best cluster-count config: `t=0.6`, `m=1`
- Mean compactness across grid: `0.310`
- Mean noise rate across grid: `0.525`

## Threshold Sweep
- `t=0.1, m=1`: clusters=2, noise=0.00, compactness=0.43
- `t=0.1, m=2`: clusters=2, noise=0.00, compactness=0.43
- `t=0.1, m=3`: clusters=2, noise=0.00, compactness=0.43
- `t=0.1, m=5`: clusters=0, noise=1.00, compactness=0.00
- `t=0.2, m=1`: clusters=2, noise=0.00, compactness=0.43
- `t=0.2, m=2`: clusters=2, noise=0.00, compactness=0.43
- `t=0.2, m=3`: clusters=2, noise=0.00, compactness=0.43
- `t=0.2, m=5`: clusters=0, noise=1.00, compactness=0.00
- `t=0.4, m=1`: clusters=6, noise=0.00, compactness=0.90
- `t=0.4, m=2`: clusters=2, noise=0.50, compactness=0.71
- `t=0.4, m=3`: clusters=0, noise=1.00, compactness=0.00
- `t=0.4, m=5`: clusters=0, noise=1.00, compactness=0.00
- `t=0.6, m=1`: clusters=8, noise=0.00, compactness=1.00
- `t=0.6, m=2`: clusters=0, noise=1.00, compactness=0.00
- `t=0.6, m=3`: clusters=0, noise=1.00, compactness=0.00
- `t=0.6, m=5`: clusters=0, noise=1.00, compactness=0.00
- `t=0.8, m=1`: clusters=8, noise=0.00, compactness=1.00
- `t=0.8, m=2`: clusters=0, noise=1.00, compactness=0.00
- `t=0.8, m=3`: clusters=0, noise=1.00, compactness=0.00
- `t=0.8, m=5`: clusters=0, noise=1.00, compactness=0.00

## Synthesis Subset
- `t=0.1, m=2`: cluster_size=4, judge=4.00
- `t=0.2, m=2`: cluster_size=4, judge=4.00
- `t=0.4, m=2`: cluster_size=2, judge=4.00

## Decision
Adopt. The practical knee is `threshold=0.1` to `0.2` with `min_cluster_size=2`: it preserves both natural clusters with `0%` noise, while `threshold=0.4` at the same minimum size spikes noise to `50%`.

## Caveat
Tiny tenant: 37 records, 2 natural clusters. Use this as a parameter landscape, not a universal optimum.
