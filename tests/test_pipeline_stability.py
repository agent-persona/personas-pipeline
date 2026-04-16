"""Pipeline stability: permutation sensitivity with proper metrics."""

from __future__ import annotations

import random

from tests.helpers.fixture_loader import load_records
from tests.helpers.cluster_assertions import membership_overlap


def test_stability_cluster_count_5_permutations():
    """5 shuffled runs produce identical cluster count."""
    from segmentation.pipeline import segment

    records = load_records("golden_mixed_small.json")
    counts = []
    for seed in range(5):
        shuffled = records.copy()
        random.Random(seed).shuffle(shuffled)
        clusters = segment(shuffled, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
        counts.append(len(clusters))
    assert all(c == counts[0] for c in counts), f"Cluster counts varied: {counts}"


def test_stability_membership_overlap_80pct():
    """Across 5 shuffled runs, membership overlap is >= 80% using best-match pairing."""
    from segmentation.pipeline import segment

    records = load_records("golden_mixed_small.json")
    all_runs = []
    for seed in range(5):
        shuffled = records.copy()
        random.Random(seed).shuffle(shuffled)
        clusters = segment(shuffled, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
        all_runs.append(clusters)

    ref = all_runs[0]
    for other in all_runs[1:]:
        overlap = membership_overlap(ref, other)
        assert overlap >= 0.8, f"Membership overlap {overlap:.2f} < 0.80"


def test_stability_top_behaviors_consistent():
    """Top behaviors sets are consistent across permutations."""
    from segmentation.pipeline import segment

    records = load_records("golden_mixed_small.json")
    behavior_sets_per_run = []
    for seed in range(5):
        shuffled = records.copy()
        random.Random(seed).shuffle(shuffled)
        clusters = segment(shuffled, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
        run_sets = frozenset(
            frozenset(c["summary"]["top_behaviors"]) for c in clusters
        )
        behavior_sets_per_run.append(run_sets)

    # All runs should produce the same top behavior sets
    assert all(bs == behavior_sets_per_run[0] for bs in behavior_sets_per_run)


def test_stability_cluster_count_delta_zero():
    """Cluster count delta across runs is 0."""
    from segmentation.pipeline import segment

    records = load_records("golden_mixed_small.json")
    counts = []
    for seed in range(5):
        shuffled = records.copy()
        random.Random(seed).shuffle(shuffled)
        clusters = segment(shuffled, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
        counts.append(len(clusters))
    delta = max(counts) - min(counts)
    assert delta == 0, f"Cluster count delta: {delta}"


def test_stability_instability_below_threshold():
    """Instability (1 - overlap) is below 0.2 across all run pairs."""
    from segmentation.pipeline import segment

    records = load_records("golden_mixed_small.json")
    all_runs = []
    for seed in range(5):
        shuffled = records.copy()
        random.Random(seed).shuffle(shuffled)
        clusters = segment(shuffled, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
        all_runs.append(clusters)

    for i in range(len(all_runs)):
        for j in range(i + 1, len(all_runs)):
            overlap = membership_overlap(all_runs[i], all_runs[j])
            instability = 1.0 - overlap
            assert instability <= 0.2, f"Instability {instability:.2f} between runs {i} and {j}"
