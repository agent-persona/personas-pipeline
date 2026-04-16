"""Cluster comparison and stability helpers."""

from __future__ import annotations


def cluster_member_ids(cluster: dict) -> set[str]:
    """Extract record_ids from a cluster's sample_records."""
    return {sr["record_id"] for sr in cluster["sample_records"]}


def clusters_match_ignoring_id(a: list[dict], b: list[dict]) -> bool:
    """Check two cluster lists have the same membership, ignoring cluster_id and order."""
    if len(a) != len(b):
        return False
    a_sets = sorted(frozenset(cluster_member_ids(c)) for c in a)
    b_sets = sorted(frozenset(cluster_member_ids(c)) for c in b)
    return a_sets == b_sets


def membership_overlap(clusters_a: list[dict], clusters_b: list[dict]) -> float:
    """Compute fraction of records assigned to the same cluster across two runs.

    Uses best-match pairing: for each cluster in A, find the cluster in B with
    the highest overlap, and count agreements. This avoids index-order sensitivity.
    """
    a_members = [cluster_member_ids(c) for c in clusters_a]
    b_members = [cluster_member_ids(c) for c in clusters_b]

    all_ids = set()
    for s in a_members:
        all_ids |= s
    for s in b_members:
        all_ids |= s

    if not all_ids:
        return 1.0

    # Build id → cluster_index maps
    a_map = {}
    for idx, members in enumerate(a_members):
        for rid in members:
            a_map[rid] = idx
    b_map = {}
    for idx, members in enumerate(b_members):
        for rid in members:
            b_map[rid] = idx

    # Find best permutation mapping from A indices to B indices
    from collections import Counter
    pair_counts: Counter[tuple[int, int]] = Counter()
    for rid in all_ids:
        if rid in a_map and rid in b_map:
            pair_counts[(a_map[rid], b_map[rid])] += 1

    # Greedy best-match: assign each A cluster to the B cluster with most shared members
    used_b: set[int] = set()
    agreements = 0
    a_indices = sorted(set(a_map.values()))
    for a_idx in a_indices:
        best_b = -1
        best_count = 0
        for (ai, bi), count in pair_counts.items():
            if ai == a_idx and bi not in used_b and count > best_count:
                best_b = bi
                best_count = count
        if best_b >= 0:
            used_b.add(best_b)
            agreements += best_count

    return agreements / len(all_ids)


def summary_matches(a: dict, b: dict, ignore_keys: set[str] | None = None) -> bool:
    """Compare two cluster summaries, ignoring specified keys (e.g., cluster_id)."""
    ignore = ignore_keys or {"cluster_id"}
    for key in a:
        if key in ignore:
            continue
        if key == "summary":
            for sk in a["summary"]:
                if sk == "extra":
                    continue
                if a["summary"][sk] != b["summary"][sk]:
                    return False
        elif a.get(key) != b.get(key):
            return False
    return True
