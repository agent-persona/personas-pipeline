from __future__ import annotations

from segmentation.models.features import UserFeatures


def jaccard_similarity(a: set, b: set) -> float:
    """Jaccard similarity between two sets. Returns 0.0 if both empty."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def cluster_users(
    features: list[UserFeatures],
    threshold: float = 0.4,
    min_cluster_size: int = 2,
) -> list[list[UserFeatures]]:
    """Greedy agglomerative clustering by Jaccard similarity over behaviors.

    Each user is compared against the centroid (union of behaviors) of every
    existing cluster. If the similarity exceeds `threshold`, the user joins
    that cluster; otherwise a new cluster is started.

    This is intentionally simple — it has no external dependencies, runs in
    O(n^2 / k), and is good enough for the v1 pipeline. The protocol is
    designed so HDBSCAN, KMeans, or an embedding-based clusterer can drop in
    behind the same `cluster_users` call signature.

    Clusters smaller than `min_cluster_size` are dropped (treated as noise).
    """
    clusters: list[list[UserFeatures]] = []

    # Sort by behavior count descending so dense users seed clusters first
    sorted_features = sorted(
        features,
        key=lambda f: len(f.behaviors),
        reverse=True,
    )

    for f in sorted_features:
        best_cluster: list[UserFeatures] | None = None
        best_sim = 0.0
        for cluster in clusters:
            centroid = set().union(*(m.behaviors for m in cluster))
            sim = jaccard_similarity(f.behaviors, centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster

        if best_cluster is not None and best_sim >= threshold:
            best_cluster.append(f)
        else:
            clusters.append([f])

    # Drop noise clusters
    return [c for c in clusters if len(c) >= min_cluster_size]
