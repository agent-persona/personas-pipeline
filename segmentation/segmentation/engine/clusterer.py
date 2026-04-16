from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from segmentation.models.features import UserFeatures


def jaccard_similarity(a: set, b: set) -> float:
    """Jaccard similarity between two sets. Returns 0.0 if both empty."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


@dataclass
class ClusterPrototype:
    """Maintained prototype that evolves as users join a cluster.

    Sets: union of all members.
    Numerics: running mean.
    Categories: running mode.
    """

    behaviors: set[str]
    pages: set[str]
    numeric_sums: dict[str, float]
    numeric_counts: dict[str, int]
    categorical_counters: dict[str, Counter]
    set_features: dict[str, set[str]]
    members: list[UserFeatures]

    @classmethod
    def from_user(cls, user: UserFeatures) -> ClusterPrototype:
        return cls(
            behaviors=set(user.behaviors),
            pages=set(user.pages),
            numeric_sums={k: v for k, v in user.numeric_features.items()},
            numeric_counts={k: 1 for k in user.numeric_features},
            categorical_counters={
                k: Counter({v: 1}) for k, v in user.categorical_features.items()
            },
            set_features={k: set(v) for k, v in user.set_features.items()},
            members=[user],
        )

    def add(self, user: UserFeatures) -> None:
        self.behaviors |= user.behaviors
        self.pages |= user.pages
        for k, v in user.numeric_features.items():
            self.numeric_sums[k] = self.numeric_sums.get(k, 0.0) + v
            self.numeric_counts[k] = self.numeric_counts.get(k, 0) + 1
        for k, v in user.categorical_features.items():
            if k not in self.categorical_counters:
                self.categorical_counters[k] = Counter()
            self.categorical_counters[k][v] += 1
        for k, v in user.set_features.items():
            if k not in self.set_features:
                self.set_features[k] = set()
            self.set_features[k] |= v
        self.members.append(user)

    def as_user_features(self) -> UserFeatures:
        return UserFeatures(
            user_id="__prototype__",
            tenant_id="__prototype__",
            behaviors=self.behaviors,
            pages=self.pages,
            numeric_features={
                k: self.numeric_sums[k] / self.numeric_counts[k]
                for k in self.numeric_sums
            },
            categorical_features={
                k: counter.most_common(1)[0][0]
                for k, counter in self.categorical_counters.items()
            },
            set_features=self.set_features,
        )


def cluster_users(
    features: list[UserFeatures],
    threshold: float = 0.4,
    min_cluster_size: int = 2,
    distance_metric: str = "jaccard",
    family_weights: dict[str, float] | None = None,
) -> list[list[UserFeatures]]:
    """Greedy agglomerative clustering.

    distance_metric="jaccard": original Jaccard-only path (unchanged).
    distance_metric="gower": Gower distance with maintained prototype.
    """
    if distance_metric == "jaccard":
        return _cluster_jaccard(features, threshold, min_cluster_size)
    elif distance_metric == "gower":
        return _cluster_gower(features, threshold, min_cluster_size, family_weights)
    else:
        raise ValueError(f"Unknown distance_metric: {distance_metric}")


def _cluster_jaccard(
    features: list[UserFeatures],
    threshold: float,
    min_cluster_size: int,
) -> list[list[UserFeatures]]:
    """Original Jaccard clustering — unchanged byte-for-byte."""
    clusters: list[list[UserFeatures]] = []

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

    return [c for c in clusters if len(c) >= min_cluster_size]


def _cluster_gower(
    features: list[UserFeatures],
    threshold: float,
    min_cluster_size: int,
    family_weights: dict[str, float] | None,
) -> list[list[UserFeatures]]:
    """Gower clustering with maintained prototype."""
    from segmentation.engine.gower import compute_ranges, gower_similarity

    ranges = compute_ranges(features)

    sorted_features = sorted(
        features,
        key=lambda f: (
            len(f.behaviors)
            + len(f.numeric_features)
            + len(f.categorical_features)
            + len(f.set_features)
        ),
        reverse=True,
    )

    prototypes: list[ClusterPrototype] = []

    for f in sorted_features:
        best_proto: ClusterPrototype | None = None
        best_sim = 0.0
        for proto in prototypes:
            sim = gower_similarity(
                f, proto.as_user_features(),
                numeric_ranges=ranges,
                family_weights=family_weights,
            )
            if sim > best_sim:  # strictly greater — mirrors Jaccard (D8)
                best_sim = sim
                best_proto = proto
        if best_proto is not None and best_sim >= threshold:
            best_proto.add(f)
        else:
            prototypes.append(ClusterPrototype.from_user(f))

    result = [p.members for p in prototypes if len(p.members) >= min_cluster_size]
    # Sort clusters deterministically by sorted member user_ids for stable output order
    result.sort(key=lambda c: sorted(u.user_id for u in c))
    return result
