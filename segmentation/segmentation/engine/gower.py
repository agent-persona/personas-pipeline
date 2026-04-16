"""Gower distance for mixed-type UserFeatures."""

from __future__ import annotations

from segmentation.engine.clusterer import jaccard_similarity
from segmentation.models.features import UserFeatures


def compute_ranges(features: list[UserFeatures]) -> dict[str, float]:
    """Compute max - min per numeric feature across all users.

    Features appearing in only 1 user get range=0.
    """
    mins: dict[str, float] = {}
    maxs: dict[str, float] = {}
    for f in features:
        for name, val in f.numeric_features.items():
            if name not in mins or val < mins[name]:
                mins[name] = val
            if name not in maxs or val > maxs[name]:
                maxs[name] = val
    return {name: maxs[name] - mins[name] for name in mins}


def gower_distance(
    a: UserFeatures,
    b: UserFeatures,
    numeric_ranges: dict[str, float] | None = None,
    family_weights: dict[str, float] | None = None,
) -> float:
    """Compute Gower distance between two users.

    Returns a value in [0, 1]. Only features present in BOTH users are
    compared. Zero comparable dimensions returns 1.0 (maximum distance).
    """
    weights = family_weights or {"sets": 1.0, "numerics": 1.0, "categories": 1.0}
    w_sets = weights.get("sets", 1.0)
    w_nums = weights.get("numerics", 1.0)
    w_cats = weights.get("categories", 1.0)

    distances: list[float] = []
    dim_weights: list[float] = []

    # Count typed dimensions (numeric + categorical) that will actually be compared.
    # Named set_features are excluded — they are set-type like behaviors/pages
    # and shouldn't trigger the "skip identical implicit sets" rule.
    typed_dims = 0
    if numeric_ranges is not None and w_nums > 0:
        typed_dims += len(set(a.numeric_features) & set(b.numeric_features) & set(numeric_ranges))
    if w_cats > 0:
        typed_dims += len(set(a.categorical_features) & set(b.categorical_features))

    # Implicit set dimensions: _behaviors, _pages
    # When typed dimensions will actually be compared, identical implicit sets
    # are skipped so they don't dilute typed feature differences.
    # When no typed dimensions exist, all set dimensions are included (behavior-only).
    for a_set, b_set in [(a.behaviors, b.behaviors), (a.pages, b.pages)]:
        if a_set or b_set:  # At least one non-empty (D9)
            dist = 1.0 - jaccard_similarity(a_set, b_set)
            if dist > 0.0 or typed_dims == 0:
                distances.append(dist)
                dim_weights.append(w_sets)

    # Named set features
    shared_sets = set(a.set_features) & set(b.set_features)
    for name in shared_sets:
        dist = 1.0 - jaccard_similarity(a.set_features[name], b.set_features[name])
        distances.append(dist)
        dim_weights.append(w_sets)

    # Numeric features
    if numeric_ranges is not None:
        shared_nums = set(a.numeric_features) & set(b.numeric_features)
        for name in shared_nums:
            if name in numeric_ranges:
                r = numeric_ranges[name]
                if r == 0.0:
                    distances.append(0.0)
                else:
                    dist = min(abs(a.numeric_features[name] - b.numeric_features[name]) / r, 1.0)
                    distances.append(dist)
                dim_weights.append(w_nums)

    # Categorical features
    shared_cats = set(a.categorical_features) & set(b.categorical_features)
    for name in shared_cats:
        dist = 0.0 if a.categorical_features[name] == b.categorical_features[name] else 1.0
        distances.append(dist)
        dim_weights.append(w_cats)

    if not distances:
        return 1.0

    total_weight = sum(dim_weights)
    if total_weight == 0.0:
        return 1.0

    return sum(d * w for d, w in zip(distances, dim_weights)) / total_weight


def gower_similarity(
    a: UserFeatures,
    b: UserFeatures,
    numeric_ranges: dict[str, float] | None = None,
    family_weights: dict[str, float] | None = None,
) -> float:
    """1.0 - gower_distance."""
    return 1.0 - gower_distance(a, b, numeric_ranges, family_weights)
