"""Task 5: Clusterer with maintained prototype — unit tests."""

from __future__ import annotations

import pytest
from segmentation.models.features import UserFeatures


def _uf(user_id="u1", behaviors=None, numeric_features=None, categorical_features=None, set_features=None):
    return UserFeatures(
        user_id=user_id,
        tenant_id="t1",
        behaviors=set(behaviors or []),
        numeric_features=numeric_features or {},
        categorical_features=categorical_features or {},
        set_features=set_features or {},
    )


def test_default_jaccard_unchanged():
    """Default params produce identical results to current implementation."""
    from segmentation.engine.clusterer import cluster_users

    # 4 engineers + 4 designers with overlapping behaviors within group
    features = [
        _uf("eng1", ["api_setup", "webhook_config"]),
        _uf("eng2", ["api_setup", "github_integration"]),
        _uf("des1", ["template_browsing", "color_picker"]),
        _uf("des2", ["template_browsing", "asset_export"]),
    ]
    clusters = cluster_users(features, threshold=0.2, min_cluster_size=2)
    assert len(clusters) == 2
    # Each cluster should have 2 members
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [2, 2]


def test_explicit_jaccard_same_as_default():
    """distance_metric='jaccard' produces same result as default."""
    from segmentation.engine.clusterer import cluster_users

    features = [
        _uf("eng1", ["api_setup", "webhook_config"]),
        _uf("eng2", ["api_setup", "github_integration"]),
        _uf("des1", ["template_browsing", "color_picker"]),
        _uf("des2", ["template_browsing", "asset_export"]),
    ]
    default = cluster_users(features, threshold=0.2, min_cluster_size=2)
    explicit = cluster_users(features, threshold=0.2, min_cluster_size=2, distance_metric="jaccard")
    assert len(default) == len(explicit)
    for d, e in zip(default, explicit):
        assert {u.user_id for u in d} == {u.user_id for u in e}


def test_gower_behavior_only():
    """Gower with behavior-only data still produces clusters."""
    from segmentation.engine.clusterer import cluster_users

    features = [
        _uf("eng1", ["api_setup", "webhook_config"]),
        _uf("eng2", ["api_setup", "github_integration"]),
        _uf("des1", ["template_browsing", "color_picker"]),
        _uf("des2", ["template_browsing", "asset_export"]),
    ]
    clusters = cluster_users(features, threshold=0.2, min_cluster_size=2, distance_metric="gower")
    assert len(clusters) >= 1


def test_gower_separates_by_categorical():
    """Gower separates users by categorical features."""
    from segmentation.engine.clusterer import cluster_users

    features = [
        _uf("eng1", ["shared_behavior"], categorical_features={"role": "eng"}),
        _uf("eng2", ["shared_behavior"], categorical_features={"role": "eng"}),
        _uf("des1", ["shared_behavior"], categorical_features={"role": "designer"}),
        _uf("des2", ["shared_behavior"], categorical_features={"role": "designer"}),
    ]
    clusters = cluster_users(features, threshold=0.3, min_cluster_size=2, distance_metric="gower")
    assert len(clusters) == 2
    cluster_roles = []
    for c in clusters:
        roles = {u.categorical_features.get("role") for u in c}
        cluster_roles.append(roles)
    # Each cluster should have users of the same role
    assert {"eng"} in cluster_roles
    assert {"designer"} in cluster_roles


def test_gower_separates_by_numeric():
    """Gower separates users by numeric features when behaviors are identical."""
    from segmentation.engine.clusterer import cluster_users

    features = [
        _uf("a", ["shared"], numeric_features={"session_duration": 100.0}),
        _uf("b", ["shared"], numeric_features={"session_duration": 120.0}),
        _uf("c", ["shared"], numeric_features={"session_duration": 9000.0}),
        _uf("d", ["shared"], numeric_features={"session_duration": 9200.0}),
    ]
    clusters = cluster_users(features, threshold=0.4, min_cluster_size=2, distance_metric="gower")
    assert len(clusters) == 2


def test_min_cluster_size_gower():
    """min_cluster_size drops small clusters in Gower mode."""
    from segmentation.engine.clusterer import cluster_users

    features = [
        _uf("a", ["api_setup"]),
        _uf("b", ["api_setup"]),
        _uf("outlier", ["completely_unique_behavior_xyz"]),
    ]
    clusters = cluster_users(features, threshold=0.3, min_cluster_size=2, distance_metric="gower")
    # outlier should be dropped
    all_ids = {u.user_id for c in clusters for u in c}
    assert "outlier" not in all_ids


def test_unknown_metric_raises():
    """Unknown distance_metric raises ValueError."""
    from segmentation.engine.clusterer import cluster_users

    with pytest.raises(ValueError, match="Unknown distance_metric"):
        cluster_users([], distance_metric="unknown")


def test_empty_input():
    """Empty features list returns empty clusters."""
    from segmentation.engine.clusterer import cluster_users

    assert cluster_users([], distance_metric="gower") == []


def test_single_user():
    """Single user with min_cluster_size=1 returns one cluster."""
    from segmentation.engine.clusterer import cluster_users

    features = [_uf("solo", ["a"])]
    clusters = cluster_users(features, min_cluster_size=1, distance_metric="gower")
    assert len(clusters) == 1
    assert clusters[0][0].user_id == "solo"


def test_higher_threshold_more_clusters():
    """Higher threshold produces at least as many clusters as lower threshold."""
    from segmentation.engine.clusterer import cluster_users

    features = [
        _uf(f"u{i}", [f"behavior_{i % 3}"], categorical_features={"role": f"role_{i % 4}"})
        for i in range(12)
    ]
    low = cluster_users(features, threshold=0.1, min_cluster_size=1, distance_metric="gower")
    high = cluster_users(features, threshold=0.8, min_cluster_size=1, distance_metric="gower")
    assert len(high) >= len(low)


def test_sort_order_richest_first():
    """Users with more features seed clusters first."""
    from segmentation.engine.clusterer import cluster_users

    rich = _uf("rich", ["a", "b", "c", "d", "e"], categorical_features={"r": "x", "i": "y", "s": "z"})
    sparse = _uf("sparse", ["a", "b"])
    clusters = cluster_users([sparse, rich], threshold=0.01, min_cluster_size=1, distance_metric="gower")
    # Rich user should seed — be first member of first cluster
    assert clusters[0][0].user_id == "rich"


def test_threshold_zero_zero_similarity_new_cluster():
    """At threshold=0.0, users with zero shared features still start new clusters."""
    from segmentation.engine.clusterer import cluster_users

    # These users share nothing
    features = [
        _uf("a", ["x"]),
        _uf("b", categorical_features={"role": "eng"}),  # no behaviors
    ]
    clusters = cluster_users(features, threshold=0.0, min_cluster_size=1, distance_metric="gower")
    # b has no behaviors, a has no categoricals → zero overlap → sim=0.0
    # 0.0 > 0.0 is False → b starts new cluster
    assert len(clusters) >= 2


def test_threshold_one_all_singletons():
    """At threshold=1.0, only identical users merge; most dropped by min_cluster_size."""
    from segmentation.engine.clusterer import cluster_users

    features = [
        _uf("a", ["x", "y"]),
        _uf("b", ["x", "z"]),
        _uf("c", ["w"]),
    ]
    clusters = cluster_users(features, threshold=1.0, min_cluster_size=2, distance_metric="gower")
    # No pair is identical → all singletons → all dropped
    assert len(clusters) == 0


def test_prototype_evolves():
    """Cluster prototype evolves: 4th user matches evolved prototype better than seed alone."""
    from segmentation.engine.clusterer import cluster_users

    # 3 users with behaviors {A}, {B}, {C} join same cluster at low threshold
    # Prototype behaviors = {A,B,C} after all join
    # 4th user with {A,B} should match the evolved prototype well
    features = [
        _uf("seed", ["A", "B", "C", "D"]),  # dense seed
        _uf("u2", ["A", "B"]),
        _uf("u3", ["B", "C"]),
        _uf("u4", ["A", "B"]),  # should match prototype {A,B,C,D} well
    ]
    clusters = cluster_users(features, threshold=0.1, min_cluster_size=1, distance_metric="gower")
    # All should be in one cluster
    assert len(clusters) == 1
    assert len(clusters[0]) == 4


def test_prototype_numeric_mean():
    """Prototype numeric features are the running mean of members."""
    from segmentation.engine.clusterer import cluster_users, ClusterPrototype

    proto = ClusterPrototype.from_user(_uf("u1", ["a"], numeric_features={"s": 100.0}))
    proto.add(_uf("u2", ["a"], numeric_features={"s": 200.0}))
    uf = proto.as_user_features()
    assert uf.numeric_features["s"] == 150.0


def test_prototype_categorical_mode():
    """Prototype categorical features are the running mode."""
    from segmentation.engine.clusterer import ClusterPrototype

    proto = ClusterPrototype.from_user(_uf("u1", categorical_features={"industry": "fintech"}))
    proto.add(_uf("u2", categorical_features={"industry": "saas"}))
    proto.add(_uf("u3", categorical_features={"industry": "fintech"}))
    uf = proto.as_user_features()
    assert uf.categorical_features["industry"] == "fintech"


def test_order_sensitivity():
    """Prototype-based clustering is stable across different input orders."""
    from segmentation.engine.clusterer import cluster_users
    import random

    features = [
        _uf("eng1", ["api_setup", "webhook_config"], categorical_features={"role": "eng"}),
        _uf("eng2", ["api_setup", "github"], categorical_features={"role": "eng"}),
        _uf("des1", ["template", "color_picker"], categorical_features={"role": "designer"}),
        _uf("des2", ["template", "asset_export"], categorical_features={"role": "designer"}),
    ]

    results = []
    for seed in range(5):
        shuffled = features.copy()
        random.Random(seed).shuffle(shuffled)
        clusters = cluster_users(shuffled, threshold=0.2, min_cluster_size=2, distance_metric="gower")
        cluster_sets = frozenset(
            frozenset(u.user_id for u in c) for c in clusters
        )
        results.append(cluster_sets)

    # All 5 runs should produce the same clustering
    assert all(r == results[0] for r in results)


def test_family_weights_passthrough():
    """family_weights param affects clustering: zeroing numerics ignores them."""
    from segmentation.engine.clusterer import cluster_users

    # Users differ only in numeric features
    features = [
        _uf("a", ["shared"], numeric_features={"s": 100.0}),
        _uf("b", ["shared"], numeric_features={"s": 9000.0}),
    ]
    # With numerics weighted, they might separate at high threshold
    # With numerics zeroed, they're identical on sets → same cluster
    clusters = cluster_users(
        features, threshold=0.5, min_cluster_size=1, distance_metric="gower",
        family_weights={"sets": 1.0, "numerics": 0.0, "categories": 0.0},
    )
    # Should be 1 cluster (numerics ignored, behaviors identical)
    assert len(clusters) == 1
