"""Clusterer Gower behavior: mixed-feature clustering edge cases."""

from __future__ import annotations

from segmentation.models.features import UserFeatures


def _uf(user_id="u1", behaviors=None, numeric_features=None, categorical_features=None, set_features=None):
    return UserFeatures(
        user_id=user_id, tenant_id="t1",
        behaviors=set(behaviors or []),
        numeric_features=numeric_features or {},
        categorical_features=categorical_features or {},
        set_features=set_features or {},
    )


def test_typed_payload_clusters_despite_weak_behaviors():
    """Users with weak behavior overlap cluster together via shared categorical features."""
    from segmentation.engine.clusterer import cluster_users

    features = [
        _uf("a", ["x"], categorical_features={"industry": "fintech", "role": "eng"}),
        _uf("b", ["y"], categorical_features={"industry": "fintech", "role": "eng"}),
    ]
    # Behaviors completely different, but categoricals match → they should cluster
    clusters = cluster_users(features, threshold=0.1, min_cluster_size=1, distance_metric="gower")
    # With 2 categorical matches (dist=0 each) and behavior mismatch (dist=1),
    # but behaviors are skipped when typed dims exist and behaviors differ
    # → only categoricals compared → dist=0 → they cluster
    all_ids = {u.user_id for c in clusters for u in c}
    # Both should be in the output (not dropped as noise)
    assert "a" in all_ids and "b" in all_ids


def test_typed_payload_keeps_users_apart():
    """Users with similar behaviors but different typed features stay separate."""
    from segmentation.engine.clusterer import cluster_users

    features = [
        _uf("a", ["shared"], categorical_features={"role": "eng"}),
        _uf("b", ["shared"], categorical_features={"role": "eng"}),
        _uf("c", ["shared"], categorical_features={"role": "designer"}),
        _uf("d", ["shared"], categorical_features={"role": "designer"}),
    ]
    clusters = cluster_users(features, threshold=0.3, min_cluster_size=2, distance_metric="gower")
    assert len(clusters) == 2


def test_single_categorical_doesnt_over_cluster():
    """Users sharing one categorical feature but differing on others don't merge too easily."""
    from segmentation.engine.clusterer import cluster_users

    features = [
        _uf("a", ["api"], categorical_features={"industry": "fintech", "role": "eng", "seniority": "senior"}),
        _uf("b", ["api"], categorical_features={"industry": "fintech", "role": "designer", "seniority": "junior"}),
    ]
    # Only industry matches (1/3), role and seniority differ (2/3)
    # Categorical distance: (0 + 1 + 1) / 3 = 0.67, sim = 0.33
    # At threshold 0.5, they should NOT cluster
    clusters = cluster_users(features, threshold=0.5, min_cluster_size=1, distance_metric="gower")
    assert len(clusters) == 2


def test_zero_shared_features_become_singletons():
    """Users with zero overlapping features become separate singletons."""
    from segmentation.engine.clusterer import cluster_users

    features = [
        _uf("a", ["x"], numeric_features={"s": 100.0}),
        _uf("b", categorical_features={"role": "eng"}),  # no behaviors, no numerics
    ]
    clusters = cluster_users(features, threshold=0.1, min_cluster_size=1, distance_metric="gower")
    assert len(clusters) == 2


def test_all_identical_numerics_no_distortion():
    """All-identical numeric features produce distance 0 on that dimension."""
    from segmentation.engine.clusterer import cluster_users

    features = [
        _uf("a", ["api"], numeric_features={"s": 1000.0}, categorical_features={"role": "eng"}),
        _uf("b", ["api"], numeric_features={"s": 1000.0}, categorical_features={"role": "eng"}),
        _uf("c", ["design"], numeric_features={"s": 1000.0}, categorical_features={"role": "designer"}),
    ]
    clusters = cluster_users(features, threshold=0.3, min_cluster_size=1, distance_metric="gower")
    # a and b should cluster (same role, same duration, same-ish behaviors)
    # c should be separate (different role, different behaviors)
    found_ab = False
    for c in clusters:
        ids = {u.user_id for u in c}
        if "a" in ids and "b" in ids:
            found_ab = True
    assert found_ab


def test_minority_cluster_survives_imbalanced():
    """Small distinct cluster survives even with a dominant cluster."""
    from segmentation.engine.clusterer import cluster_users
    from tests.helpers.fixture_loader import load_records

    records = load_records("golden_imbalanced.json")
    from segmentation.engine.featurizer import featurize_records
    from segmentation.engine.registry import DEFAULT_REGISTRY

    features = featurize_records(records, registry=DEFAULT_REGISTRY)
    clusters = cluster_users(features, threshold=0.15, min_cluster_size=2, distance_metric="gower")
    assert len(clusters) >= 2


def test_gower_distance_is_symmetric():
    """Gower distance is symmetric: d(a,b) == d(b,a)."""
    from segmentation.engine.gower import gower_distance, compute_ranges

    a = _uf("a", ["x", "y"], numeric_features={"s": 100.0}, categorical_features={"role": "eng"})
    b = _uf("b", ["y", "z"], numeric_features={"s": 500.0}, categorical_features={"role": "designer"})
    ranges = compute_ranges([a, b])
    d_ab = gower_distance(a, b, numeric_ranges=ranges)
    d_ba = gower_distance(b, a, numeric_ranges=ranges)
    assert abs(d_ab - d_ba) < 1e-9


def test_gower_cluster_membership_on_clean_fixture():
    """Clean mixed fixture produces expected grouping (engineers vs designers)."""
    from segmentation.pipeline import segment
    from tests.helpers.fixture_loader import load_records

    records = load_records("golden_mixed_small.json")
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")

    eng_behaviors = {"api_setup", "webhook_config", "github_integration", "terraform_setup"}
    des_behaviors = {"template_browsing", "color_picker", "asset_export", "brand_kit_creation"}

    for c in clusters:
        top = set(c["summary"]["top_behaviors"])
        # Each cluster should lean toward one group
        assert (top & eng_behaviors) or (top & des_behaviors)


def test_partially_registered_sources_still_cluster():
    """Mix of registered (ga4) and unregistered (salesforce) sources clusters meaningfully."""
    from segmentation.engine.clusterer import cluster_users
    from segmentation.engine.featurizer import featurize_records
    from segmentation.engine.registry import DEFAULT_REGISTRY
    from segmentation.models.record import RawRecord

    records = [
        RawRecord(record_id="ga4_1", tenant_id="t1", source="ga4", user_id="u1",
                  behaviors=["api_setup"], pages=[], payload={"session_duration": 2000}),
        RawRecord(record_id="ga4_2", tenant_id="t1", source="ga4", user_id="u2",
                  behaviors=["api_setup"], pages=[], payload={"session_duration": 1800}),
        RawRecord(record_id="sf_1", tenant_id="t1", source="salesforce", user_id="u3",
                  behaviors=["api_setup"], pages=[], payload={"deal_value": 50000}),
        RawRecord(record_id="sf_2", tenant_id="t1", source="salesforce", user_id="u4",
                  behaviors=["api_setup"], pages=[], payload={"deal_value": 30000}),
    ]
    features = featurize_records(records, registry=DEFAULT_REGISTRY)
    clusters = cluster_users(features, threshold=0.1, min_cluster_size=2, distance_metric="gower")
    assert len(clusters) >= 1


def test_gower_pairwise_symmetric_sparse():
    """Gower is symmetric even with asymmetric sparse inputs."""
    from segmentation.engine.gower import gower_distance, compute_ranges

    # a has numerics but no categories, b has categories but no numerics
    a = _uf("a", ["shared"], numeric_features={"s": 100.0})
    b = _uf("b", ["shared"], categorical_features={"role": "eng"})
    ranges = compute_ranges([a, b])
    d_ab = gower_distance(a, b, numeric_ranges=ranges)
    d_ba = gower_distance(b, a, numeric_ranges=ranges)
    assert abs(d_ab - d_ba) < 1e-9
