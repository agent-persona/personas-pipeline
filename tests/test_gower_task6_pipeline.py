"""Task 6: Pipeline + summarizer integration — unit tests."""

from __future__ import annotations

import sys
sys.path.insert(0, "../crawler")

from crawler.connectors import GA4Connector, HubspotConnector, IntercomConnector
from segmentation.models.record import RawRecord


def _get_mock_records():
    """Fetch all mock records from the 3 connectors."""
    tenant = "tenant_acme_corp"
    records = []
    for Conn in [GA4Connector, HubspotConnector, IntercomConnector]:
        records.extend(Conn().fetch(tenant))
    return [RawRecord.model_validate(r.model_dump()) for r in records]


def test_default_params_unchanged():
    """Default params produce valid ClusterData identical to before."""
    from segmentation.pipeline import segment
    sys.path.insert(0, "../synthesis")
    from synthesis.models.cluster import ClusterData

    records = _get_mock_records()
    clusters = segment(
        records,
        tenant_industry="B2B SaaS",
        tenant_product="Project management tool",
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    assert len(clusters) == 2
    for c in clusters:
        ClusterData.model_validate(c)


def test_jaccard_explicit_identical():
    """Explicit distance_metric='jaccard' matches default."""
    from segmentation.pipeline import segment

    records = _get_mock_records()
    default = segment(records, similarity_threshold=0.15, min_cluster_size=2)
    explicit = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="jaccard")
    assert len(default) == len(explicit)


def test_gower_produces_valid_output():
    """Gower mode produces valid ClusterData."""
    from segmentation.pipeline import segment
    sys.path.insert(0, "../synthesis")
    from synthesis.models.cluster import ClusterData

    records = _get_mock_records()
    clusters = segment(
        records,
        tenant_industry="B2B SaaS",
        tenant_product="Project management tool",
        similarity_threshold=0.15,
        min_cluster_size=2,
        distance_metric="gower",
    )
    assert len(clusters) >= 1
    for c in clusters:
        ClusterData.model_validate(c)


def test_gower_with_explicit_registry():
    """Gower with explicit DEFAULT_REGISTRY works."""
    from segmentation.pipeline import segment
    from segmentation.engine.registry import DEFAULT_REGISTRY

    records = _get_mock_records()
    clusters = segment(
        records,
        similarity_threshold=0.15,
        min_cluster_size=2,
        distance_metric="gower",
        feature_registry=DEFAULT_REGISTRY,
    )
    assert len(clusters) >= 1


def test_gower_output_validates_clusterdata():
    """Every Gower output dict validates as ClusterData."""
    from segmentation.pipeline import segment
    sys.path.insert(0, "../synthesis")
    from synthesis.models.cluster import ClusterData

    records = _get_mock_records()
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    for c in clusters:
        validated = ClusterData.model_validate(c)
        assert validated.cluster_id
        assert validated.summary.cluster_size >= 2


def test_gower_extra_has_typed_features():
    """Gower output includes typed_features in extra with numeric_averages."""
    from segmentation.pipeline import segment

    records = _get_mock_records()
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    # At least one cluster should have typed_features
    has_typed = any("typed_features" in c["summary"]["extra"] for c in clusters)
    assert has_typed
    for c in clusters:
        if "typed_features" in c["summary"]["extra"]:
            tf = c["summary"]["extra"]["typed_features"]
            assert "numeric_averages" in tf
            assert "session_duration" in tf["numeric_averages"]


def test_gower_avg_session_duration_populated():
    """avg_session_duration_seconds is populated for clusters with GA4 data."""
    from segmentation.pipeline import segment

    records = _get_mock_records()
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    # Both clusters have GA4 data
    for c in clusters:
        duration = c["summary"]["avg_session_duration_seconds"]
        assert duration is not None
        assert isinstance(duration, float)
        assert 100 < duration < 10000


def test_sample_records_populated():
    """Every cluster has at least 1 sample record."""
    from segmentation.pipeline import segment

    records = _get_mock_records()
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    for c in clusters:
        assert len(c["sample_records"]) >= 1


def test_gower_categorical_modes_in_extra():
    """Gower clusters with HubSpot data have categorical_modes with industry key."""
    from segmentation.pipeline import segment

    records = _get_mock_records()
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    has_industry = False
    for c in clusters:
        if "typed_features" in c["summary"]["extra"]:
            modes = c["summary"]["extra"]["typed_features"].get("categorical_modes", {})
            if "industry" in modes:
                has_industry = True
    assert has_industry


def test_jaccard_ignores_registry():
    """Passing feature_registry with jaccard mode produces identical output to no registry."""
    from segmentation.pipeline import segment
    from segmentation.engine.registry import DEFAULT_REGISTRY

    records = _get_mock_records()
    without = segment(records, similarity_threshold=0.15, min_cluster_size=2)
    with_reg = segment(records, similarity_threshold=0.15, min_cluster_size=2, feature_registry=DEFAULT_REGISTRY)
    assert len(without) == len(with_reg)
    for w, r in zip(without, with_reg):
        assert w["summary"]["cluster_size"] == r["summary"]["cluster_size"]
        assert w["summary"]["top_behaviors"] == r["summary"]["top_behaviors"]
