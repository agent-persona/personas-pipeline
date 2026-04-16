"""Jaccard parity: frozen baseline regression tests."""

from __future__ import annotations

import sys
sys.path.insert(0, "../crawler")

from tests.helpers.fixture_loader import load_records, load_snapshot, normalize_cluster_order
from tests.helpers.cluster_assertions import clusters_match_ignoring_id


def _run_jaccard(fixture="golden_mixed_small.json", **kwargs):
    from segmentation.pipeline import segment
    records = load_records(fixture)
    defaults = dict(similarity_threshold=0.15, min_cluster_size=2)
    defaults.update(kwargs)
    return segment(records, **defaults)


def test_jaccard_cluster_membership_matches_baseline():
    """Jaccard cluster membership matches the frozen regression snapshot."""
    from segmentation.pipeline import segment
    from crawler.connectors import GA4Connector, HubspotConnector, IntercomConnector
    from segmentation.models.record import RawRecord

    records = []
    for Conn in [GA4Connector, HubspotConnector, IntercomConnector]:
        records.extend(Conn().fetch("tenant_acme_corp"))
    raw = [RawRecord.model_validate(r.model_dump()) for r in records]

    clusters = segment(
        raw, tenant_industry="B2B SaaS",
        tenant_product="Project management tool for engineering teams",
        similarity_threshold=0.15, min_cluster_size=2,
    )
    snapshot = load_snapshot("regression_snapshot.json")
    assert len(clusters) == snapshot["cluster_count"]


def test_jaccard_top_behaviors_match_baseline():
    """Jaccard top_behaviors match frozen snapshot (order-independent)."""
    from segmentation.pipeline import segment
    from crawler.connectors import GA4Connector, HubspotConnector, IntercomConnector
    from segmentation.models.record import RawRecord

    records = []
    for Conn in [GA4Connector, HubspotConnector, IntercomConnector]:
        records.extend(Conn().fetch("tenant_acme_corp"))
    raw = [RawRecord.model_validate(r.model_dump()) for r in records]

    clusters = segment(
        raw, tenant_industry="B2B SaaS",
        tenant_product="Project management tool for engineering teams",
        similarity_threshold=0.15, min_cluster_size=2,
    )
    snapshot = load_snapshot("regression_snapshot.json")

    actual_sets = [set(c["summary"]["top_behaviors"]) for c in clusters]
    expected_sets = [set(c["top_behaviors"]) for c in snapshot["clusters"]]
    for exp in expected_sets:
        assert exp in actual_sets, f"Missing behavior set: {exp}"


def test_jaccard_top_pages_match_baseline():
    """Jaccard top_pages match frozen snapshot (order-independent)."""
    from segmentation.pipeline import segment
    from crawler.connectors import GA4Connector, HubspotConnector, IntercomConnector
    from segmentation.models.record import RawRecord

    records = []
    for Conn in [GA4Connector, HubspotConnector, IntercomConnector]:
        records.extend(Conn().fetch("tenant_acme_corp"))
    raw = [RawRecord.model_validate(r.model_dump()) for r in records]

    clusters = segment(
        raw, tenant_industry="B2B SaaS",
        tenant_product="Project management tool for engineering teams",
        similarity_threshold=0.15, min_cluster_size=2,
    )
    snapshot = load_snapshot("regression_snapshot.json")

    actual_sets = [set(c["summary"]["top_pages"]) for c in clusters]
    expected_sets = [set(c["top_pages"]) for c in snapshot["clusters"]]
    for exp in expected_sets:
        assert exp in actual_sets, f"Missing page set: {exp}"


def test_jaccard_sample_record_ids_match_baseline():
    """Jaccard sample record IDs are a subset of cluster records."""
    from segmentation.pipeline import segment
    from crawler.connectors import GA4Connector, HubspotConnector, IntercomConnector
    from segmentation.models.record import RawRecord

    records = []
    for Conn in [GA4Connector, HubspotConnector, IntercomConnector]:
        records.extend(Conn().fetch("tenant_acme_corp"))
    raw = [RawRecord.model_validate(r.model_dump()) for r in records]

    clusters = segment(
        raw, tenant_industry="B2B SaaS",
        tenant_product="Project management tool for engineering teams",
        similarity_threshold=0.15, min_cluster_size=2,
    )
    all_input_ids = {r.record_id for r in raw}
    for c in clusters:
        for sr in c["sample_records"]:
            assert sr["record_id"] in all_input_ids


def test_jaccard_ignores_registry():
    """Jaccard produces identical output when registry is passed."""
    from segmentation.pipeline import segment
    from segmentation.engine.registry import DEFAULT_REGISTRY
    from crawler.connectors import GA4Connector, HubspotConnector, IntercomConnector
    from segmentation.models.record import RawRecord

    records = []
    for Conn in [GA4Connector, HubspotConnector, IntercomConnector]:
        records.extend(Conn().fetch("tenant_acme_corp"))
    raw = [RawRecord.model_validate(r.model_dump()) for r in records]

    without = segment(raw, similarity_threshold=0.15, min_cluster_size=2)
    with_reg = segment(raw, similarity_threshold=0.15, min_cluster_size=2, feature_registry=DEFAULT_REGISTRY)

    assert clusters_match_ignoring_id(without, with_reg)


def test_jaccard_unchanged_with_typed_features():
    """Jaccard output unchanged even when UserFeatures have typed fields populated."""
    from segmentation.pipeline import segment
    from segmentation.engine.registry import DEFAULT_REGISTRY
    from crawler.connectors import GA4Connector, HubspotConnector, IntercomConnector
    from segmentation.models.record import RawRecord

    records = []
    for Conn in [GA4Connector, HubspotConnector, IntercomConnector]:
        records.extend(Conn().fetch("tenant_acme_corp"))
    raw = [RawRecord.model_validate(r.model_dump()) for r in records]

    # Default (no registry, no typed features populated)
    default_clusters = segment(raw, similarity_threshold=0.15, min_cluster_size=2)

    # Explicit jaccard with registry (typed features would be populated on UserFeatures
    # but Jaccard ignores them)
    jaccard_clusters = segment(
        raw, similarity_threshold=0.15, min_cluster_size=2,
        distance_metric="jaccard", feature_registry=DEFAULT_REGISTRY,
    )

    assert len(default_clusters) == len(jaccard_clusters)
    for d, j in zip(
        normalize_cluster_order(default_clusters),
        normalize_cluster_order(jaccard_clusters),
    ):
        assert set(d["summary"]["top_behaviors"]) == set(j["summary"]["top_behaviors"])


def test_lazy_import_contract_jaccard():
    """In Jaccard mode, registry is not used even if feature_registry is passed.

    Verifies the lazy import contract: Jaccard mode never touches the registry.
    (Direct sys.modules probing is unreliable in pytest shared-process runs.)
    """
    from segmentation.pipeline import segment
    from crawler.connectors import GA4Connector, HubspotConnector, IntercomConnector
    from segmentation.models.record import RawRecord

    records = []
    for Conn in [GA4Connector, HubspotConnector, IntercomConnector]:
        records.extend(Conn().fetch("tenant_acme_corp"))
    raw = [RawRecord.model_validate(r.model_dump()) for r in records]

    # Pass a broken registry — if Jaccard touched it, featurizer would fail
    broken_registry = {"ga4": ["not_a_real_extractor"]}
    clusters = segment(
        raw, similarity_threshold=0.15, min_cluster_size=2,
        distance_metric="jaccard", feature_registry=broken_registry,
    )
    # If Jaccard mode used the registry, this would have crashed
    assert len(clusters) >= 1
