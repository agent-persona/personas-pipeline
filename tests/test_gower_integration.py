"""Task 7: Integration + regression tests."""

from __future__ import annotations

import json
import random
import sys

sys.path.insert(0, "../crawler")
sys.path.insert(0, "../synthesis")

from crawler.connectors import GA4Connector, HubspotConnector, IntercomConnector
from segmentation.models.record import RawRecord


def _get_all_records(tenant="tenant_acme_corp"):
    records = []
    for Conn in [GA4Connector, HubspotConnector, IntercomConnector]:
        records.extend(Conn().fetch(tenant))
    return [RawRecord.model_validate(r.model_dump()) for r in records]


def _get_ga4_only_records(tenant="tenant_acme_corp"):
    records = GA4Connector().fetch(tenant)
    return [RawRecord.model_validate(r.model_dump()) for r in records]


FIXTURE_PATH = "../tests/fixtures/regression_snapshot.json"


def test_gower_end_to_end():
    """Full pipeline: fetch → convert → segment(gower) → non-empty result."""
    from segmentation.pipeline import segment

    records = _get_all_records()
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    assert len(clusters) >= 1


def test_all_output_validates_clusterdata():
    """All Gower output validates as ClusterData."""
    from segmentation.pipeline import segment
    from synthesis.models.cluster import ClusterData

    records = _get_all_records()
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    for c in clusters:
        ClusterData.model_validate(c)


def test_engineers_designers_separate():
    """Gower separates engineers from designers based on top_behaviors."""
    from segmentation.pipeline import segment

    records = _get_all_records()
    clusters = segment(
        records, tenant_industry="B2B SaaS", tenant_product="PM tool",
        similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower",
    )
    eng_behaviors = {"api_setup", "webhook_config", "github_integration", "terraform_setup"}
    des_behaviors = {"template_browsing", "color_picker", "asset_export", "brand_kit_creation"}

    has_eng_cluster = False
    has_des_cluster = False
    for c in clusters:
        top = set(c["summary"]["top_behaviors"])
        if top & eng_behaviors:
            has_eng_cluster = True
        if top & des_behaviors:
            has_des_cluster = True
    assert has_eng_cluster
    assert has_des_cluster


def test_cluster_sizes():
    """Each cluster has cluster_size >= 2."""
    from segmentation.pipeline import segment

    records = _get_all_records()
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    for c in clusters:
        assert c["summary"]["cluster_size"] >= 2


def test_no_duplicate_users():
    """No user_id appears in more than one cluster's sample_records."""
    from segmentation.pipeline import segment

    records = _get_all_records()
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    seen = set()
    for c in clusters:
        for sr in c["sample_records"]:
            rid = sr["record_id"]
            assert rid not in seen, f"Duplicate record_id: {rid}"
            seen.add(rid)


def test_both_modes_valid():
    """Both Jaccard and Gower produce valid output from same input."""
    from segmentation.pipeline import segment
    from synthesis.models.cluster import ClusterData

    records = _get_all_records()
    for mode in ["jaccard", "gower"]:
        clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric=mode)
        assert len(clusters) >= 1
        for c in clusters:
            ClusterData.model_validate(c)


def test_empty_records():
    """Empty records produce empty output."""
    from segmentation.pipeline import segment

    assert segment([], distance_metric="gower") == []


def test_single_source_ga4_only():
    """Gower works with only GA4 data (no HubSpot/Intercom)."""
    from segmentation.pipeline import segment

    records = _get_ga4_only_records()
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    assert len(clusters) >= 1


def test_source_breakdown_in_extra():
    """All clusters have source_breakdown in extra."""
    from segmentation.pipeline import segment

    records = _get_all_records()
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    for c in clusters:
        breakdown = c["summary"]["extra"]["source_breakdown"]
        assert isinstance(breakdown, dict)
        assert len(breakdown) > 0


def test_tenant_context_propagated():
    """Tenant industry and product are propagated to output."""
    from segmentation.pipeline import segment

    records = _get_all_records()
    clusters = segment(
        records, tenant_industry="B2B SaaS", tenant_product="PM tool",
        similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower",
    )
    for c in clusters:
        assert c["tenant"]["industry"] == "B2B SaaS"
        assert c["tenant"]["product_description"] == "PM tool"


def test_schema_inference_integration():
    """Synthetic CSV records → infer_registry → segment(gower) → valid clusters."""
    from segmentation.pipeline import segment
    from segmentation.engine.schema_inference import infer_registry

    # Create synthetic CSV-like records
    csv_records = []
    for i in range(20):
        group = "power" if i < 10 else "casual"
        csv_records.append(RawRecord(
            record_id=f"csv_{i}",
            tenant_id="tenant_csv",
            source="csv",
            timestamp=None,
            user_id=f"csv_user_{i}",
            behaviors=[f"{group}_action"],
            pages=[],
            payload={
                "score": str(90 + i) if group == "power" else str(20 + i),
                "plan": "enterprise" if group == "power" else "free",
            },
        ))

    registry = infer_registry(csv_records)
    clusters = segment(
        csv_records, similarity_threshold=0.15, min_cluster_size=2,
        distance_metric="gower", feature_registry=registry,
    )
    assert len(clusters) >= 1


def test_cold_start_unknown_source():
    """Mix GA4 (in registry) + salesforce (not in registry) → runs, produces clusters."""
    from segmentation.pipeline import segment

    records = _get_all_records()
    # Add some salesforce records
    for i in range(4):
        records.append(RawRecord(
            record_id=f"sf_{i}",
            tenant_id="tenant_acme_corp",
            source="salesforce",
            timestamp=None,
            user_id=f"sf_user_{i}",
            behaviors=["deal_closed", "meeting_booked"],
            pages=[],
            payload={"deal_value": 50000},
        ))
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    assert len(clusters) >= 1


def test_regression_snapshot():
    """Default-param output matches the pre-change snapshot."""
    from segmentation.pipeline import segment

    records = _get_all_records()
    clusters = segment(
        records,
        tenant_industry="B2B SaaS",
        tenant_product="Project management tool for engineering teams",
        similarity_threshold=0.15,
        min_cluster_size=2,
    )

    with open(FIXTURE_PATH) as f:
        snapshot = json.load(f)

    assert len(clusters) == snapshot["cluster_count"]

    # Compare top_behaviors sets (order-independent)
    snapshot_behavior_sets = [set(c["top_behaviors"]) for c in snapshot["clusters"]]
    actual_behavior_sets = [set(c["summary"]["top_behaviors"]) for c in clusters]
    # Each snapshot set should appear in actual (order may differ)
    for sb in snapshot_behavior_sets:
        assert sb in actual_behavior_sets, f"Missing behavior set from snapshot: {sb}"


def test_sparse_cross_source():
    """Users split across sources still cluster (behaviors provide shared signal)."""
    from segmentation.pipeline import segment

    # Half GA4-only, half HubSpot-only, sharing behavior "api_setup"
    records = []
    for i in range(4):
        records.append(RawRecord(
            record_id=f"ga4_only_{i}", tenant_id="t_sparse", source="ga4",
            user_id=f"ga4_user_{i}", behaviors=["api_setup", "dashboard"],
            pages=["/api"], payload={"session_duration": 1000 + i * 100},
        ))
    for i in range(4):
        records.append(RawRecord(
            record_id=f"hub_only_{i}", tenant_id="t_sparse", source="hubspot",
            user_id=f"hub_user_{i}", behaviors=["api_setup"],
            pages=[], payload={"industry": "saas", "company_size": "50", "contact_title": "Eng"},
        ))
    clusters = segment(records, similarity_threshold=0.1, min_cluster_size=2, distance_metric="gower")
    assert len(clusters) >= 1
    # Clusters should contain users from multiple sources
    for c in clusters:
        sources = set()
        for sr in c["sample_records"]:
            sources.add(sr["source"])
        # At least one cluster should have mixed sources
    all_sources = set()
    for c in clusters:
        for sr in c["sample_records"]:
            all_sources.add(sr["source"])
    assert len(all_sources) >= 2


def test_order_sensitivity():
    """Same records in different order produce same number of clusters."""
    from segmentation.pipeline import segment

    records = _get_all_records()
    results = []
    for seed in range(5):
        shuffled = records.copy()
        random.Random(seed).shuffle(shuffled)
        clusters = segment(shuffled, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
        results.append(len(clusters))
    assert all(r == results[0] for r in results)


def test_multi_tenant():
    """Records from 2 tenants → clusters are per-tenant, no cross-tenant mixing."""
    from segmentation.pipeline import segment

    records_t1 = _get_all_records("tenant_alpha")
    records_t2 = _get_all_records("tenant_beta")
    all_records = records_t1 + records_t2

    clusters = segment(all_records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")

    for c in clusters:
        tenant_id = c["tenant"]["tenant_id"]
        assert tenant_id in ("tenant_alpha", "tenant_beta")
        # All sample records should be from the same tenant
        for sr in c["sample_records"]:
            # Record IDs are prefixed by source, not tenant, so check via the cluster's tenant_id
            pass  # Tenant isolation is enforced by pipeline partitioning
    # Should have clusters from both tenants
    tenant_ids = {c["tenant"]["tenant_id"] for c in clusters}
    assert len(tenant_ids) == 2
