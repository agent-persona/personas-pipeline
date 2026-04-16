"""End-to-end segmentation pipeline: raw records in, cluster summaries out."""

from __future__ import annotations

from collections import defaultdict

from segmentation.engine.clusterer import cluster_users
from segmentation.engine.featurizer import featurize_records
from segmentation.engine.summarizer import build_cluster_data
from segmentation.models.record import RawRecord


def segment(
    records: list[RawRecord],
    *,
    tenant_industry: str | None = None,
    tenant_product: str | None = None,
    existing_persona_names: list[str] | None = None,
    similarity_threshold: float = 0.4,
    min_cluster_size: int = 2,
    distance_metric: str = "jaccard",
    feature_registry: dict | None = None,
    family_weights: dict[str, float] | None = None,
) -> list[dict]:
    """Run the full segmentation pipeline on a batch of records.

    Returns a list of cluster summary dicts, each one valid input for the
    synthesis service's `POST /synthesize` endpoint (after being loaded into
    its `ClusterData` Pydantic model).

    Records are partitioned by tenant_id; clustering is per-tenant.

    distance_metric: "jaccard" (default, original behavior) or "gower"
    feature_registry: typed feature extraction rules (Gower only; defaults to DEFAULT_REGISTRY)
    family_weights: per-family weights for Gower distance (default: equal)
    """
    # Lazy import of DEFAULT_REGISTRY (D10) — only when needed
    registry_to_use = None
    if distance_metric == "gower":
        from segmentation.engine.registry import DEFAULT_REGISTRY
        registry_to_use = feature_registry if feature_registry is not None else DEFAULT_REGISTRY

    by_tenant: dict[str, list[RawRecord]] = defaultdict(list)
    for r in records:
        by_tenant[r.tenant_id].append(r)

    cluster_summaries: list[dict] = []
    for tenant_id, tenant_records in by_tenant.items():
        features = featurize_records(tenant_records, registry=registry_to_use)
        clusters = cluster_users(
            features,
            threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
            distance_metric=distance_metric,
            family_weights=family_weights if distance_metric == "gower" else None,
        )
        for cluster in clusters:
            summary = build_cluster_data(
                cluster_users=cluster,
                all_records=tenant_records,
                tenant_id=tenant_id,
                tenant_industry=tenant_industry,
                tenant_product=tenant_product,
                existing_persona_names=existing_persona_names,
            )
            cluster_summaries.append(summary)

    return cluster_summaries
