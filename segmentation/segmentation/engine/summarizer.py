from __future__ import annotations

import uuid
from collections import Counter

from segmentation.models.features import UserFeatures
from segmentation.models.record import RawRecord

# How many sample records to include in the cluster handoff to synthesis
DEFAULT_SAMPLE_SIZE = 12


def build_cluster_data(
    cluster_users: list[UserFeatures],
    all_records: list[RawRecord],
    tenant_id: str,
    tenant_industry: str | None = None,
    tenant_product: str | None = None,
    existing_persona_names: list[str] | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> dict:
    """Build a cluster summary in the exact JSON shape that synthesis expects.

    The output dict is the contract: it conforms to synthesis's `ClusterData`
    Pydantic model. This is the integration seam — segmentation produces it,
    synthesis consumes it. The two services are decoupled by this JSON shape.

    Returns a dict (not a Pydantic model) so segmentation has no runtime
    dependency on synthesis. Either side can validate against the shared
    schema independently.
    """
    cluster_user_ids = {u.user_id for u in cluster_users}
    cluster_record_ids: set[str] = set()
    for u in cluster_users:
        cluster_record_ids.update(u.record_ids)

    # Pull the raw records that belong to this cluster
    cluster_records = [r for r in all_records if r.record_id in cluster_record_ids]

    # Aggregate behaviors and pages across the whole cluster
    behavior_counter: Counter[str] = Counter()
    page_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    for r in cluster_records:
        behavior_counter.update(r.behaviors)
        page_counter.update(r.pages)
        source_counter[r.source] += 1

    top_behaviors = [b for b, _ in behavior_counter.most_common(8)]
    top_pages = [p for p, _ in page_counter.most_common(8)]

    # Pick representative sample records (one of each top behavior if possible)
    sample = _pick_representative_sample(cluster_records, top_behaviors, sample_size)

    cluster_id = f"clust_{uuid.uuid4().hex[:12]}"

    return {
        "cluster_id": cluster_id,
        "tenant": {
            "tenant_id": tenant_id,
            "industry": tenant_industry,
            "product_description": tenant_product,
            "existing_persona_names": existing_persona_names or [],
        },
        "summary": {
            "cluster_size": len(cluster_user_ids),
            "top_behaviors": top_behaviors,
            "top_pages": top_pages,
            "conversion_rate": None,
            "avg_session_duration_seconds": None,
            "top_referrers": [],
            "extra": {
                "n_records": len(cluster_records),
                "source_breakdown": dict(source_counter),
            },
        },
        "sample_records": [
            {
                "record_id": r.record_id,
                "source": r.source,
                "timestamp": r.timestamp,
                "payload": r.payload,
            }
            for r in sample
        ],
        "enrichment": {
            "firmographic": {},
            "intent_signals": [],
            "technographic": {},
            "extra": {},
        },
    }


def _pick_representative_sample(
    records: list[RawRecord],
    top_behaviors: list[str],
    sample_size: int,
) -> list[RawRecord]:
    """Pick a sample that covers the top behaviors before filling with random."""
    if len(records) <= sample_size:
        return records

    picked: list[RawRecord] = []
    picked_ids: set[str] = set()

    # First pass: pick one record per top behavior
    for behavior in top_behaviors:
        for r in records:
            if r.record_id in picked_ids:
                continue
            if behavior in r.behaviors:
                picked.append(r)
                picked_ids.add(r.record_id)
                break
        if len(picked) >= sample_size:
            return picked

    # Second pass: fill the rest in original order
    for r in records:
        if len(picked) >= sample_size:
            break
        if r.record_id not in picked_ids:
            picked.append(r)
            picked_ids.add(r.record_id)

    return picked
