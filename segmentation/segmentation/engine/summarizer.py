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
            "avg_session_duration_seconds": _compute_avg_session_duration(cluster_users),
            "top_referrers": [],
            "extra": _build_extra(cluster_users, cluster_records, source_counter),
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


def _compute_avg_session_duration(users: list[UserFeatures]) -> float | None:
    """Compute average session duration from numeric features, or None."""
    durations = [
        u.numeric_features["session_duration"]
        for u in users
        if "session_duration" in u.numeric_features
    ]
    if not durations:
        return None
    return sum(durations) / len(durations)


def _build_extra(
    users: list[UserFeatures],
    cluster_records: list[RawRecord],
    source_counter: Counter,
) -> dict:
    """Build the extra dict with source breakdown and optional typed features."""
    extra: dict = {
        "n_records": len(cluster_records),
        "source_breakdown": dict(source_counter),
    }

    # Add typed feature aggregates if any user has them
    has_typed = any(
        u.numeric_features or u.categorical_features or u.set_features
        for u in users
    )
    if has_typed:
        # Numeric averages
        numeric_avgs: dict[str, float] = {}
        numeric_keys: set[str] = set()
        for u in users:
            numeric_keys.update(u.numeric_features.keys())
        for key in numeric_keys:
            values = [u.numeric_features[key] for u in users if key in u.numeric_features]
            if values:
                numeric_avgs[key] = sum(values) / len(values)

        # Categorical modes
        categorical_modes: dict[str, str] = {}
        cat_keys: set[str] = set()
        for u in users:
            cat_keys.update(u.categorical_features.keys())
        for key in cat_keys:
            values = [u.categorical_features[key] for u in users if key in u.categorical_features]
            if values:
                counts = Counter(values)
                categorical_modes[key] = counts.most_common(1)[0][0]

        # Set unions
        set_unions: dict[str, list[str]] = {}
        set_keys: set[str] = set()
        for u in users:
            set_keys.update(u.set_features.keys())
        for key in set_keys:
            union: set[str] = set()
            for u in users:
                if key in u.set_features:
                    union |= u.set_features[key]
            set_unions[key] = sorted(union)

        extra["typed_features"] = {
            "numeric_averages": numeric_avgs,
            "categorical_modes": categorical_modes,
            "set_unions": set_unions,
        }

    return extra


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
