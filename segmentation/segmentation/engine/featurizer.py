from __future__ import annotations

from collections import defaultdict

from segmentation.models.features import UserFeatures
from segmentation.models.record import RawRecord


def featurize_records(records: list[RawRecord]) -> list[UserFeatures]:
    """Group records by (tenant_id, user_id) and aggregate into UserFeatures.

    Records without a user_id are treated as anonymous singletons keyed by
    record_id, so they still participate in clustering.
    """
    grouped: dict[tuple[str, str], list[RawRecord]] = defaultdict(list)
    for r in records:
        key = (r.tenant_id, r.user_id or f"anon_{r.record_id}")
        grouped[key].append(r)

    features: list[UserFeatures] = []
    for (tenant_id, user_id), user_records in grouped.items():
        behaviors: set[str] = set()
        pages: set[str] = set()
        sources: set[str] = set()
        record_ids: list[str] = []
        for r in user_records:
            behaviors.update(r.behaviors)
            pages.update(r.pages)
            sources.add(r.source)
            record_ids.append(r.record_id)
        features.append(
            UserFeatures(
                user_id=user_id,
                tenant_id=tenant_id,
                behaviors=behaviors,
                pages=pages,
                sources=sources,
                record_ids=record_ids,
            )
        )
    return features
