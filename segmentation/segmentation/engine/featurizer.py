from __future__ import annotations

from collections import Counter, defaultdict

from segmentation.models.features import FeatureType, UserFeatures
from segmentation.models.record import RawRecord


def featurize_records(
    records: list[RawRecord],
    registry: dict[str, list] | None = None,
) -> list[UserFeatures]:
    """Group records by (tenant_id, user_id) and aggregate into UserFeatures.

    Records without a user_id are treated as anonymous singletons keyed by
    record_id, so they still participate in clustering.

    When a registry is provided, typed features are extracted from record
    payloads. Without a registry, behavior is identical to the original code.
    """
    from segmentation.engine.registry import get_extractors

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

        # Typed feature extraction (only when registry provided)
        numeric_features: dict[str, float] = {}
        categorical_features: dict[str, str] = {}
        set_features: dict[str, set[str]] = {}

        if registry is not None:
            # Collect raw values per feature_name
            numeric_raw: dict[str, list[float]] = defaultdict(list)
            categorical_raw: dict[str, list[str]] = defaultdict(list)
            set_raw: dict[str, set[str]] = defaultdict(set)

            for r in user_records:
                extractors = get_extractors(registry, r.source)
                for ext in extractors:
                    raw_value = r.payload.get(ext.payload_key)
                    if raw_value is None:
                        continue

                    # Apply normalize (D5: before aggregation)
                    value = raw_value
                    if ext.normalize is not None:
                        try:
                            value = ext.normalize(value)
                        except Exception:
                            continue  # Skip this value

                    if ext.feature_type == FeatureType.NUMERIC:
                        try:
                            numeric_raw[ext.feature_name].append(float(value))
                        except (ValueError, TypeError):
                            continue  # Non-parseable numeric skipped
                    elif ext.feature_type == FeatureType.CATEGORICAL:
                        categorical_raw[ext.feature_name].append(str(value))
                    elif ext.feature_type == FeatureType.SET:
                        if isinstance(value, list):
                            set_raw[ext.feature_name].update(value)
                        else:
                            set_raw[ext.feature_name].add(str(value))

            # Aggregate: numeric → mean
            for name, values in numeric_raw.items():
                if values:
                    numeric_features[name] = sum(values) / len(values)

            # Aggregate: categorical → mode (first-seen tie-break)
            for name, values in categorical_raw.items():
                if values:
                    counts = Counter(values)
                    max_count = counts.most_common(1)[0][1]
                    # First seen with max count (preserves input order)
                    for v in values:
                        if counts[v] == max_count:
                            categorical_features[name] = v
                            break

            # Aggregate: set → union
            for name, value_set in set_raw.items():
                set_features[name] = value_set

        features.append(
            UserFeatures(
                user_id=user_id,
                tenant_id=tenant_id,
                behaviors=behaviors,
                pages=pages,
                sources=sources,
                record_ids=record_ids,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                set_features=set_features,
            )
        )
    return features
