"""CSV schema inference: auto-detect column types from arbitrary uploads."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from segmentation.engine.registry import FeatureExtractor, Registry
from segmentation.models.features import FeatureType
from segmentation.models.record import RawRecord

IDENTITY_FIELDS = {"user_id", "record_id", "timestamp", "id"}
MIN_RECORDS = 5


def _try_float(v) -> float | None:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


@dataclass
class InferredFeature:
    extractor: FeatureExtractor
    confidence: float


def infer_registry(
    records: list[RawRecord],
    source_name: str = "csv",
    confidence_threshold: float = 0.7,
) -> Registry:
    """Infer a feature registry from arbitrary record payloads.

    Returns {source_name: [extractors]} for features that pass the confidence
    threshold. Returns {} if fewer than MIN_RECORDS records.
    """
    if len(records) < MIN_RECORDS:
        return {}

    # Collect all values per payload key
    key_values: dict[str, list] = defaultdict(list)
    key_nulls: dict[str, int] = defaultdict(int)
    total = len(records)

    for r in records:
        seen_keys = set()
        for key, value in r.payload.items():
            seen_keys.add(key)
            if value is None:
                key_nulls[key] += 1
            else:
                key_values[key].append(value)
        # Keys not in this record count as null
        for key in key_values:
            if key not in seen_keys and key not in r.payload:
                key_nulls[key] += 1

    extractors: list[FeatureExtractor] = []

    for key, values in key_values.items():
        # Skip identity fields
        if key in IDENTITY_FIELDS:
            continue

        # Skip if >50% null
        null_count = key_nulls.get(key, 0)
        # Records that don't have the key at all also count as null
        present_count = len(values) + null_count
        missing_records = total - present_count
        effective_nulls = null_count + missing_records
        if effective_nulls > total * 0.5:
            continue

        non_null = values
        non_null_count = len(non_null)
        if non_null_count == 0:
            continue

        # Try NUMERIC: >80% parse as float
        float_successes = 0
        for v in non_null:
            try:
                float(v)
                float_successes += 1
            except (ValueError, TypeError):
                pass

        float_fraction = float_successes / non_null_count

        if float_fraction > 0.8:
            # Check for ID-like numerics: all unique, all integers, AND large values
            # (e.g., zip codes "94105", account numbers). Small scores like "85" are kept.
            unique_ratio = len(set(str(v) for v in non_null)) / non_null_count
            if unique_ratio > 0.9 and non_null_count >= 10:
                # Check if values look like IDs: all large integers (>999)
                large_ints = 0
                for v in non_null:
                    fv = _try_float(v)
                    if fv is not None and fv == int(fv) and abs(fv) > 999:
                        large_ints += 1
                if large_ints / non_null_count > 0.8:
                    continue  # Likely ID column

            confidence = float_fraction
            if confidence >= confidence_threshold:
                extractors.append(
                    FeatureExtractor(key, key, FeatureType.NUMERIC)
                )
            continue

        # Try CATEGORICAL: ≤20 unique values
        # Confidence = 1.0 - (unique_count / non_null_count)
        # Low cardinality = high confidence; high cardinality = low confidence
        unique_values = set(str(v) for v in non_null)
        unique_count = len(unique_values)

        if unique_count <= 20:
            confidence = 1.0 - (unique_count / non_null_count)
            if confidence >= confidence_threshold:
                extractors.append(
                    FeatureExtractor(key, key, FeatureType.CATEGORICAL)
                )
            continue

        # Otherwise: skip (high cardinality text)

    if not extractors:
        return {}

    return {source_name: extractors}
