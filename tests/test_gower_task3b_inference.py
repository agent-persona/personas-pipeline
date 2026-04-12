"""Task 3b: CSV schema inference — unit tests."""

from __future__ import annotations

from segmentation.models.record import RawRecord


def _csv_records(payloads, source="csv"):
    """Create RawRecords from a list of payload dicts."""
    return [
        RawRecord(
            record_id=f"csv_{i}",
            tenant_id="t1",
            source=source,
            timestamp=None,
            user_id=f"u{i}",
            behaviors=[],
            pages=[],
            payload=p,
        )
        for i, p in enumerate(payloads)
    ]


def test_numeric_inferred():
    """Column with all-float values is inferred as NUMERIC with confidence=1.0."""
    from segmentation.engine.schema_inference import infer_registry
    from segmentation.models.features import FeatureType

    records = _csv_records([{"score": "85"} for _ in range(10)])
    registry = infer_registry(records)
    extractors = registry.get("csv", [])
    score_ext = next(e for e in extractors if e.feature_name == "score")
    assert score_ext.feature_type == FeatureType.NUMERIC


def test_categorical_inferred():
    """Column with few unique values is inferred as CATEGORICAL."""
    from segmentation.engine.schema_inference import infer_registry
    from segmentation.models.features import FeatureType

    payloads = [{"plan": v} for v in ["pro", "free", "enterprise"] * 4][:10]
    records = _csv_records(payloads)
    registry = infer_registry(records)
    extractors = registry.get("csv", [])
    plan_ext = next(e for e in extractors if e.feature_name == "plan")
    assert plan_ext.feature_type == FeatureType.CATEGORICAL


def test_high_cardinality_skipped():
    """Column with all-unique values is skipped."""
    from segmentation.engine.schema_inference import infer_registry

    records = _csv_records([{"uuid": f"uuid-{i}"} for i in range(10)])
    registry = infer_registry(records)
    extractors = registry.get("csv", [])
    assert not any(e.feature_name == "uuid" for e in extractors)


def test_mostly_null_skipped():
    """Column with >50% null values is skipped."""
    from segmentation.engine.schema_inference import infer_registry

    payloads = [{"note": None}] * 8 + [{"note": "hello"}] * 2
    records = _csv_records(payloads)
    registry = infer_registry(records)
    extractors = registry.get("csv", [])
    assert not any(e.feature_name == "note" for e in extractors)


def test_mixed_payload():
    """Mixed columns: score=NUMERIC, plan=CATEGORICAL, name=skipped."""
    from segmentation.engine.schema_inference import infer_registry
    from segmentation.models.features import FeatureType

    payloads = [
        {"score": str(80 + i), "plan": ["pro", "free"][i % 2], "name": f"user_{i}"}
        for i in range(10)
    ]
    records = _csv_records(payloads)
    registry = infer_registry(records)
    extractors = registry.get("csv", [])
    names = {e.feature_name for e in extractors}
    assert "score" in names
    assert "plan" in names
    assert "name" not in names  # 10 unique in 10 records = 100% unique → skipped


def test_returns_valid_registry():
    """Result is a valid Registry passable to featurize_records."""
    from segmentation.engine.schema_inference import infer_registry
    from segmentation.engine.featurizer import featurize_records

    records = _csv_records([{"score": "85"} for _ in range(10)])
    registry = infer_registry(records)
    # Should not raise
    features = featurize_records(records, registry=registry)
    assert len(features) == 10


def test_empty_records():
    """Empty record list returns empty registry."""
    from segmentation.engine.schema_inference import infer_registry

    assert infer_registry([]) == {}


def test_below_minimum_records():
    """4 records (below minimum 5) returns empty. 5 records returns non-empty."""
    from segmentation.engine.schema_inference import infer_registry

    records_4 = _csv_records([{"score": "85"}] * 4)
    assert infer_registry(records_4) == {}

    records_5 = _csv_records([{"score": "85"}] * 5)
    assert len(infer_registry(records_5).get("csv", [])) > 0


def test_boolean_as_categorical():
    """Boolean-like values inferred as CATEGORICAL."""
    from segmentation.engine.schema_inference import infer_registry
    from segmentation.models.features import FeatureType

    payloads = [{"active": v} for v in (["true", "false"] * 5)]
    records = _csv_records(payloads)
    registry = infer_registry(records)
    extractors = registry.get("csv", [])
    active_ext = next(e for e in extractors if e.feature_name == "active")
    assert active_ext.feature_type == FeatureType.CATEGORICAL


def test_numeric_with_some_failures():
    """9 float + 1 N/A → NUMERIC (90% > 80% threshold)."""
    from segmentation.engine.schema_inference import infer_registry
    from segmentation.models.features import FeatureType

    payloads = [{"score": str(80 + i)} for i in range(9)] + [{"score": "N/A"}]
    records = _csv_records(payloads)
    registry = infer_registry(records)
    extractors = registry.get("csv", [])
    score_ext = next(e for e in extractors if e.feature_name == "score")
    assert score_ext.feature_type == FeatureType.NUMERIC


def test_identity_field_skipped():
    """Fields named user_id, record_id, timestamp, id are skipped."""
    from segmentation.engine.schema_inference import infer_registry

    payloads = [{"user_id": f"u{i}", "score": str(80 + i)} for i in range(10)]
    records = _csv_records(payloads)
    registry = infer_registry(records)
    extractors = registry.get("csv", [])
    assert not any(e.feature_name == "user_id" for e in extractors)
    assert any(e.feature_name == "score" for e in extractors)


def test_all_identical_categorical():
    """All-identical values → CATEGORICAL (1 unique, ≤20, <50% of 10)."""
    from segmentation.engine.schema_inference import infer_registry
    from segmentation.models.features import FeatureType

    records = _csv_records([{"status": "active"}] * 10)
    registry = infer_registry(records)
    extractors = registry.get("csv", [])
    status_ext = next(e for e in extractors if e.feature_name == "status")
    assert status_ext.feature_type == FeatureType.CATEGORICAL


def test_low_confidence_excluded():
    """Field with 12 unique in 20 records → confidence 0.4 → excluded at default threshold 0.7."""
    from segmentation.engine.schema_inference import infer_registry

    payloads = [{"tag": f"tag_{i % 12}"} for i in range(20)]
    records = _csv_records(payloads)
    registry = infer_registry(records)
    extractors = registry.get("csv", [])
    assert not any(e.feature_name == "tag" for e in extractors)


def test_confidence_threshold_zero_includes_all():
    """confidence_threshold=0.0 includes all classifiable features."""
    from segmentation.engine.schema_inference import infer_registry

    payloads = [{"tag": f"tag_{i % 12}"} for i in range(20)]
    records = _csv_records(payloads)
    registry = infer_registry(records, confidence_threshold=0.0)
    extractors = registry.get("csv", [])
    assert any(e.feature_name == "tag" for e in extractors)
