"""Task 3: Adaptive featurizer — unit tests."""

from __future__ import annotations

from segmentation.models.record import RawRecord


def _make_record(source="ga4", user_id="u1", behaviors=None, pages=None, payload=None):
    return RawRecord(
        record_id=f"{source}_{user_id}_{id(payload)}",
        tenant_id="t1",
        source=source,
        timestamp="2026-04-01T12:00:00Z",
        user_id=user_id,
        behaviors=behaviors or [],
        pages=pages or [],
        payload=payload or {},
    )


def test_no_registry_empty_typed_features():
    """Without registry, typed feature dicts remain empty."""
    from segmentation.engine.featurizer import featurize_records

    records = [_make_record(payload={"session_duration": 2340})]
    features = featurize_records(records)
    assert features[0].numeric_features == {}
    assert features[0].categorical_features == {}
    assert features[0].set_features == {}


def test_registry_none_identical():
    """Explicit registry=None behaves identically to no registry."""
    from segmentation.engine.featurizer import featurize_records

    records = [_make_record(payload={"session_duration": 2340})]
    features = featurize_records(records, registry=None)
    assert features[0].numeric_features == {}


def test_ga4_numeric_mean():
    """GA4 session_duration extracted as numeric, averaged across records."""
    from segmentation.engine.featurizer import featurize_records
    from segmentation.engine.registry import DEFAULT_REGISTRY

    records = [
        _make_record(source="ga4", user_id="u1", behaviors=["a"], payload={"session_duration": 2000}),
        _make_record(source="ga4", user_id="u1", behaviors=["b"], payload={"session_duration": 1000}),
    ]
    features = featurize_records(records, registry=DEFAULT_REGISTRY)
    assert len(features) == 1
    assert features[0].numeric_features["session_duration"] == 1500.0


def test_hubspot_categorical_extraction():
    """HubSpot fields extracted as categorical features with correct names."""
    from segmentation.engine.featurizer import featurize_records
    from segmentation.engine.registry import DEFAULT_REGISTRY

    records = [
        _make_record(
            source="hubspot", user_id="u1", behaviors=["technical_role"],
            payload={"company_size": "50-200", "industry": "fintech", "contact_title": "Engineer"},
        ),
    ]
    features = featurize_records(records, registry=DEFAULT_REGISTRY)
    cats = features[0].categorical_features
    assert "company_size" in cats
    assert "industry" in cats
    assert "role" in cats  # contact_title mapped to feature_name="role"


def test_intercom_topic_no_message():
    """Intercom topic extracted; message field NOT extracted (not in registry)."""
    from segmentation.engine.featurizer import featurize_records
    from segmentation.engine.registry import DEFAULT_REGISTRY

    records = [
        _make_record(
            source="intercom", user_id="u1", behaviors=["question"],
            payload={"topic": "api_feedback", "message": "some text here"},
        ),
    ]
    features = featurize_records(records, registry=DEFAULT_REGISTRY)
    assert features[0].categorical_features["topic"] == "api_feedback"
    assert "message" not in features[0].categorical_features
    assert "message" not in features[0].numeric_features


def test_multi_source_user():
    """User with GA4 + HubSpot records gets both numeric and categorical features."""
    from segmentation.engine.featurizer import featurize_records
    from segmentation.engine.registry import DEFAULT_REGISTRY

    records = [
        _make_record(source="ga4", user_id="u1", behaviors=["api"], payload={"session_duration": 2340}),
        _make_record(source="hubspot", user_id="u1", behaviors=["tech"], payload={"industry": "fintech", "company_size": "50", "contact_title": "Eng"}),
    ]
    features = featurize_records(records, registry=DEFAULT_REGISTRY)
    assert len(features) == 1
    assert "session_duration" in features[0].numeric_features
    assert "industry" in features[0].categorical_features


def test_missing_payload_key():
    """Record with empty payload for ga4 source — no crash, no numeric features."""
    from segmentation.engine.featurizer import featurize_records
    from segmentation.engine.registry import DEFAULT_REGISTRY

    records = [_make_record(source="ga4", user_id="u1", behaviors=["a"], payload={})]
    features = featurize_records(records, registry=DEFAULT_REGISTRY)
    assert features[0].numeric_features == {}


def test_categorical_mode():
    """Most common categorical value wins."""
    from segmentation.engine.featurizer import featurize_records
    from segmentation.engine.registry import DEFAULT_REGISTRY

    records = [
        _make_record(source="hubspot", user_id="u1", behaviors=["r"], payload={"industry": "fintech", "company_size": "1", "contact_title": "E"}),
        _make_record(source="hubspot", user_id="u1", behaviors=["r"], payload={"industry": "saas", "company_size": "1", "contact_title": "E"}),
        _make_record(source="hubspot", user_id="u1", behaviors=["r"], payload={"industry": "fintech", "company_size": "1", "contact_title": "E"}),
    ]
    features = featurize_records(records, registry=DEFAULT_REGISTRY)
    assert features[0].categorical_features["industry"] == "fintech"


def test_categorical_tiebreak_first_seen():
    """On tie, first seen value in input order wins."""
    from segmentation.engine.featurizer import featurize_records
    from segmentation.engine.registry import DEFAULT_REGISTRY

    records = [
        _make_record(source="hubspot", user_id="u1", behaviors=["r"], payload={"industry": "fintech", "company_size": "1", "contact_title": "E"}),
        _make_record(source="hubspot", user_id="u1", behaviors=["r"], payload={"industry": "saas", "company_size": "1", "contact_title": "E"}),
    ]
    features = featurize_records(records, registry=DEFAULT_REGISTRY)
    assert features[0].categorical_features["industry"] == "fintech"


def test_anonymous_user_gets_typed_features():
    """Records without user_id still get typed features via anon_ key."""
    from segmentation.engine.featurizer import featurize_records
    from segmentation.engine.registry import DEFAULT_REGISTRY

    records = [
        _make_record(source="ga4", user_id=None, behaviors=["a"], payload={"session_duration": 100}),
    ]
    # Override user_id to None
    records[0].user_id = None
    features = featurize_records(records, registry=DEFAULT_REGISTRY)
    assert len(features) == 1
    assert features[0].numeric_features.get("session_duration") == 100.0


def test_cold_start_unknown_source():
    """Source not in registry — only behaviors/pages extracted, no typed features, no error."""
    from segmentation.engine.featurizer import featurize_records
    from segmentation.engine.registry import DEFAULT_REGISTRY

    records = [
        _make_record(source="salesforce", user_id="u1", behaviors=["deal_closed"],
                     payload={"deal_value": 50000, "stage": "closed_won"}),
    ]
    features = featurize_records(records, registry=DEFAULT_REGISTRY)
    assert features[0].behaviors == {"deal_closed"}
    assert features[0].numeric_features == {}
    assert features[0].categorical_features == {}


def test_non_parseable_numeric_skipped():
    """Non-float session_duration value is silently skipped."""
    from segmentation.engine.featurizer import featurize_records
    from segmentation.engine.registry import DEFAULT_REGISTRY

    records = [
        _make_record(source="ga4", user_id="u1", behaviors=["a"], payload={"session_duration": "N/A"}),
    ]
    features = featurize_records(records, registry=DEFAULT_REGISTRY)
    assert features[0].numeric_features == {}


def test_normalize_raises_skips_value():
    """Broken normalize skips that value, other features still extracted."""
    from segmentation.engine.registry import FeatureExtractor, get_extractors
    from segmentation.engine.featurizer import featurize_records
    from segmentation.models.features import FeatureType

    broken_registry = {
        "test_src": [
            FeatureExtractor("good_field", "good", FeatureType.CATEGORICAL),
            FeatureExtractor("bad_field", "bad", FeatureType.CATEGORICAL, normalize=lambda x: x.no_such_method()),
        ],
    }
    records = [
        _make_record(source="test_src", user_id="u1", behaviors=["a"],
                     payload={"good_field": "value", "bad_field": "anything"}),
    ]
    features = featurize_records(records, registry=broken_registry)
    assert features[0].categorical_features.get("good") == "value"
    assert "bad" not in features[0].categorical_features
