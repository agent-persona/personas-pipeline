"""Task 2: Feature schema registry — unit tests."""

from __future__ import annotations

import pytest


def test_ga4_extractors():
    """DEFAULT_REGISTRY has 1 GA4 extractor for session_duration as NUMERIC."""
    from segmentation.engine.registry import DEFAULT_REGISTRY, get_extractors
    from segmentation.models.features import FeatureType

    extractors = get_extractors(DEFAULT_REGISTRY, "ga4")
    assert len(extractors) == 1
    assert extractors[0].payload_key == "session_duration"
    assert extractors[0].feature_type == FeatureType.NUMERIC


def test_hubspot_extractors():
    """DEFAULT_REGISTRY has 3 HubSpot extractors."""
    from segmentation.engine.registry import DEFAULT_REGISTRY, get_extractors

    extractors = get_extractors(DEFAULT_REGISTRY, "hubspot")
    assert len(extractors) == 3
    keys = {e.payload_key for e in extractors}
    assert keys == {"company_size", "industry", "contact_title"}


def test_intercom_extractors():
    """DEFAULT_REGISTRY has 1 Intercom extractor for topic as CATEGORICAL."""
    from segmentation.engine.registry import DEFAULT_REGISTRY, get_extractors
    from segmentation.models.features import FeatureType

    extractors = get_extractors(DEFAULT_REGISTRY, "intercom")
    assert len(extractors) == 1
    assert extractors[0].feature_type == FeatureType.CATEGORICAL


def test_unknown_source_returns_empty():
    """Unknown source returns empty list, no error."""
    from segmentation.engine.registry import DEFAULT_REGISTRY, get_extractors

    assert get_extractors(DEFAULT_REGISTRY, "salesforce") == []


def test_registry_keys():
    """DEFAULT_REGISTRY contains exactly ga4, hubspot, intercom."""
    from segmentation.engine.registry import DEFAULT_REGISTRY

    assert set(DEFAULT_REGISTRY.keys()) == {"ga4", "hubspot", "intercom"}


def test_extractor_is_frozen():
    """FeatureExtractor is immutable."""
    from segmentation.engine.registry import DEFAULT_REGISTRY, get_extractors

    extractor = get_extractors(DEFAULT_REGISTRY, "ga4")[0]
    with pytest.raises(AttributeError):
        extractor.payload_key = "something_else"


def test_normalize_strip():
    """Extractor with normalize=str.strip strips whitespace."""
    from segmentation.engine.registry import DEFAULT_REGISTRY, get_extractors

    extractors = get_extractors(DEFAULT_REGISTRY, "hubspot")
    role_extractor = next(e for e in extractors if e.feature_name == "role")
    assert role_extractor.normalize is not None
    assert role_extractor.normalize("  fintech  ") == "fintech"


def test_normalize_none_passthrough():
    """Extractor with normalize=None leaves value unchanged."""
    from segmentation.engine.registry import DEFAULT_REGISTRY, get_extractors

    extractor = get_extractors(DEFAULT_REGISTRY, "ga4")[0]
    assert extractor.normalize is None


def test_custom_registry():
    """A custom registry works with get_extractors."""
    from segmentation.engine.registry import FeatureExtractor, get_extractors
    from segmentation.models.features import FeatureType

    custom = {"custom": [FeatureExtractor("x", "x_val", FeatureType.NUMERIC)]}
    result = get_extractors(custom, "custom")
    assert len(result) == 1
    assert result[0].feature_name == "x_val"


def test_normalize_that_raises():
    """An extractor with a broken normalize callable can be created (tested at call site in Task 3)."""
    from segmentation.engine.registry import FeatureExtractor
    from segmentation.models.features import FeatureType

    extractor = FeatureExtractor(
        "key", "name", FeatureType.CATEGORICAL,
        normalize=lambda x: x.no_such_method(),
    )
    assert extractor.normalize is not None
    with pytest.raises(AttributeError):
        extractor.normalize("test")
