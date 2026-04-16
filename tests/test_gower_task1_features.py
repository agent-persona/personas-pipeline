"""Task 1: FeatureType enum + UserFeatures typed fields — unit tests."""

from __future__ import annotations

import pytest


def test_empty_defaults():
    """New typed feature fields default to empty dicts."""
    from segmentation.models.features import UserFeatures

    uf = UserFeatures(user_id="u1", tenant_id="t1")
    assert uf.numeric_features == {}
    assert uf.categorical_features == {}
    assert uf.set_features == {}


def test_numeric_features_roundtrip():
    """Numeric features survive model_dump/model_validate."""
    from segmentation.models.features import UserFeatures

    uf = UserFeatures(
        user_id="u1",
        tenant_id="t1",
        numeric_features={"session_duration": 2340.0},
    )
    assert uf.numeric_features["session_duration"] == 2340.0
    dumped = uf.model_dump()
    restored = UserFeatures.model_validate(dumped)
    assert restored.numeric_features["session_duration"] == 2340.0


def test_categorical_features_store():
    """Categorical features store multiple keys correctly."""
    from segmentation.models.features import UserFeatures

    uf = UserFeatures(
        user_id="u1",
        tenant_id="t1",
        categorical_features={"role": "eng", "industry": "fintech"},
    )
    assert uf.categorical_features["role"] == "eng"
    assert uf.categorical_features["industry"] == "fintech"
    assert len(uf.categorical_features) == 2


def test_set_features_store():
    """Set features store sets correctly."""
    from segmentation.models.features import UserFeatures

    uf = UserFeatures(
        user_id="u1",
        tenant_id="t1",
        set_features={"skills": {"python", "go"}},
    )
    assert uf.set_features["skills"] == {"python", "go"}


def test_feature_type_enum():
    """FeatureType enum has correct values and supports string construction."""
    from segmentation.models.features import FeatureType

    assert FeatureType.SET.value == "set"
    assert FeatureType.NUMERIC.value == "numeric"
    assert FeatureType.CATEGORICAL.value == "categorical"
    assert FeatureType("numeric") is FeatureType.NUMERIC


def test_backward_compat_original_fields_only():
    """UserFeatures with only original fields has new fields as empty dicts."""
    from segmentation.models.features import UserFeatures

    uf = UserFeatures(
        user_id="u1",
        tenant_id="t1",
        behaviors={"a"},
        pages={"/home"},
        sources={"ga4"},
        record_ids=["r1"],
    )
    dumped = uf.model_dump()
    assert dumped["numeric_features"] == {}
    assert dumped["categorical_features"] == {}
    assert dumped["set_features"] == {}
    restored = UserFeatures.model_validate(dumped)
    assert restored.numeric_features == {}


def test_model_dump_includes_numeric():
    """model_dump includes numeric_features when populated."""
    from segmentation.models.features import UserFeatures

    uf = UserFeatures(
        user_id="u1",
        tenant_id="t1",
        numeric_features={"x": 1.5},
    )
    dumped = uf.model_dump()
    assert "numeric_features" in dumped
    assert dumped["numeric_features"] == {"x": 1.5}


def test_behaviors_field_still_works():
    """Original behaviors field is unaffected by new fields."""
    from segmentation.models.features import UserFeatures

    uf = UserFeatures(user_id="u1", tenant_id="t1", behaviors={"a", "b"})
    assert uf.behaviors == {"a", "b"}


def test_feature_type_has_exactly_three_members():
    """FeatureType enum has exactly 3 members."""
    from segmentation.models.features import FeatureType

    assert len(FeatureType) == 3


def test_empty_set_stored():
    """An empty set in set_features is stored, not dropped."""
    from segmentation.models.features import UserFeatures

    uf = UserFeatures(
        user_id="u1",
        tenant_id="t1",
        set_features={"skills": set()},
    )
    assert "skills" in uf.set_features
    assert uf.set_features["skills"] == set()
