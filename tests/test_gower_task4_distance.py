"""Task 4: Gower distance function — unit tests."""

from __future__ import annotations

import pytest
from segmentation.models.features import UserFeatures


def _uf(**kwargs):
    """Shorthand to create UserFeatures with defaults."""
    defaults = {"user_id": "u1", "tenant_id": "t1"}
    defaults.update(kwargs)
    return UserFeatures(**defaults)


def test_identical_users_distance_zero():
    """Two identical users have Gower distance 0.0."""
    from segmentation.engine.gower import gower_distance

    a = _uf(behaviors={"x", "y"}, numeric_features={"s": 10.0}, categorical_features={"r": "eng"})
    assert gower_distance(a, a, numeric_ranges={"s": 20.0}) == 0.0


def test_completely_different_behaviors():
    """Users with zero behavior overlap have distance 1.0."""
    from segmentation.engine.gower import gower_distance

    a = _uf(behaviors={"x"})
    b = _uf(behaviors={"y"})
    assert gower_distance(a, b) == 1.0


def test_numeric_distance():
    """Numeric distance is |a-b|/range."""
    from segmentation.engine.gower import gower_distance

    a = _uf(numeric_features={"x": 0.0})
    b = _uf(numeric_features={"x": 10.0})
    assert gower_distance(a, b, numeric_ranges={"x": 20.0}) == 0.5


def test_numeric_range_zero():
    """Range=0 means distance=0.0, not division by zero."""
    from segmentation.engine.gower import gower_distance

    a = _uf(numeric_features={"x": 5.0})
    b = _uf(numeric_features={"x": 5.0})
    assert gower_distance(a, b, numeric_ranges={"x": 0.0}) == 0.0


def test_categorical_match():
    """Matching categorical values have distance 0.0."""
    from segmentation.engine.gower import gower_distance

    a = _uf(categorical_features={"role": "eng"})
    b = _uf(categorical_features={"role": "eng"})
    assert gower_distance(a, b) == 0.0


def test_categorical_mismatch():
    """Non-matching categorical values have distance 1.0."""
    from segmentation.engine.gower import gower_distance

    a = _uf(categorical_features={"role": "eng"})
    b = _uf(categorical_features={"role": "designer"})
    assert gower_distance(a, b) == 1.0


def test_mixed_features_average():
    """Mixed features: average of behavior, numeric, and categorical distances."""
    from segmentation.engine.gower import gower_distance

    # behaviors: {a,b} vs {b,c} → jaccard = 1/3, dist = 2/3 ≈ 0.6667
    # numeric x: |0-5|/20 = 0.25
    # categorical role: match → 0.0
    a = _uf(behaviors={"a", "b"}, numeric_features={"x": 0.0}, categorical_features={"role": "eng"})
    b = _uf(behaviors={"b", "c"}, numeric_features={"x": 5.0}, categorical_features={"role": "eng"})
    dist = gower_distance(a, b, numeric_ranges={"x": 20.0})
    # (2/3 + 0.25 + 0.0) / 3 ≈ 0.3056
    assert abs(dist - (2 / 3 + 0.25 + 0.0) / 3) < 1e-9


def test_feature_only_in_one_user_excluded():
    """A numeric feature present in only one user is excluded from the average."""
    from segmentation.engine.gower import gower_distance

    a = _uf(numeric_features={"x": 1.0, "y": 2.0})
    b = _uf(numeric_features={"x": 1.0})
    # Only "x" compared → distance = 0.0
    assert gower_distance(a, b, numeric_ranges={"x": 10.0, "y": 10.0}) == 0.0


def test_no_features_overlap():
    """Users with no shared features have distance 1.0."""
    from segmentation.engine.gower import gower_distance

    a = _uf(numeric_features={"x": 1.0})
    b = _uf(categorical_features={"role": "eng"})
    assert gower_distance(a, b, numeric_ranges={"x": 10.0}) == 1.0


def test_gower_similarity_inverse():
    """gower_similarity == 1.0 - gower_distance for any inputs."""
    from segmentation.engine.gower import gower_distance, gower_similarity

    a = _uf(behaviors={"a"}, numeric_features={"x": 0.0})
    b = _uf(behaviors={"a", "b"}, numeric_features={"x": 5.0})
    ranges = {"x": 10.0}
    dist = gower_distance(a, b, numeric_ranges=ranges)
    sim = gower_similarity(a, b, numeric_ranges=ranges)
    assert abs(dist + sim - 1.0) < 1e-9


def test_compute_ranges_basic():
    """compute_ranges returns max-min per numeric feature."""
    from segmentation.engine.gower import compute_ranges

    users = [
        _uf(numeric_features={"x": 10.0}),
        _uf(user_id="u2", numeric_features={"x": 30.0}),
        _uf(user_id="u3", numeric_features={"x": 20.0}),
    ]
    ranges = compute_ranges(users)
    assert ranges["x"] == 20.0


def test_compute_ranges_identical_values():
    """All-identical values → range=0."""
    from segmentation.engine.gower import compute_ranges

    users = [
        _uf(numeric_features={"x": 5.0}),
        _uf(user_id="u2", numeric_features={"x": 5.0}),
    ]
    assert compute_ranges(users)["x"] == 0.0


def test_both_behaviors_empty_skipped():
    """Both users with empty behaviors → behavior dimension skipped, only categories compared."""
    from segmentation.engine.gower import gower_distance

    a = _uf(categorical_features={"role": "eng"})
    b = _uf(categorical_features={"role": "designer"})
    # No behaviors, no pages → only categorical dimension
    assert gower_distance(a, b) == 1.0


def test_numeric_ranges_none_skips_numerics():
    """When numeric_ranges is None, all numeric dimensions are skipped."""
    from segmentation.engine.gower import gower_distance

    a = _uf(behaviors={"a"}, numeric_features={"x": 0.0})
    b = _uf(behaviors={"a"}, numeric_features={"x": 100.0})
    # Numerics skipped → only behaviors compared → identical → distance 0.0
    assert gower_distance(a, b, numeric_ranges=None) == 0.0


def test_compute_ranges_single_user_feature():
    """Feature in only 1 user → range=0, key still in dict."""
    from segmentation.engine.gower import compute_ranges

    users = [
        _uf(numeric_features={"x": 10.0}),
        _uf(user_id="u2", numeric_features={}),  # no x
    ]
    ranges = compute_ranges(users)
    assert "x" in ranges
    assert ranges["x"] == 0.0


def test_asymmetric_behaviors_one_empty():
    """One user has behaviors, other doesn't → dimension included, distance=1.0."""
    from segmentation.engine.gower import gower_distance

    a = _uf(behaviors={"a"})
    b = _uf(behaviors=set())
    # jaccard({a}, {}) = 0 → dist = 1.0
    assert gower_distance(a, b) == 1.0


def test_family_weighting_custom():
    """Custom family weights change the weighted average."""
    from segmentation.engine.gower import gower_distance

    # 1 set dim (behaviors): dist = 1.0 (completely different)
    # 1 numeric dim: dist = 0.5
    # 1 categorical dim: dist = 0.0 (match)
    a = _uf(behaviors={"x"}, numeric_features={"n": 0.0}, categorical_features={"r": "eng"})
    b = _uf(behaviors={"y"}, numeric_features={"n": 10.0}, categorical_features={"r": "eng"})
    ranges = {"n": 20.0}
    weights = {"sets": 1.0, "numerics": 2.0, "categories": 1.0}
    dist = gower_distance(a, b, numeric_ranges=ranges, family_weights=weights)
    # (1.0*1.0 + 0.5*2.0 + 0.0*1.0) / (1.0+2.0+1.0) = 2.0/4.0 = 0.5
    assert abs(dist - 0.5) < 1e-9


def test_family_weighting_default_equal():
    """Default weights (None) give equal weight to all dimensions."""
    from segmentation.engine.gower import gower_distance

    a = _uf(behaviors={"x"}, numeric_features={"n": 0.0}, categorical_features={"r": "eng"})
    b = _uf(behaviors={"y"}, numeric_features={"n": 10.0}, categorical_features={"r": "eng"})
    ranges = {"n": 20.0}
    dist = gower_distance(a, b, numeric_ranges=ranges, family_weights=None)
    # (1.0 + 0.5 + 0.0) / 3 = 0.5
    assert abs(dist - 0.5) < 1e-9
