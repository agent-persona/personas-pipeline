"""Tests for experiment 5.11 — reference-based vs reference-free judging."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "evals"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from reference_judging import (
    build_free_prompt,
    build_reference_prompt,
    compare_modes,
    spearman_correlation,
    PROXY_REFERENCE_PERSONA,
    PROXY_EXPECTED_OVERALL,
)


SAMPLE_PERSONA = {
    "name": "Test Persona",
    "summary": "A test persona for judging.",
    "goals": ["Goal 1", "Goal 2"],
    "pains": ["Pain 1"],
}


def test_free_prompt_contains_persona():
    prompt = build_free_prompt(SAMPLE_PERSONA)
    assert "Test Persona" in prompt
    assert "PROXY REFERENCE" not in prompt


def test_reference_prompt_contains_both():
    prompt = build_reference_prompt(SAMPLE_PERSONA)
    assert "Test Persona" in prompt
    assert "PROXY REFERENCE" in prompt
    assert "4.5/5" in prompt


def test_proxy_is_clearly_labeled():
    assert "PROXY" in PROXY_REFERENCE_PERSONA["_meta"]
    assert "NOT" in PROXY_REFERENCE_PERSONA["_meta"]
    assert "gold" in PROXY_REFERENCE_PERSONA["_meta"].lower()


def test_compare_modes_basic():
    free = [3.0, 4.0, 3.5, 4.5, 2.5]
    ref = [3.8, 4.1, 3.9, 4.2, 3.7]
    result = compare_modes(free, ref)
    assert result.free_stats.n == 5
    assert result.ref_stats.n == 5
    # Reference mode has lower variance in this example
    assert result.ref_stats.std < result.free_stats.std
    assert result.variance_reduction_ratio > 0


def test_compare_modes_anchoring_detected():
    # All ref scores near 4.5 (the anchor)
    free = [2.0, 3.0, 4.0, 5.0, 3.5]
    ref = [4.3, 4.5, 4.6, 4.4, 4.7]
    result = compare_modes(free, ref)
    assert result.anchoring_detected is True


def test_compare_modes_no_anchoring():
    # Ref scores spread out, not near anchor
    free = [2.0, 3.0, 4.0, 5.0, 1.0]
    ref = [2.1, 3.1, 4.1, 5.1, 1.1]
    result = compare_modes(free, ref)
    assert result.anchoring_detected is False


def test_spearman_perfect_correlation():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert abs(spearman_correlation(x, y) - 1.0) < 0.001


def test_spearman_inverse_correlation():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert abs(spearman_correlation(x, y) - (-1.0)) < 0.001


def test_spearman_too_few():
    import math
    assert math.isnan(spearman_correlation([1.0, 2.0], [2.0, 1.0]))


def test_to_dict_format():
    free = [3.0, 4.0, 3.5]
    ref = [3.8, 4.1, 3.9]
    result = compare_modes(free, ref)
    d = result.to_dict()
    assert "free_mode" in d
    assert "reference_mode" in d
    assert "variance_reduction_ratio" in d
    assert "anchoring_detected" in d
    assert isinstance(d["free_mode"]["scores"], list)
