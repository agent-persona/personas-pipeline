"""Tests for Experiment 6.13: Persona overlap heatmap.

Verifies:
1. Similarity primitives (Jaccard, list, text, struct)
2. Persona-pair similarity computation
3. Similarity matrix construction
4. Diagonal density metric
5. Per-field density breakdown
6. Max overlap pair detection
7. Edge cases (empty, single persona, identical personas)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "evals"))

from overlap_heatmap import (
    ALL_COMPARED_FIELDS,
    SimilarityMatrix,
    compute_similarity_matrix,
    diagonal_density,
    jaccard,
    list_field_similarity,
    max_overlap_pair,
    per_field_density,
    persona_similarity,
    struct_field_similarity,
    text_field_similarity,
)


# ── Fixtures ─────────────────────────────────────────────────────────

def _minimal_persona(
    name: str = "Test",
    goals: list[str] | None = None,
    pains: list[str] | None = None,
    vocabulary: list[str] | None = None,
    summary: str = "A test persona",
    demographics: dict | None = None,
    firmographics: dict | None = None,
) -> dict:
    return {
        "name": name,
        "summary": summary,
        "goals": goals or ["reduce costs", "increase efficiency"],
        "pains": pains or ["too many manual tasks", "slow processes"],
        "motivations": ["career growth"],
        "objections": ["too expensive"],
        "channels": ["slack", "email"],
        "vocabulary": vocabulary or ["agile", "sprint", "standup"],
        "decision_triggers": ["free trial"],
        "sample_quotes": ["I need something that just works"],
        "demographics": demographics or {"age_range": "25-34", "gender_distribution": "mixed"},
        "firmographics": firmographics or {"industry": "tech", "company_size": "SMB"},
    }


PERSONA_A = _minimal_persona(
    name="Alex the Engineer",
    goals=["automate deployments", "reduce downtime"],
    pains=["too many manual steps", "legacy systems"],
    vocabulary=["terraform", "CI/CD", "kubernetes"],
    summary="Infrastructure-focused engineer who automates everything",
    demographics={"age_range": "30-40", "education_level": "masters"},
    firmographics={"industry": "fintech", "company_size": "50-200"},
)

PERSONA_B = _minimal_persona(
    name="Maya the Designer",
    goals=["improve user experience", "ship designs faster"],
    pains=["unclear requirements", "too many revisions"],
    vocabulary=["figma", "prototyping", "user research"],
    summary="UX designer focused on speed and user empathy",
    demographics={"age_range": "25-34", "education_level": "bachelors"},
    firmographics={"industry": "e-commerce", "company_size": "10-50"},
)


# ── Jaccard tests ────────────────────────────────────────────────────

class TestJaccard:
    def test_identical(self):
        assert jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0

    def test_disjoint(self):
        assert jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial(self):
        assert jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(0.5)

    def test_both_empty(self):
        assert jaccard(set(), set()) == 0.0

    def test_one_empty(self):
        assert jaccard({"a"}, set()) == 0.0


# ── List field similarity tests ──────────────────────────────────────

class TestListFieldSimilarity:
    def test_identical_lists(self):
        sim = list_field_similarity(
            ["reduce costs", "increase speed"],
            ["reduce costs", "increase speed"],
        )
        assert sim == 1.0

    def test_disjoint_lists(self):
        sim = list_field_similarity(
            ["alpha beta gamma"],
            ["delta epsilon zeta"],
        )
        assert sim == 0.0

    def test_partial_overlap(self):
        sim = list_field_similarity(
            ["reduce manual work"],
            ["reduce automated work"],
        )
        assert 0.0 < sim < 1.0

    def test_empty_lists(self):
        assert list_field_similarity([], []) == 0.0


# ── Text field similarity tests ──────────────────────────────────────

class TestTextFieldSimilarity:
    def test_identical(self):
        assert text_field_similarity("hello world", "hello world") == 1.0

    def test_disjoint(self):
        assert text_field_similarity("alpha beta", "gamma delta") == 0.0

    def test_case_insensitive(self):
        assert text_field_similarity("Hello World", "hello world") == 1.0

    def test_empty(self):
        assert text_field_similarity("", "") == 0.0


# ── Struct field similarity tests ────────────────────────────────────

class TestStructFieldSimilarity:
    def test_identical(self):
        d = {"a": "foo", "b": "bar"}
        assert struct_field_similarity(d, d) == 1.0

    def test_disjoint_values(self):
        assert struct_field_similarity({"a": "x"}, {"a": "y"}) == 0.0

    def test_case_insensitive(self):
        assert struct_field_similarity({"a": "Foo"}, {"a": "foo"}) == 1.0

    def test_empty(self):
        assert struct_field_similarity({}, {}) == 0.0

    def test_list_subfields(self):
        sim = struct_field_similarity(
            {"roles": ["engineer", "manager"]},
            {"roles": ["engineer", "designer"]},
        )
        assert 0.0 < sim < 1.0  # partial overlap via Jaccard


# ── Persona similarity tests ────────────────────────────────────────

class TestPersonaSimilarity:
    def test_identical_personas(self):
        sim = persona_similarity(PERSONA_A, PERSONA_A)
        assert sim.overall == 1.0

    def test_different_personas_less_than_one(self):
        sim = persona_similarity(PERSONA_A, PERSONA_B)
        assert sim.overall < 1.0

    def test_different_personas_greater_than_zero(self):
        sim = persona_similarity(PERSONA_A, PERSONA_B)
        assert sim.overall > 0.0  # some structural overlap

    def test_per_field_populated(self):
        sim = persona_similarity(PERSONA_A, PERSONA_B)
        for f in ALL_COMPARED_FIELDS:
            assert f in sim.per_field

    def test_symmetric(self):
        sim_ab = persona_similarity(PERSONA_A, PERSONA_B)
        sim_ba = persona_similarity(PERSONA_B, PERSONA_A)
        assert sim_ab.overall == pytest.approx(sim_ba.overall, abs=1e-6)


# ── Similarity matrix tests ─────────────────────────────────────────

class TestSimilarityMatrix:
    def test_diagonal_is_one(self):
        matrix = compute_similarity_matrix([PERSONA_A, PERSONA_B])
        assert matrix.matrix[0][0] == 1.0
        assert matrix.matrix[1][1] == 1.0

    def test_symmetric(self):
        matrix = compute_similarity_matrix([PERSONA_A, PERSONA_B])
        assert matrix.matrix[0][1] == pytest.approx(matrix.matrix[1][0])

    def test_n_correct(self):
        matrix = compute_similarity_matrix([PERSONA_A, PERSONA_B])
        assert matrix.n == 2

    def test_names_extracted(self):
        matrix = compute_similarity_matrix([PERSONA_A, PERSONA_B])
        assert "Alex the Engineer" in matrix.names
        assert "Maya the Designer" in matrix.names

    def test_three_personas(self):
        p3 = _minimal_persona(name="Third", goals=["world peace"])
        matrix = compute_similarity_matrix([PERSONA_A, PERSONA_B, p3])
        assert matrix.n == 3
        assert len(matrix.matrix) == 3
        assert len(matrix.matrix[0]) == 3


# ── Diagonal density tests ──────────────────────────────────────────

class TestDiagonalDensity:
    def test_identical_personas_density_one(self):
        matrix = compute_similarity_matrix([PERSONA_A, PERSONA_A])
        assert diagonal_density(matrix) == 1.0

    def test_different_personas_density_less_than_one(self):
        matrix = compute_similarity_matrix([PERSONA_A, PERSONA_B])
        assert diagonal_density(matrix) < 1.0

    def test_single_persona_zero(self):
        matrix = compute_similarity_matrix([PERSONA_A])
        assert diagonal_density(matrix) == 0.0

    def test_empty_zero(self):
        matrix = SimilarityMatrix(names=[], matrix=[], per_field=[], n=0)
        assert diagonal_density(matrix) == 0.0


# ── Per-field density tests ──────────────────────────────────────────

class TestPerFieldDensity:
    def test_all_fields_present(self):
        matrix = compute_similarity_matrix([PERSONA_A, PERSONA_B])
        densities = per_field_density(matrix)
        for f in ALL_COMPARED_FIELDS:
            assert f in densities

    def test_identical_personas_all_one(self):
        matrix = compute_similarity_matrix([PERSONA_A, PERSONA_A])
        densities = per_field_density(matrix)
        for f, d in densities.items():
            assert d == pytest.approx(1.0, abs=1e-6), f"{f} should be 1.0"


# ── Max overlap pair tests ──────────────────────────────────────────

class TestMaxOverlapPair:
    def test_two_personas(self):
        matrix = compute_similarity_matrix([PERSONA_A, PERSONA_B])
        a, b, sim = max_overlap_pair(matrix)
        assert sim > 0
        assert {a, b} == {"Alex the Engineer", "Maya the Designer"}

    def test_three_personas_picks_closest(self):
        # A near-clone of A should be most similar to A (name differs slightly)
        clone = dict(PERSONA_A)
        clone["name"] = "Alex Clone"
        matrix = compute_similarity_matrix([PERSONA_A, PERSONA_B, clone])
        a, b, sim = max_overlap_pair(matrix)
        assert sim > 0.9  # near-clone, only name differs
        assert "Alex" in a and "Alex" in b

    def test_single_persona(self):
        matrix = compute_similarity_matrix([PERSONA_A])
        a, b, sim = max_overlap_pair(matrix)
        assert a == "" and b == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
