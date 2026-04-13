"""Tests for Experiment 6.16: Vocabulary uniqueness (Jensen-Shannon divergence).

Tests validate the correctness of the JS divergence computation with known
inputs and expected outputs.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "evals"))

from vocab_divergence import (
    js_divergence,
    mean_pairwise_js_divergence,
    pairwise_js_matrix,
    vocab_overlap_stats,
)


# ── js_divergence unit tests ─────────────────────────────────────────


class TestJSDivergence:
    def test_identical_vocabularies(self):
        """JSD of identical distributions should be 0."""
        vocab = ["sprint", "backlog", "standup", "retro"]
        assert js_divergence(vocab, vocab) == pytest.approx(0.0)

    def test_completely_disjoint(self):
        """JSD of non-overlapping vocabularies should be 1.0 (max)."""
        vocab_a = ["sprint", "backlog", "standup"]
        vocab_b = ["kubernetes", "docker", "terraform"]
        assert js_divergence(vocab_a, vocab_b) == pytest.approx(1.0)

    def test_partial_overlap(self):
        """JSD with partial overlap should be between 0 and 1."""
        vocab_a = ["sprint", "backlog", "deploy", "api"]
        vocab_b = ["sprint", "backlog", "dashboard", "metrics"]
        result = js_divergence(vocab_a, vocab_b)
        assert 0.0 < result < 1.0

    def test_symmetry(self):
        """JSD should be symmetric: JSD(A, B) == JSD(B, A)."""
        vocab_a = ["sprint", "backlog", "deploy"]
        vocab_b = ["dashboard", "metrics", "deploy"]
        assert js_divergence(vocab_a, vocab_b) == pytest.approx(
            js_divergence(vocab_b, vocab_a)
        )

    def test_empty_both(self):
        """JSD of two empty lists should be 0."""
        assert js_divergence([], []) == 0.0

    def test_one_empty(self):
        """JSD when one side is empty should be 1.0 (max divergence)."""
        assert js_divergence(["sprint", "backlog"], []) == 1.0
        assert js_divergence([], ["sprint", "backlog"]) == 1.0

    def test_case_insensitive(self):
        """Terms should be compared case-insensitively."""
        vocab_a = ["Sprint", "BACKLOG", "Deploy"]
        vocab_b = ["sprint", "backlog", "deploy"]
        assert js_divergence(vocab_a, vocab_b) == pytest.approx(0.0)

    def test_with_duplicates(self):
        """Duplicate terms affect the frequency distribution."""
        # Same unique terms but different frequencies
        vocab_a = ["sprint", "sprint", "sprint", "backlog"]
        vocab_b = ["sprint", "backlog", "backlog", "backlog"]
        result = js_divergence(vocab_a, vocab_b)
        assert 0.0 < result < 1.0

    def test_single_term_each(self):
        """Single distinct terms → maximum divergence."""
        assert js_divergence(["alpha"], ["beta"]) == pytest.approx(1.0)

    def test_single_same_term(self):
        """Single identical term → zero divergence."""
        assert js_divergence(["alpha"], ["alpha"]) == pytest.approx(0.0)

    def test_bounded_zero_one(self):
        """Result should always be in [0, 1]."""
        test_cases = [
            (["a", "b", "c"], ["d", "e", "f"]),
            (["a", "a", "a"], ["a", "b", "c"]),
            (["x"], ["x", "x", "x", "x"]),
            (["a", "b"], ["a", "b", "c", "d", "e"]),
        ]
        for va, vb in test_cases:
            result = js_divergence(va, vb)
            assert 0.0 <= result <= 1.0, f"JSD({va}, {vb}) = {result} out of bounds"

    def test_known_value(self):
        """Verify against a manually computed JSD value.

        P = [1/2, 1/2, 0]  (terms: a, b, c)
        Q = [0, 1/2, 1/2]
        M = [1/4, 1/2, 1/4]
        KL(P||M) = 1/2 * log2(2) + 1/2 * log2(1) = 0.5
        KL(Q||M) = 1/2 * log2(1) + 1/2 * log2(2) = 0.5
        JSD = 0.5 * 0.5 + 0.5 * 0.5 = 0.5
        """
        vocab_a = ["a", "b"]
        vocab_b = ["b", "c"]
        assert js_divergence(vocab_a, vocab_b) == pytest.approx(0.5)


# ── mean_pairwise_js_divergence tests ────────────────────────────────


class TestMeanPairwiseJSD:
    def test_single_persona(self):
        """Can't compute pairwise with < 2 personas."""
        assert mean_pairwise_js_divergence([["sprint", "backlog"]]) == 0.0

    def test_empty_list(self):
        assert mean_pairwise_js_divergence([]) == 0.0

    def test_two_identical_personas(self):
        vocab = ["sprint", "backlog", "standup"]
        assert mean_pairwise_js_divergence([vocab, vocab]) == pytest.approx(0.0)

    def test_two_disjoint_personas(self):
        vocab_a = ["sprint", "backlog"]
        vocab_b = ["kubernetes", "docker"]
        assert mean_pairwise_js_divergence([vocab_a, vocab_b]) == pytest.approx(1.0)

    def test_three_personas(self):
        """Mean of 3 pairwise comparisons."""
        vocabs = [
            ["sprint", "backlog", "scrum"],
            ["kubernetes", "docker", "terraform"],
            ["sprint", "kubernetes", "api"],
        ]
        result = mean_pairwise_js_divergence(vocabs)
        # Should be mean of JSD(0,1), JSD(0,2), JSD(1,2)
        jsd_01 = js_divergence(vocabs[0], vocabs[1])
        jsd_02 = js_divergence(vocabs[0], vocabs[2])
        jsd_12 = js_divergence(vocabs[1], vocabs[2])
        expected = (jsd_01 + jsd_02 + jsd_12) / 3
        assert result == pytest.approx(expected)

    def test_all_identical(self):
        vocab = ["sprint", "deploy", "api"]
        assert mean_pairwise_js_divergence([vocab, vocab, vocab]) == pytest.approx(0.0)


# ── pairwise_js_matrix tests ────────────────────────────────────────


class TestPairwiseJSMatrix:
    def test_structure(self):
        vocabs = [["a", "b"], ["c", "d"], ["a", "c"]]
        names = ["Alpha", "Beta", "Gamma"]
        result = pairwise_js_matrix(vocabs, names)
        assert "mean_jsd" in result
        assert "min_jsd" in result
        assert "max_jsd" in result
        assert "n_personas" in result
        assert "n_pairs" in result
        assert "pairs" in result
        assert result["n_personas"] == 3
        assert result["n_pairs"] == 3  # C(3,2) = 3

    def test_pair_names(self):
        vocabs = [["a"], ["b"]]
        names = ["Persona_X", "Persona_Y"]
        result = pairwise_js_matrix(vocabs, names)
        assert len(result["pairs"]) == 1
        assert result["pairs"][0]["persona_a"] == "Persona_X"
        assert result["pairs"][0]["persona_b"] == "Persona_Y"

    def test_default_names(self):
        vocabs = [["a"], ["b"]]
        result = pairwise_js_matrix(vocabs)
        assert result["pairs"][0]["persona_a"] == "persona_0"
        assert result["pairs"][0]["persona_b"] == "persona_1"

    def test_single_persona(self):
        result = pairwise_js_matrix([["a", "b"]])
        assert result["mean_jsd"] == 0.0
        assert result["n_pairs"] == 0

    def test_min_max_consistency(self):
        vocabs = [["a", "b"], ["c", "d"], ["a", "c"]]
        result = pairwise_js_matrix(vocabs)
        assert result["min_jsd"] <= result["mean_jsd"] <= result["max_jsd"]


# ── vocab_overlap_stats tests ────────────────────────────────────────


class TestVocabOverlapStats:
    def test_no_overlap(self):
        vocabs = [["a", "b"], ["c", "d"]]
        result = vocab_overlap_stats(vocabs)
        assert result["mean_jaccard"] == pytest.approx(0.0)
        assert result["unique_ratio"] == pytest.approx(1.0)

    def test_full_overlap(self):
        vocab = ["a", "b", "c"]
        result = vocab_overlap_stats([vocab, vocab])
        assert result["mean_jaccard"] == pytest.approx(1.0)
        assert result["unique_ratio"] == pytest.approx(0.0)

    def test_partial_overlap(self):
        vocabs = [["a", "b", "c"], ["b", "c", "d"]]
        result = vocab_overlap_stats(vocabs)
        # Jaccard: |{b,c}| / |{a,b,c,d}| = 2/4 = 0.5
        assert result["mean_jaccard"] == pytest.approx(0.5)
        # Shared: {b, c}, unique to one: {a, d}
        assert result["terms_shared_across_personas"] == 2
        assert result["terms_unique_to_one_persona"] == 2

    def test_single_persona(self):
        result = vocab_overlap_stats([["a", "b"]])
        assert result["mean_jaccard"] == 0.0
        assert result["unique_ratio"] == 1.0

    def test_three_personas(self):
        vocabs = [["a", "b"], ["b", "c"], ["c", "d"]]
        result = vocab_overlap_stats(vocabs)
        assert result["total_unique_terms"] == 4  # a, b, c, d
        # Shared across at least 2: b (in 0,1), c (in 1,2)
        assert result["terms_shared_across_personas"] == 2
        assert result["terms_unique_to_one_persona"] == 2  # a, d
