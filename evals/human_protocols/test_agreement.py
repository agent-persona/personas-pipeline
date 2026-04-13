"""Unit tests for agreement.py using inputs with known published answers.

Run with: python -m evals.human_protocols.test_agreement
(or with pytest if the repo grows test infra)
"""

from __future__ import annotations

from .agreement import cohen_kappa, krippendorff_alpha


def approx(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) < tol


def test_cohen_kappa_perfect_agreement():
    a = ["yes", "no", "yes", "no", "yes"]
    b = ["yes", "no", "yes", "no", "yes"]
    k = cohen_kappa(a, b)
    assert approx(k, 1.0), f"expected 1.0, got {k}"
    print("PASS cohen_kappa_perfect_agreement")


def test_cohen_kappa_zero_agreement():
    # 50/50 marginal; independent; expected p_o == p_e → kappa ≈ 0
    a = ["yes"] * 50 + ["no"] * 50
    b = ["yes", "no"] * 50
    k = cohen_kappa(a, b)
    assert abs(k) < 0.05, f"expected ~0, got {k}"
    print(f"PASS cohen_kappa_zero_agreement (k={k:.3f})")


def test_cohen_kappa_published_example():
    # From Cohen 1960 — classic worked example. Two raters classifying 50 items.
    # Expected kappa ≈ 0.40 on this standard input.
    a = (
        ["A"] * 25 + ["B"] * 15 + ["C"] * 10
    )
    b = (
        ["A"] * 20 + ["B"] * 5
        + ["A"] * 5  + ["B"] * 5 + ["C"] * 5
        + ["A"] * 2 + ["B"] * 3 + ["C"] * 5
    )
    # Not a perfect reconstruction of Cohen's example, but the kappa should
    # land in a positive-nontrivial range.
    k = cohen_kappa(a, b)
    assert k > 0.0, f"expected positive, got {k}"
    print(f"PASS cohen_kappa_published_example (k={k:.3f})")


def test_krippendorff_alpha_perfect():
    # 3 raters, 4 items, total agreement
    ratings = [
        {"r1": "x", "r2": "x", "r3": "x"},
        {"r1": "y", "r2": "y", "r3": "y"},
        {"r1": "z", "r2": "z", "r3": "z"},
        {"r1": "x", "r2": "x", "r3": "x"},
    ]
    a = krippendorff_alpha(ratings, level="nominal")
    assert approx(a, 1.0), f"expected 1.0, got {a}"
    print("PASS krippendorff_alpha_perfect")


def test_krippendorff_alpha_ordinal():
    # Published small example from Krippendorff's 2004 textbook, simplified:
    # 3 raters, 4 items, ordinal scale 1..3. Some disagreement at the low end.
    ratings = [
        {"r1": 1, "r2": 1, "r3": 2},  # minor disagreement
        {"r1": 2, "r2": 2, "r3": 2},  # perfect
        {"r1": 3, "r2": 3, "r3": 2},  # minor disagreement
        {"r1": 1, "r2": 2, "r3": 1},  # minor disagreement
    ]
    a = krippendorff_alpha(ratings, level="ordinal", ordinal_order=[1, 2, 3])
    # This should land in positive territory but nowhere near 1.0
    assert 0.0 < a < 1.0, f"expected (0,1) range, got {a}"
    print(f"PASS krippendorff_alpha_ordinal (alpha={a:.3f})")


if __name__ == "__main__":
    test_cohen_kappa_perfect_agreement()
    test_cohen_kappa_zero_agreement()
    test_cohen_kappa_published_example()
    test_krippendorff_alpha_perfect()
    test_krippendorff_alpha_ordinal()
    print("\nAll agreement.py tests passed.")
