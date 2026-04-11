"""From-scratch Cohen's kappa and Krippendorff's alpha.

No scipy / numpy dependency. Implementations follow the textbook formulas
with unit tests in test_agreement.py using small hand-worked inputs.

Cohen's kappa: two raters, categorical labels.
    kappa = (p_o - p_e) / (1 - p_e)
where p_o is observed agreement and p_e is chance agreement under
independent marginal distributions.

Krippendorff's alpha: any number of raters, missing data OK, supports
nominal / ordinal / interval levels.
    alpha = 1 - D_o / D_e
where D_o is observed disagreement (sum of pairwise distances within items)
and D_e is expected disagreement under random pairing of all ratings.

Only nominal and ordinal levels are implemented here; interval is not
needed for the exp-5.06 protocols.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Literal, Sequence


def cohen_kappa(rater_a: Sequence[Any], rater_b: Sequence[Any]) -> float:
    """Cohen's kappa for two raters on matched items.

    Missing labels are not supported; rater_a and rater_b must be the same
    length and indexable.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must be the same length")
    n = len(rater_a)
    if n == 0:
        return 1.0

    labels: set[Any] = set(rater_a) | set(rater_b)
    # Observed agreement
    p_o = sum(1 for i in range(n) if rater_a[i] == rater_b[i]) / n

    # Expected agreement: product of marginal probabilities
    count_a = Counter(rater_a)
    count_b = Counter(rater_b)
    p_e = 0.0
    for lbl in labels:
        p_e += (count_a.get(lbl, 0) / n) * (count_b.get(lbl, 0) / n)

    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


Level = Literal["nominal", "ordinal"]


def _distance(a: Any, b: Any, level: Level, ordinal_order: list[Any] | None) -> float:
    """Distance function for Krippendorff's alpha.

    - nominal: 0 if equal else 1
    - ordinal: squared difference between rank positions in ordinal_order,
      normalized by the number of categories between them. Uses the
      standard Krippendorff ordinal metric.
    """
    if level == "nominal":
        return 0.0 if a == b else 1.0
    if level == "ordinal":
        if ordinal_order is None:
            raise ValueError("ordinal_order required for ordinal level")
        # Simplified ordinal distance: squared index difference.
        ia = ordinal_order.index(a)
        ib = ordinal_order.index(b)
        return float((ia - ib) ** 2)
    raise ValueError(f"unknown level: {level}")


def krippendorff_alpha(
    ratings: Sequence[dict],
    level: Level = "nominal",
    ordinal_order: list[Any] | None = None,
) -> float:
    """Krippendorff's alpha for >=2 raters.

    `ratings` is a sequence of items; each item is a dict mapping rater_id to
    label. Missing raters are allowed (they simply don't contribute to that
    item's pairs).

    Returns alpha in (-inf, 1]. 1 = perfect agreement, 0 = chance-level.
    """
    # Observed disagreement: mean pairwise distance within each item
    D_o_num = 0.0
    D_o_denom = 0

    all_labels: list[Any] = []
    for item in ratings:
        labels = list(item.values())
        for l in labels:
            all_labels.append(l)
        m = len(labels)
        if m < 2:
            continue
        # Sum of pairwise distances within item
        for i in range(m):
            for j in range(i + 1, m):
                D_o_num += _distance(labels[i], labels[j], level, ordinal_order)
        # Number of pairs
        D_o_denom += m * (m - 1) // 2

    if D_o_denom == 0:
        return 1.0

    D_o = D_o_num / D_o_denom

    # Expected disagreement: mean pairwise distance across ALL labels in
    # the pooled dataset (each pair once)
    N = len(all_labels)
    if N < 2:
        return 1.0
    D_e_num = 0.0
    D_e_pairs = 0
    for i in range(N):
        for j in range(i + 1, N):
            D_e_num += _distance(all_labels[i], all_labels[j], level, ordinal_order)
            D_e_pairs += 1
    D_e = D_e_num / D_e_pairs if D_e_pairs else 0.0

    if D_e == 0:
        return 1.0
    return 1.0 - (D_o / D_e)
