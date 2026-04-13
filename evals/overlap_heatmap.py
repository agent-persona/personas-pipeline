"""Experiment 6.13: Persona overlap heatmap.

Computes field-by-field similarity between every pair of personas in a set
and produces a similarity matrix. The key metric — *diagonal density* — is
the mean off-diagonal similarity: lower means personas are more distinct.

Similarity is computed per-field, then averaged across fields:
  - **list fields** (goals, pains, motivations, objections, channels,
    vocabulary, decision_triggers, sample_quotes): Jaccard similarity
    on lowercased word bags per item.
  - **text fields** (name, summary): Jaccard on word sets.
  - **struct fields** (demographics, firmographics): field-by-field
    exact-match ratio.

Usage:
    matrix = compute_similarity_matrix(personas)
    density = diagonal_density(matrix)
    render_heatmap(matrix, persona_names, path)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


# ── Field definitions ────────────────────────────────────────────────

LIST_FIELDS = (
    "goals",
    "pains",
    "motivations",
    "objections",
    "channels",
    "vocabulary",
    "decision_triggers",
    "sample_quotes",
)

TEXT_FIELDS = (
    "name",
    "summary",
)

STRUCT_FIELDS = (
    "demographics",
    "firmographics",
)

# Fields that contribute to overlap (source_evidence, journey_stages,
# schema_version excluded — they're structural, not persona identity).
ALL_COMPARED_FIELDS = LIST_FIELDS + TEXT_FIELDS + STRUCT_FIELDS


# ── Similarity primitives ───────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens, strip punctuation."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity: |A∩B| / |A∪B|. 0 if both empty."""
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def list_field_similarity(list_a: list[str], list_b: list[str]) -> float:
    """Similarity between two persona list fields.

    Pools all items into word bags, then computes Jaccard.
    This captures thematic overlap (e.g. two personas both mentioning
    "reduce manual work" in goals) without requiring exact string match.
    """
    words_a = set()
    for item in list_a:
        words_a |= _tokenize(item)
    words_b = set()
    for item in list_b:
        words_b |= _tokenize(item)
    return jaccard(words_a, words_b)


def text_field_similarity(text_a: str, text_b: str) -> float:
    """Word-level Jaccard between two text strings."""
    return jaccard(_tokenize(text_a), _tokenize(text_b))


def struct_field_similarity(struct_a: dict, struct_b: dict) -> float:
    """Field-by-field exact-match ratio between two flat dicts.

    For each key present in either dict, score 1.0 if values match
    (case-insensitive for strings), 0.0 otherwise. Return mean.
    """
    all_keys = set(struct_a.keys()) | set(struct_b.keys())
    if not all_keys:
        return 0.0
    matches = 0
    for key in all_keys:
        va = struct_a.get(key)
        vb = struct_b.get(key)
        if va is None or vb is None:
            continue
        if isinstance(va, str) and isinstance(vb, str):
            if va.lower().strip() == vb.lower().strip():
                matches += 1
        elif isinstance(va, list) and isinstance(vb, list):
            # For list sub-fields (e.g. role_titles), use Jaccard
            sa = {str(x).lower() for x in va}
            sb = {str(x).lower() for x in vb}
            matches += jaccard(sa, sb)
        elif va == vb:
            matches += 1
    return matches / len(all_keys)


# ── Pairwise persona similarity ─────────────────────────────────────

@dataclass
class FieldSimilarity:
    """Similarity breakdown for one persona pair."""
    persona_a: str
    persona_b: str
    overall: float
    per_field: dict[str, float] = field(default_factory=dict)


def persona_similarity(a: dict, b: dict) -> FieldSimilarity:
    """Compute field-by-field similarity between two persona dicts."""
    scores: dict[str, float] = {}

    for f in LIST_FIELDS:
        la = a.get(f, [])
        lb = b.get(f, [])
        if isinstance(la, list) and isinstance(lb, list):
            scores[f] = list_field_similarity(la, lb)
        else:
            scores[f] = 0.0

    for f in TEXT_FIELDS:
        ta = str(a.get(f, ""))
        tb = str(b.get(f, ""))
        scores[f] = text_field_similarity(ta, tb)

    for f in STRUCT_FIELDS:
        sa = a.get(f, {})
        sb = b.get(f, {})
        if isinstance(sa, dict) and isinstance(sb, dict):
            scores[f] = struct_field_similarity(sa, sb)
        else:
            scores[f] = 0.0

    valid = [v for v in scores.values() if not math.isnan(v)]
    overall = sum(valid) / len(valid) if valid else 0.0

    return FieldSimilarity(
        persona_a=a.get("name", "?"),
        persona_b=b.get("name", "?"),
        overall=overall,
        per_field=scores,
    )


# ── Similarity matrix ───────────────────────────────────────────────

@dataclass
class SimilarityMatrix:
    """NxN similarity matrix for a set of personas."""
    names: list[str]
    matrix: list[list[float]]             # matrix[i][j] = overall similarity
    per_field: list[list[dict[str, float]]]  # per_field[i][j] = {field: sim}
    n: int


def compute_similarity_matrix(personas: list[dict]) -> SimilarityMatrix:
    """Build the full NxN similarity matrix."""
    n = len(personas)
    names = [p.get("name", f"Persona {i+1}") for i, p in enumerate(personas)]
    matrix = [[0.0] * n for _ in range(n)]
    per_field = [[{} for _ in range(n)] for _ in range(n)]

    for i in range(n):
        matrix[i][i] = 1.0
        per_field[i][i] = {f: 1.0 for f in ALL_COMPARED_FIELDS}
        for j in range(i + 1, n):
            sim = persona_similarity(personas[i], personas[j])
            matrix[i][j] = sim.overall
            matrix[j][i] = sim.overall
            per_field[i][j] = sim.per_field
            per_field[j][i] = sim.per_field

    return SimilarityMatrix(names=names, matrix=matrix, per_field=per_field, n=n)


# ── Diagonal density metric ─────────────────────────────────────────

def diagonal_density(matrix: SimilarityMatrix) -> float:
    """Mean off-diagonal similarity. Lower = more distinct personas.

    The diagonal is always 1.0 (self-similarity), so we exclude it.
    A set of perfectly distinct personas would score near 0.0.
    A set of clones would score near 1.0.
    """
    if matrix.n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(matrix.n):
        for j in range(matrix.n):
            if i != j:
                total += matrix.matrix[i][j]
                count += 1
    return total / count if count else 0.0


def per_field_density(matrix: SimilarityMatrix) -> dict[str, float]:
    """Mean off-diagonal similarity per field. Identifies which fields
    drive the most overlap between personas."""
    if matrix.n < 2:
        return {f: 0.0 for f in ALL_COMPARED_FIELDS}

    sums: dict[str, float] = {f: 0.0 for f in ALL_COMPARED_FIELDS}
    count = 0
    for i in range(matrix.n):
        for j in range(matrix.n):
            if i != j:
                for f in ALL_COMPARED_FIELDS:
                    sums[f] += matrix.per_field[i][j].get(f, 0.0)
                count += 1

    return {f: s / count if count else 0.0 for f, s in sums.items()}


def max_overlap_pair(matrix: SimilarityMatrix) -> tuple[str, str, float]:
    """Return the most-overlapping persona pair (excluding self)."""
    if matrix.n < 2:
        return ("", "", 0.0)
    best_i, best_j, best_sim = 0, 1, -1.0
    for i in range(matrix.n):
        for j in range(i + 1, matrix.n):
            if matrix.matrix[i][j] > best_sim:
                best_sim = matrix.matrix[i][j]
                best_i, best_j = i, j
    return (matrix.names[best_i], matrix.names[best_j], best_sim)


# ── Heatmap rendering ───────────────────────────────────────────────

def render_heatmap(
    matrix: SimilarityMatrix,
    output_path: Path | str,
    title: str = "Persona Overlap Heatmap",
) -> Path:
    """Render the similarity matrix as a heatmap PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_path = Path(output_path)
    data = np.array(matrix.matrix)

    fig, ax = plt.subplots(figsize=(max(6, matrix.n * 1.5), max(5, matrix.n * 1.2)))
    im = ax.imshow(data, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")

    # Short names for axis labels
    short_names = []
    for name in matrix.names:
        words = name.split()
        short_names.append(words[0] if len(words) > 0 else name)

    ax.set_xticks(range(matrix.n))
    ax.set_yticks(range(matrix.n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)

    # Annotate cells with values
    for i in range(matrix.n):
        for j in range(matrix.n):
            val = data[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=8)

    ax.set_title(title, fontsize=12, pad=12)
    fig.colorbar(im, ax=ax, label="Similarity", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def render_field_heatmap(
    matrix: SimilarityMatrix,
    output_path: Path | str,
    title: str = "Per-Field Overlap Density",
) -> Path:
    """Render per-field density as a bar-style heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_path = Path(output_path)
    densities = per_field_density(matrix)

    fields = list(densities.keys())
    values = [densities[f] for f in fields]

    fig, ax = plt.subplots(figsize=(10, max(4, len(fields) * 0.4)))
    colors = plt.cm.YlOrRd([v for v in values])
    bars = ax.barh(fields, values, color=colors)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Mean Off-Diagonal Similarity")
    ax.set_title(title, fontsize=12)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path
