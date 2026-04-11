"""Experiment 2.09: Best-of-N selection helpers."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from statistics import mean

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

TEXT_FIELDS = ("name", "summary")

STRUCT_FIELDS = ("demographics", "firmographics")


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def list_field_similarity(list_a: list[str], list_b: list[str]) -> float:
    tokens_a = set()
    tokens_b = set()
    for item in list_a:
        tokens_a |= _tokenize(item)
    for item in list_b:
        tokens_b |= _tokenize(item)
    return _jaccard(tokens_a, tokens_b)


def text_field_similarity(text_a: str, text_b: str) -> float:
    return _jaccard(_tokenize(text_a), _tokenize(text_b))


def struct_field_similarity(struct_a: dict, struct_b: dict) -> float:
    keys = set(struct_a.keys()) | set(struct_b.keys())
    if not keys:
        return 0.0
    score = 0.0
    for key in keys:
        va = struct_a.get(key)
        vb = struct_b.get(key)
        if va is None or vb is None:
            continue
        if isinstance(va, str) and isinstance(vb, str):
            score += 1.0 if va.lower().strip() == vb.lower().strip() else 0.0
        elif isinstance(va, list) and isinstance(vb, list):
            score += _jaccard({str(x).lower() for x in va}, {str(x).lower() for x in vb})
        else:
            score += 1.0 if va == vb else 0.0
    return score / len(keys)


def persona_similarity(persona_a: dict, persona_b: dict) -> float:
    scores: list[float] = []
    for field in LIST_FIELDS:
        scores.append(list_field_similarity(persona_a.get(field, []), persona_b.get(field, [])))
    for field in TEXT_FIELDS:
        scores.append(text_field_similarity(str(persona_a.get(field, "")), str(persona_b.get(field, ""))))
    for field in STRUCT_FIELDS:
        scores.append(struct_field_similarity(persona_a.get(field, {}), persona_b.get(field, {})))
    return mean(scores) if scores else 0.0


def persona_distinctiveness(candidate: dict, pool: list[dict]) -> float:
    if len(pool) <= 1:
        return 1.0
    sims = []
    for other in pool:
        if other is candidate:
            continue
        sims.append(persona_similarity(candidate, other))
    if not sims:
        return 1.0
    return max(0.0, 1.0 - mean(sims))


@dataclass
class CandidateScore:
    persona: dict
    overall: float
    distinctiveness: float
    composite_score: float
    rationale: str = ""
    model_used: str = ""
    cost_usd: float = 0.0


@dataclass
class ClusterSelection:
    cluster_id: str
    control: CandidateScore
    candidates: list[CandidateScore] = field(default_factory=list)
    best_score: CandidateScore | None = None
    best_diverse: CandidateScore | None = None
    mean_pool_similarity: float = 0.0
    mean_pool_distinctiveness: float = 0.0
    control_rank: int = 0
    best_score_rank: int = 0
    best_diverse_rank: int = 0


def score_candidates(candidates: list[CandidateScore]) -> tuple[CandidateScore, CandidateScore]:
    if not candidates:
        raise ValueError("expected at least one candidate")
    best_score = max(candidates, key=lambda c: (c.overall, c.distinctiveness, -c.cost_usd))
    best_diverse = max(candidates, key=lambda c: (c.composite_score, c.overall, -c.cost_usd))
    return best_score, best_diverse


def summarize_cluster(selection: ClusterSelection) -> dict:
    return {
        "cluster_id": selection.cluster_id,
        "control": asdict(selection.control),
        "candidates": [asdict(candidate) for candidate in selection.candidates],
        "best_score": asdict(selection.best_score) if selection.best_score else None,
        "best_diverse": asdict(selection.best_diverse) if selection.best_diverse else None,
        "mean_pool_similarity": selection.mean_pool_similarity,
        "mean_pool_distinctiveness": selection.mean_pool_distinctiveness,
    }


def aggregate_summary(selections: list[ClusterSelection]) -> dict:
    if not selections:
        return {}

    def avg(values: list[float]) -> float:
        return mean(values) if values else 0.0

    return {
        "n_clusters": len(selections),
        "mean_control_overall": avg([s.control.overall for s in selections]),
        "mean_best_score_overall": avg([s.best_score.overall for s in selections if s.best_score]),
        "mean_best_diverse_overall": avg([s.best_diverse.overall for s in selections if s.best_diverse]),
        "mean_control_cost": avg([s.control.cost_usd for s in selections]),
        "mean_best_score_cost": avg([s.best_score.cost_usd for s in selections if s.best_score]),
        "mean_best_diverse_cost": avg([s.best_diverse.cost_usd for s in selections if s.best_diverse]),
        "mean_pool_similarity": avg([s.mean_pool_similarity for s in selections]),
        "mean_pool_distinctiveness": avg([s.mean_pool_distinctiveness for s in selections]),
        "mean_best_score_gain": avg(
            [s.best_score.overall - s.control.overall for s in selections if s.best_score]
        ),
        "mean_best_diverse_gain": avg(
            [s.best_diverse.overall - s.control.overall for s in selections if s.best_diverse]
        ),
    }
