"""Experiment 2.12: Self-consistency voting."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from statistics import mean

from synthesis.engine.groundedness import check_groundedness
from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1

from evals.judge_helper_2_12 import LLMJudge

VOTED_FIELDS = (
    "goals",
    "pains",
    "motivations",
    "objections",
    "channels",
    "vocabulary",
    "decision_triggers",
    "sample_quotes",
)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _similarity(a: str, b: str) -> float:
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a and not tokens_b:
        return 1.0
    union = tokens_a | tokens_b
    return len(tokens_a & tokens_b) / len(union) if union else 0.0


def _cluster_items(items: list[str], threshold: float = 0.6) -> list[list[str]]:
    clusters: list[list[str]] = []
    for item in items:
        placed = False
        for cluster in clusters:
            if any(_similarity(item, existing) >= threshold for existing in cluster):
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])
    return clusters


def majority_vote(items_by_sample: list[list[str]], min_support: int = 3) -> list[str]:
    raw_items = [item for sample in items_by_sample for item in sample if isinstance(item, str) and item.strip()]
    clusters = _cluster_items(raw_items)
    voted: list[str] = []
    for cluster in clusters:
        support = 0
        for sample in items_by_sample:
            if any(any(_similarity(candidate, item) >= 0.6 for item in cluster) for candidate in sample):
                support += 1
        if support >= min_support:
            counts = Counter(cluster)
            voted.append(counts.most_common(1)[0][0])
    return voted


def vote_support_ratio(items_by_sample: list[list[str]], voted_items: list[str], min_support: int = 3) -> float:
    if not voted_items:
        return 0.0
    supported = 0
    for item in voted_items:
        support = 0
        for sample in items_by_sample:
            if any(_similarity(item, candidate) >= 0.6 for candidate in sample):
                support += 1
        if support >= min_support:
            supported += 1
    return supported / len(voted_items)


def build_voted_persona(best_sample: dict, samples: list[dict]) -> dict:
    voted = dict(best_sample)
    for field in VOTED_FIELDS:
        items_by_sample = [sample.get(field, []) for sample in samples]
        if all(isinstance(items, list) for items in items_by_sample):
            voted[field] = majority_vote(items_by_sample)
    return voted


@dataclass
class PersonaComparison:
    cluster_id: str
    control_persona: str
    best_sample_persona: str
    voted_persona: str
    control_overall: float
    best_sample_overall: float
    voted_overall: float
    control_groundedness: float
    best_sample_groundedness: float
    voted_groundedness: float
    content_richness: int
    vote_support_ratio: float
    best_sample_rationale: str
    voted_rationale: str


@dataclass
class VotingSummary:
    n_clusters: int
    mean_control_overall: float
    mean_best_sample_overall: float
    mean_voted_overall: float
    mean_control_groundedness: float
    mean_best_sample_groundedness: float
    mean_voted_groundedness: float
    mean_content_richness: float
    mean_vote_support_ratio: float


def content_richness(persona: dict) -> int:
    richness = 0
    for field in VOTED_FIELDS:
        values = persona.get(field, [])
        if isinstance(values, list):
            richness += len({str(v).strip().lower() for v in values if str(v).strip()})
    return richness


def compare_personas(
    cluster: ClusterData,
    control: dict,
    best_sample: dict,
    voted: dict,
    control_score,
    best_score,
    voted_score,
    control_groundedness: float,
    best_sample_groundedness: float,
    voted_groundedness: float,
    vote_support_ratio: float,
) -> PersonaComparison:
    return PersonaComparison(
        cluster_id=cluster.cluster_id,
        control_persona=control.get("name", "unknown"),
        best_sample_persona=best_sample.get("name", "unknown"),
        voted_persona=voted.get("name", "unknown"),
        control_overall=control_score.overall,
        best_sample_overall=best_score.overall,
        voted_overall=voted_score.overall,
        control_groundedness=control_groundedness,
        best_sample_groundedness=best_sample_groundedness,
        voted_groundedness=voted_groundedness,
        content_richness=content_richness(voted),
        vote_support_ratio=vote_support_ratio,
        best_sample_rationale=best_score.rationale,
        voted_rationale=voted_score.rationale,
    )


def summarize(comparisons: list[PersonaComparison]) -> VotingSummary:
    if not comparisons:
        return VotingSummary(
            n_clusters=0,
            mean_control_overall=0.0,
            mean_best_sample_overall=0.0,
            mean_voted_overall=0.0,
            mean_control_groundedness=0.0,
            mean_best_sample_groundedness=0.0,
            mean_voted_groundedness=0.0,
            mean_content_richness=0.0,
            mean_vote_support_ratio=0.0,
        )

    return VotingSummary(
        n_clusters=len(comparisons),
        mean_control_overall=mean(c.control_overall for c in comparisons),
        mean_best_sample_overall=mean(c.best_sample_overall for c in comparisons),
        mean_voted_overall=mean(c.voted_overall for c in comparisons),
        mean_control_groundedness=mean(c.control_groundedness for c in comparisons),
        mean_best_sample_groundedness=mean(c.best_sample_groundedness for c in comparisons),
        mean_voted_groundedness=mean(c.voted_groundedness for c in comparisons),
        mean_content_richness=mean(c.content_richness for c in comparisons),
        mean_vote_support_ratio=mean(c.vote_support_ratio for c in comparisons),
    )


def results_to_dict(comparisons: list[PersonaComparison], summary: VotingSummary) -> dict:
    return {
        "comparisons": [asdict(c) for c in comparisons],
        "summary": asdict(summary),
    }


def validate_persona(persona: dict, cluster: ClusterData) -> tuple[PersonaV1, float]:
    model = PersonaV1.model_validate(persona)
    groundedness = check_groundedness(model, cluster).score
    return model, groundedness
