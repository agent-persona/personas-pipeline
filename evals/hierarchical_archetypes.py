"""Experiment 6.23: Hierarchical archetype analysis helpers."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from itertools import combinations
from statistics import mean

from synthesis.models.cluster import ClusterData


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _persona_text(persona: dict) -> str:
    parts: list[str] = []
    for field in (
        "name",
        "summary",
        "goals",
        "pains",
        "motivations",
        "objections",
        "channels",
        "vocabulary",
        "decision_triggers",
        "sample_quotes",
    ):
        value = persona.get(field, [])
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        else:
            parts.append(str(value))
    for stage in persona.get("journey_stages", []):
        if isinstance(stage, dict):
            parts.append(str(stage.get("stage", "")))
            parts.append(str(stage.get("mindset", "")))
            parts.extend(str(item) for item in stage.get("key_actions", []))
            parts.extend(str(item) for item in stage.get("content_preferences", []))
    return " ".join(parts)


def persona_tokens(persona: dict) -> set[str]:
    return _tokenize(_persona_text(persona))


def unique_trait_count(persona: dict) -> int:
    traits: set[str] = set()
    for field in (
        "goals",
        "pains",
        "motivations",
        "objections",
        "channels",
        "vocabulary",
        "decision_triggers",
        "sample_quotes",
    ):
        value = persona.get(field, [])
        if isinstance(value, list):
            traits |= {str(item).strip().lower() for item in value if str(item).strip()}
    for stage in persona.get("journey_stages", []):
        if isinstance(stage, dict):
            for key in ("stage", "mindset"):
                value = str(stage.get(key, "")).strip().lower()
                if value:
                    traits.add(value)
            traits |= {str(item).strip().lower() for item in stage.get("key_actions", []) if str(item).strip()}
            traits |= {str(item).strip().lower() for item in stage.get("content_preferences", []) if str(item).strip()}
    return len(traits)


def persona_similarity(persona_a: dict, persona_b: dict) -> float:
    return _jaccard(persona_tokens(persona_a), persona_tokens(persona_b))


def pairwise_distinctiveness(personas: list[dict]) -> float:
    if len(personas) < 2:
        return 0.0
    scores = [1.0 - persona_similarity(a, b) for a, b in combinations(personas, 2)]
    return mean(scores) if scores else 0.0


def source_tokens(cluster: ClusterData) -> set[str]:
    parts: list[str] = [
        cluster.cluster_id,
        cluster.tenant.tenant_id,
        cluster.tenant.industry or "",
        cluster.tenant.product_description or "",
    ]
    parts.extend(cluster.summary.top_behaviors)
    parts.extend(cluster.summary.top_pages)
    parts.extend(cluster.summary.top_referrers)
    for key, value in cluster.summary.extra.items():
        parts.append(str(key))
        parts.append(str(value))
    for record in cluster.sample_records:
        parts.append(record.record_id)
        parts.append(record.source)
        if record.timestamp:
            parts.append(record.timestamp)
        if record.payload:
            parts.append(json.dumps(record.payload, sort_keys=True, default=str))
    if cluster.enrichment.firmographic:
        parts.append(json.dumps(cluster.enrichment.firmographic, sort_keys=True, default=str))
    if cluster.enrichment.intent_signals:
        parts.extend(cluster.enrichment.intent_signals)
    if cluster.enrichment.technographic:
        parts.append(json.dumps(cluster.enrichment.technographic, sort_keys=True, default=str))
    return _tokenize(" ".join(parts))


def coverage_ratio(cluster: ClusterData, personas: list[dict]) -> float:
    source = source_tokens(cluster)
    persona_union: set[str] = set()
    for persona in personas:
        persona_union |= persona_tokens(persona)
    if not source:
        return 0.0
    return len(source & persona_union) / len(source)


def info_density(persona: dict) -> int:
    return unique_trait_count(persona)


@dataclass
class PersonaArtifact:
    level: str
    group_id: str
    facet: str
    persona: dict
    judge_overall: float
    judge_dimensions: dict[str, float]
    groundedness: float
    model_used: str
    cost_usd: float


@dataclass
class GroupSummary:
    group_id: str
    flat_similarity: float
    hierarchical_similarity: float
    within_group_distinctiveness: float
    cross_group_distinctiveness: float
    coverage_ratio: float
    mean_information_density: float
    mean_judge_overall: float


@dataclass
class ExperimentSummary:
    provider: str
    judge_model: str
    synthesis_model: str
    n_clusters: int
    flat_mean_judge_overall: float
    hierarchy_mean_judge_overall: float
    flat_coverage_ratio: float
    hierarchy_coverage_ratio: float
    flat_information_density: float
    hierarchy_information_density: float
    within_parent_distinctiveness: float
    across_parent_distinctiveness: float
    set_coherence_delta: float
    coverage_delta: float
    information_density_delta: float


def summarize_group(group_id: str, flat_persona: dict, hierarchical_personas: list[dict], cluster: ClusterData, judge_scores: list[float]) -> GroupSummary:
    flat_similarity = 1.0
    hierarchical_similarity = pairwise_distinctiveness(hierarchical_personas)
    if len(hierarchical_personas) >= 2:
        hierarchical_similarity = 1.0 - hierarchical_similarity
    return GroupSummary(
        group_id=group_id,
        flat_similarity=flat_similarity,
        hierarchical_similarity=hierarchical_similarity,
        within_group_distinctiveness=pairwise_distinctiveness(hierarchical_personas[1:]) if len(hierarchical_personas) > 1 else 0.0,
        cross_group_distinctiveness=0.0,
        coverage_ratio=coverage_ratio(cluster, [flat_persona, *hierarchical_personas]),
        mean_information_density=mean(info_density(persona) for persona in [flat_persona, *hierarchical_personas]),
        mean_judge_overall=mean(judge_scores) if judge_scores else 0.0,
    )


def results_to_dict(data: dict) -> dict:
    return {
        "summary": asdict(data["summary"]),
        "flat": [asdict(item) for item in data["flat"]],
        "hierarchy": [asdict(item) for item in data["hierarchy"]],
        "groups": [asdict(item) for item in data["groups"]],
    }
