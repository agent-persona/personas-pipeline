"""Greedy sequential ablation for experiment 1.22."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

from synthesis.models.cluster import ClusterData

from evals.judge_helper_1_22 import JudgeScore, LLMJudge

REMOVABLE_FIELDS = (
    "goals",
    "pains",
    "motivations",
    "objections",
    "channels",
    "vocabulary",
    "decision_triggers",
    "sample_quotes",
    "journey_stages",
)


@dataclass
class AblationStep:
    step: int
    removed_field: str
    score_before: float
    score_after: float
    score_drop: float
    remaining_fields: list[str]
    rationale: str


@dataclass
class ClusterSpineResult:
    cluster_id: str
    control_persona_name: str
    control_overall: float
    control_groundedness: float
    control_model_used: str
    control_cost_usd: float
    control_dimensions: dict[str, float]
    control_rationale: str
    collapse_threshold: float
    steps: list[AblationStep]
    collapse_step: int
    spine_fields: list[str]
    spine_overall: float
    spine_dimensions: dict[str, float]
    spine_rationale: str
    removal_order: list[str]
    noncollapsed_fields: list[str]


@dataclass
class SpineSummary:
    n_clusters: int
    mean_control_overall: float
    mean_control_groundedness: float
    mean_spine_overall: float
    mean_quality_drop: float
    mean_spine_size: float
    mean_steps_to_collapse: float
    consensus_spine_fields: list[str]
    union_spine_fields: list[str]
    removal_ranking: list[dict[str, object]]


def ablate_field(persona: dict, field: str) -> dict:
    degraded = deepcopy(persona)
    degraded[field] = []
    return degraded


async def score_persona(judge: LLMJudge, persona: dict) -> JudgeScore:
    return await judge.score_persona(persona)


async def greedy_spine_ablation(
    cluster: ClusterData,
    control_persona: dict,
    judge: LLMJudge,
    control_groundedness: float,
    control_model_used: str,
    control_cost_usd: float,
    collapse_threshold: float = 2.0,
) -> ClusterSpineResult:
    current = deepcopy(control_persona)
    control_score = await score_persona(judge, current)
    current_score = control_score
    remaining_fields = [field for field in REMOVABLE_FIELDS if current.get(field)]
    steps: list[AblationStep] = []
    removal_order: list[str] = []
    collapse_step = 0

    while remaining_fields:
        candidate_scores = []
        for field in remaining_fields:
            candidate = ablate_field(current, field)
            score = await score_persona(judge, candidate)
            candidate_scores.append((field, candidate, score))

        field, candidate, score = max(candidate_scores, key=lambda item: (item[2].overall, item[0]))
        score_drop = current_score.overall - score.overall
        removal_order.append(field)
        score_before = current_score.overall
        current = candidate
        current_score = score
        remaining_fields = [f for f in remaining_fields if f != field]
        steps.append(
            AblationStep(
                step=len(steps) + 1,
                removed_field=field,
                score_before=score_before,
                score_after=score.overall,
                score_drop=score_drop,
                remaining_fields=[f for f in REMOVABLE_FIELDS if current.get(f)],
                rationale=score.rationale,
            )
        )
        collapse_step = len(steps)
        if score.overall < collapse_threshold:
            break

    spine_fields = [f for f in REMOVABLE_FIELDS if current.get(f)]
    return ClusterSpineResult(
        cluster_id=cluster.cluster_id,
        control_persona_name=control_persona.get("name", "unknown"),
        control_overall=control_score.overall,
        control_groundedness=control_groundedness,
        control_model_used=control_model_used,
        control_cost_usd=control_cost_usd,
        control_dimensions=control_score.dimensions,
        control_rationale=control_score.rationale,
        collapse_threshold=collapse_threshold,
        steps=steps,
        collapse_step=collapse_step,
        spine_fields=spine_fields,
        spine_overall=current_score.overall,
        spine_dimensions=current_score.dimensions,
        spine_rationale=current_score.rationale,
        removal_order=removal_order,
        noncollapsed_fields=spine_fields,
    )


def summarize_results(results: list[ClusterSpineResult]) -> SpineSummary:
    if not results:
        return SpineSummary(
            n_clusters=0,
            mean_control_overall=0.0,
            mean_control_groundedness=0.0,
            mean_spine_overall=0.0,
            mean_quality_drop=0.0,
            mean_spine_size=0.0,
            mean_steps_to_collapse=0.0,
            consensus_spine_fields=[],
            union_spine_fields=[],
            removal_ranking=[],
        )

    control_scores = [result.control_overall for result in results]
    control_groundedness = [result.control_groundedness for result in results]
    spine_scores = [result.spine_overall for result in results]
    spine_sizes = [len(result.spine_fields) for result in results]
    step_counts = [result.collapse_step for result in results]

    all_fields = list(REMOVABLE_FIELDS)
    ranking = []
    for field in all_fields:
        observed_steps = []
        for result in results:
            if field in result.removal_order:
                observed_steps.append(result.removal_order.index(field) + 1)
            else:
                observed_steps.append(result.collapse_step + 1)
        ranking.append(
            {
                "field": field,
                "mean_removal_step": sum(observed_steps) / len(observed_steps),
                "survival_rate": sum(1 for result in results if field in result.spine_fields) / len(results),
            }
        )
    ranking.sort(key=lambda item: (item["mean_removal_step"], item["field"]))

    consensus = sorted(set(REMOVABLE_FIELDS).intersection(*(set(result.spine_fields) for result in results)))
    union = sorted(set().union(*(set(result.spine_fields) for result in results)))

    return SpineSummary(
        n_clusters=len(results),
        mean_control_overall=sum(control_scores) / len(control_scores),
        mean_control_groundedness=sum(control_groundedness) / len(control_groundedness),
        mean_spine_overall=sum(spine_scores) / len(spine_scores),
        mean_quality_drop=sum(c - s for c, s in zip(control_scores, spine_scores)) / len(results),
        mean_spine_size=sum(spine_sizes) / len(spine_sizes),
        mean_steps_to_collapse=sum(step_counts) / len(step_counts),
        consensus_spine_fields=consensus,
        union_spine_fields=union,
        removal_ranking=ranking,
    )


def results_to_dict(results: list[ClusterSpineResult], summary: SpineSummary) -> dict:
    return {
        "clusters": [asdict(result) for result in results],
        "summary": asdict(summary),
    }
