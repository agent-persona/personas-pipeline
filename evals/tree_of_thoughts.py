"""Experiment 2.10: Tree-of-thoughts synthesis helper."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from synthesis.engine.groundedness import check_groundedness
from synthesis.engine.prompt_builder import SYSTEM_PROMPT, build_messages, build_tool_definition
from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1

from evals.judge_helper_2_10 import LLMJudge


@dataclass
class CandidateResult:
    stage: str
    overall_score: float
    judge_rationale: str
    persona: dict
    synthesis_cost_usd: float


@dataclass
class TreeOfThoughtsResult:
    control: CandidateResult
    candidates: list[CandidateResult]
    refined: CandidateResult
    total_candidate_cost_usd: float
    total_token_budget_proxy: float
    convergence_delta: float


async def refine_persona_with_feedback(
    cluster: ClusterData,
    backend,
    persona: dict,
    judge_feedback: str,
) -> tuple[dict, float]:
    messages = build_messages(cluster)
    messages.append(
        {
            "role": "user",
            "content": (
                "Refine the persona below. Improve weak spots called out in the judge feedback "
                "while staying fully grounded in the provided records.\n\n"
                "CURRENT PERSONA:\n"
                + json.dumps(persona, indent=2, default=str)
                + "\n\nJUDGE FEEDBACK:\n"
                + judge_feedback
            ),
        }
    )
    llm_result = await backend.generate(
        system=SYSTEM_PROMPT,
        messages=messages,
        tool=build_tool_definition(),
    )
    refined = PersonaV1.model_validate(llm_result.tool_input)
    groundedness = check_groundedness(refined, cluster)
    if not groundedness.passed:
        raise ValueError(f"Refined persona failed groundedness: {groundedness.violations}")
    return refined.model_dump(mode="json"), llm_result.estimated_cost_usd


async def run_tree_of_thoughts(
    cluster: ClusterData,
    control_backend,
    candidate_backend,
    judge: LLMJudge,
    synthesize_fn,
) -> TreeOfThoughtsResult:
    control_raw = await synthesize_fn(cluster, control_backend, max_retries=4)
    control_score = await judge.score_persona(control_raw.persona.model_dump(mode="json"))
    control = CandidateResult(
        stage="control",
        overall_score=control_score.overall,
        judge_rationale=control_score.rationale,
        persona=control_raw.persona.model_dump(mode="json"),
        synthesis_cost_usd=control_raw.total_cost_usd,
    )

    candidates: list[CandidateResult] = []
    total_cost = control.synthesis_cost_usd
    for candidate_index in range(3):
        candidate_raw = await synthesize_fn(cluster, candidate_backend, max_retries=4)
        candidate_persona = candidate_raw.persona.model_dump(mode="json")
        candidate_score = await judge.score_persona(candidate_persona)
        candidate = CandidateResult(
            stage=f"candidate_{candidate_index + 1}",
            overall_score=candidate_score.overall,
            judge_rationale=candidate_score.rationale,
            persona=candidate_persona,
            synthesis_cost_usd=candidate_raw.total_cost_usd,
        )
        candidates.append(candidate)
        total_cost += candidate.synthesis_cost_usd

    kept = sorted(candidates, key=lambda candidate: candidate.overall_score, reverse=True)[:2]
    best = kept[0]
    refined_persona, refine_cost = await refine_persona_with_feedback(
        cluster=cluster,
        backend=control_backend,
        persona=best.persona,
        judge_feedback=best.judge_rationale,
    )
    refined_score = await judge.score_persona(refined_persona)
    refined = CandidateResult(
        stage="refined",
        overall_score=refined_score.overall,
        judge_rationale=refined_score.rationale,
        persona=refined_persona,
        synthesis_cost_usd=refine_cost,
    )
    total_cost += refine_cost

    return TreeOfThoughtsResult(
        control=control,
        candidates=candidates,
        refined=refined,
        total_candidate_cost_usd=total_cost,
        total_token_budget_proxy=total_cost,
        convergence_delta=refined.overall_score - best.overall_score,
    )


def result_to_dict(result: TreeOfThoughtsResult) -> dict:
    return asdict(result)
