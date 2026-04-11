"""Experiment 2.22: Beam search synthesis helper."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from synthesis.engine.groundedness import check_groundedness
from synthesis.engine.prompt_builder import SYSTEM_PROMPT, build_messages, build_tool_definition
from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1

from evals.judge_helper_2_22 import LLMJudge

FOCUS_MODES = ("grounding", "distinctiveness", "balanced")


@dataclass
class BeamCandidate:
    round_id: int
    focus: str
    overall_score: float
    judge_rationale: str
    persona: dict
    cost_usd: float


@dataclass
class BeamSearchResult:
    control_overall: float
    control_cost_usd: float
    final_overall: float
    final_cost_usd: float
    rounds: list[list[BeamCandidate]]
    cost_multiplier: float
    quality_delta: float


def _focus_instruction(focus: str) -> str:
    if focus == "grounding":
        return "Prioritize evidence traceability and explicit support from source records."
    if focus == "distinctiveness":
        return "Prioritize vivid voice, memorable specificity, and contrast from generic personas."
    return "Balance groundedness, distinctiveness, coherence, and actionability evenly."


async def refine_candidate(cluster: ClusterData, backend, seed_persona: dict, focus: str) -> tuple[dict, float]:
    try:
        messages = build_messages(cluster)
        messages.append(
            {
                "role": "user",
                "content": (
                    "Refine the current persona into a stronger version.\n"
                    + _focus_instruction(focus)
                    + "\n\nCURRENT PERSONA:\n"
                    + json.dumps(seed_persona, indent=2, default=str)
                ),
            }
        )
        llm_result = await backend.generate(
            system=SYSTEM_PROMPT,
            messages=messages,
            tool=build_tool_definition(),
        )
        persona = PersonaV1.model_validate(llm_result.tool_input)
        groundedness = check_groundedness(persona, cluster)
        if not groundedness.passed:
            raise ValueError(f"Refined persona failed groundedness: {groundedness.violations}")
        return persona.model_dump(mode="json"), llm_result.estimated_cost_usd
    except Exception:
        return seed_persona, 0.0


async def robust_synthesize(cluster: ClusterData, backend, synthesize_fn) -> object:
    attempt_temperatures = [getattr(backend, "temperature", None), 0.4, None]
    for temperature in attempt_temperatures:
        if hasattr(backend, "with_temperature"):
            current_backend = backend.with_temperature(temperature)
        else:
            current_backend = backend.__class__(
                client=backend.client,
                model=backend.model,
                temperature=temperature,
                top_p=getattr(backend, "top_p", None),
            )
        try:
            return await synthesize_fn(cluster, current_backend, max_retries=4)
        except Exception:
            continue
    raise RuntimeError(f"failed to synthesize cluster {cluster.cluster_id}")


async def run_beam_search(
    cluster: ClusterData,
    control_backend,
    search_backend,
    judge: LLMJudge,
    synthesize_fn,
    beam_width: int = 3,
    rounds: int = 2,
) -> BeamSearchResult:
    control_raw = await robust_synthesize(cluster, control_backend, synthesize_fn)
    control_persona = control_raw.persona.model_dump(mode="json")
    control_score = await judge.score_persona(control_persona)

    current_frontier: list[BeamCandidate] = []
    round_results: list[list[BeamCandidate]] = []
    total_search_cost = 0.0

    for focus in FOCUS_MODES[:beam_width]:
        candidate_raw = await robust_synthesize(cluster, search_backend, synthesize_fn)
        persona = candidate_raw.persona.model_dump(mode="json")
        score = await judge.score_persona(persona)
        candidate = BeamCandidate(
            round_id=1,
            focus=focus,
            overall_score=score.overall,
            judge_rationale=score.rationale,
            persona=persona,
            cost_usd=candidate_raw.total_cost_usd,
        )
        current_frontier.append(candidate)
        total_search_cost += candidate.cost_usd

    round_results.append(sorted(current_frontier, key=lambda item: item.overall_score, reverse=True))

    for round_id in range(2, rounds + 1):
        next_round: list[BeamCandidate] = []
        for candidate in round_results[-1][:beam_width]:
            for focus in FOCUS_MODES[:beam_width]:
                refined_persona, refined_cost = await refine_candidate(
                    cluster=cluster,
                    backend=control_backend,
                    seed_persona=candidate.persona,
                    focus=focus,
                )
                score = await judge.score_persona(refined_persona)
                next_round.append(
                    BeamCandidate(
                        round_id=round_id,
                        focus=focus,
                        overall_score=score.overall,
                        judge_rationale=score.rationale,
                        persona=refined_persona,
                        cost_usd=refined_cost,
                    )
                )
                total_search_cost += refined_cost
        current_frontier = sorted(next_round, key=lambda item: item.overall_score, reverse=True)[:beam_width]
        round_results.append(current_frontier)

    best = round_results[-1][0]
    return BeamSearchResult(
        control_overall=control_score.overall,
        control_cost_usd=control_raw.total_cost_usd,
        final_overall=best.overall_score,
        final_cost_usd=total_search_cost,
        rounds=round_results,
        cost_multiplier=(total_search_cost / control_raw.total_cost_usd) if control_raw.total_cost_usd else 0.0,
        quality_delta=best.overall_score - control_score.overall,
    )


def result_to_dict(result: BeamSearchResult) -> dict:
    return asdict(result)
