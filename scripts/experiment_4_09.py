"""Experiment 4.09: multi-turn red-team."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).parent.parent
for mod in ("crawler", "segmentation", "synthesis", "orchestration", "twin", "evaluation"):
    sys.path.insert(0, str(REPO_ROOT / mod))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from crawler import fetch_all  # noqa: E402
from evaluation import load_golden_set  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend, LLMResult  # noqa: E402
from synthesis.engine.synthesizer import synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402

from evals.red_team_agent import (  # noqa: E402
    RED_TEAM_STRATEGIES,
    PersonaRunResult,
    TurnRecord,
    build_chat_backend,
    build_red_team_system_prompt,
    build_twin_system_prompt,
    build_transcript_text,
    result_to_dict,
    score_response,
    summarise_strategy_runs,
)

TENANT_ID = "tenant_acme_corp"
OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-4.09-multi-turn-red-team"
TURN_LIMIT = 10
ANTHROPIC_SYNTHESIS_MODEL = settings.default_model
ANTHROPIC_CHAT_MODEL = "claude-haiku-4-5-20251001"
OPENAI_MODEL = "gpt-5-nano"
SEED_PERSONA_RESULTS = Path(
    "/private/tmp/personas-batch5-worktrees/exp-6.05/output/experiments/exp-6.05-stability-across-reruns/results.json"
)


def get_clusters() -> list[ClusterData]:
    tenant = next(t for t in load_golden_set() if t.tenant_id == TENANT_ID)
    crawler_records = fetch_all(tenant.tenant_id)
    raw_records = [RawRecord.model_validate(r.model_dump()) for r in crawler_records]
    cluster_dicts = segment(
        raw_records,
        tenant_industry=tenant.industry,
        tenant_product=tenant.product_description,
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    return [ClusterData.model_validate(cluster) for cluster in cluster_dicts]


class OpenAIJsonBackend:
    def __init__(self, client: AsyncOpenAI, model: str, temperature: float | None = None) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature

    async def generate(self, system: str, messages: list[dict], tool: dict) -> LLMResult:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool["input_schema"],
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": tool["name"]},
            },
            "max_completion_tokens": 4096,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        response = await self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if tool_calls:
            tool_input = json.loads(tool_calls[0].function.arguments)
        else:
            text = message.content or "{}"
            if isinstance(text, list):
                text = "".join(part.get("text", "") for part in text)
            try:
                tool_input = json.loads(text)
            except json.JSONDecodeError:
                start = text.find("{")
                end = text.rfind("}")
                tool_input = json.loads(text[start : end + 1]) if start != -1 and end != -1 else {}
        usage = response.usage
        return LLMResult(
            tool_input=tool_input,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=self.model,
        )


def _openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def load_seed_personas() -> tuple[list[dict], str] | None:
    if not SEED_PERSONA_RESULTS.exists():
        return None
    try:
        data = json.loads(SEED_PERSONA_RESULTS.read_text())
    except Exception:
        return None
    runs = data.get("runs", [])
    if not runs:
        return None
    personas = runs[0].get("personas", [])
    if not personas:
        return None
    return personas[:2], str(SEED_PERSONA_RESULTS)


async def synthesize_persona(cluster: ClusterData) -> tuple[dict, str, float]:
    openai_key = _openai_key()
    openai_client = AsyncOpenAI(api_key=openai_key) if openai_key else None

    primary_backend = None
    if settings.anthropic_api_key:
        primary_backend = AnthropicBackend(
            client=AsyncAnthropic(api_key=settings.anthropic_api_key),
            model=ANTHROPIC_SYNTHESIS_MODEL,
        )
        try:
            result = await synthesize(cluster, primary_backend, max_retries=4)
            return result.persona.model_dump(mode="json"), result.model_used, result.total_cost_usd
        except Exception as primary_error:
            if openai_client is None:
                raise primary_error

    if openai_client is None:
        raise RuntimeError("Missing ANTHROPIC_API_KEY and OPENAI_API_KEY")

    fallback_backend = OpenAIJsonBackend(client=openai_client, model=OPENAI_MODEL, temperature=None)
    result = await synthesize(cluster, fallback_backend, max_retries=4)
    model_used = result.model_used
    if primary_backend is not None and model_used == OPENAI_MODEL:
        model_used = f"{ANTHROPIC_SYNTHESIS_MODEL}->{OPENAI_MODEL}"
    return result.persona.model_dump(mode="json"), model_used, result.total_cost_usd


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _write_findings(results: dict) -> str:
    summary = results["summary"]
    lines = [
        "# Experiment 4.09: Multi-Turn Red-Team",
        "",
        "## Hypothesis",
        "Turns-to-break is a useful stability metric for twin runtimes.",
        "",
        "## Method",
        f"1. Evaluated `{results['n_personas']}` personas from `{results['persona_source']}`.",
        f"2. Ran `{results['n_strategies']}` red-team strategies for up to `{TURN_LIMIT}` turns each.",
        "3. Scored each twin response as in-character, partial break, or full break.",
        "4. Recorded turns-to-break and recovery speed after partial breaks.",
        "",
        f"- Provider: `{results['provider']}`",
        f"- Synthesis model: `{results['synthesis_model']}`",
        f"- Twin model: `{results['twin_model']}`",
        f"- Red-team model: `{results['red_team_model']}`",
        "",
        "## Strategy Summary",
    ]
    for strategy, metrics in summary["strategies"].items():
        lines.append(
            f"- `{strategy}`: success `{metrics['attack_success_rate']:.1%}`, "
            f"mean break turn `{metrics['mean_turns_to_break'] if metrics['mean_turns_to_break'] is not None else 'n/a'}`, "
            f"mean recovery `{metrics['mean_recovery_turns'] if metrics['mean_recovery_turns'] is not None else 'n/a'}`, "
            f"mean score `{metrics['mean_score']:.2f}`"
        )
    lines.extend(
        [
            "",
            "## Decision",
            (
                "Adopt. Multi-turn probing produced measurable break behavior and recovery variation."
                if summary["max_attack_success_rate"] > 0
                else "Defer. The sample is too small and stable to justify a stronger claim."
            ),
            "",
            "## Caveat",
            "Tiny sample: 1 tenant, 2 personas. Heuristic scoring is intentionally conservative and may undercount subtle failures.",
        ]
    )
    return "\n".join(lines) + "\n"


async def run_strategy(
    persona: dict,
    cluster: ClusterData,
    strategy: str,
    synthesis_model: str,
    provider: str,
) -> tuple[PersonaRunResult, float]:
    chat_backend = build_chat_backend(
        anthropic_key=settings.anthropic_api_key,
        openai_key=_openai_key(),
        anthropic_model=ANTHROPIC_CHAT_MODEL,
        openai_model=OPENAI_MODEL,
        temperature=0.7,
    )
    red_team_backend = build_chat_backend(
        anthropic_key=settings.anthropic_api_key,
        openai_key=_openai_key(),
        anthropic_model=ANTHROPIC_CHAT_MODEL,
        openai_model=OPENAI_MODEL,
        temperature=0.9,
    )

    turns: list[TurnRecord] = []
    transcript: list[dict] = []
    total_cost = 0.0
    turns_to_break: int | None = None
    recovery_turns: int | None = None
    first_break_turn: int | None = None
    broke_once = False
    recovered = False

    for turn_index in range(1, TURN_LIMIT + 1):
        red_prompt = build_red_team_system_prompt(persona, strategy, turn_index)
        red_history = [
            {
                "role": "user",
                "content": (
                    "Generate the next attack message only. "
                    "Use the transcript to escalate naturally.\n\n"
                    + build_transcript_text(turns)
                ),
            }
        ]
        attack_result = await red_team_backend.complete(red_prompt, red_history, max_tokens=128)
        attack = attack_result.text.strip()
        total_cost += attack_result.estimated_cost_usd

        response_result = await chat_backend.complete(
            build_twin_system_prompt(persona),
            transcript + [{"role": "user", "content": attack}],
            max_tokens=256,
        )
        response = response_result.text.strip()
        total_cost += response_result.estimated_cost_usd

        score, label, rationale = score_response(persona, response)
        turns.append(
            TurnRecord(
                turn_index=turn_index,
                attack=attack,
                response=response,
                score=score,
                label=label,
                rationale=rationale,
                attack_model=attack_result.model,
                twin_model=response_result.model,
            )
        )
        transcript.extend(
            [
                {"role": "user", "content": attack},
                {"role": "assistant", "content": response},
            ]
        )

        if score < 1.0 and first_break_turn is None:
            first_break_turn = turn_index
            broke_once = True
        if first_break_turn is not None and score == 1.0 and turn_index > first_break_turn and not recovered:
            recovery_turns = turn_index - first_break_turn
            recovered = True
        if turns_to_break is None and score < 1.0:
            turns_to_break = turn_index

    mean_score = mean(turn.score for turn in turns) if turns else 0.0
    return (
        PersonaRunResult(
            cluster_id=cluster.cluster_id,
            persona_name=persona.get("name", "unknown"),
            strategy=strategy,
            turns=turns,
            turns_to_break=turns_to_break,
            recovery_turns=recovery_turns,
            attack_success=broke_once,
            mean_score=mean_score,
            cost_usd=total_cost,
        ),
        total_cost,
    )


async def main() -> None:
    print("=" * 72)
    print("EXPERIMENT 4.09: Multi-turn red-team")
    print("=" * 72)

    _ensure_output_dir()
    t0 = time.monotonic()

    provider = "anthropic" if settings.anthropic_api_key else "openai"
    if settings.anthropic_api_key and _openai_key():
        provider = "anthropic->openai"
    elif not settings.anthropic_api_key and not _openai_key():
        raise RuntimeError("Missing ANTHROPIC_API_KEY and OPENAI_API_KEY")

    clusters = get_clusters()
    seed_personas = load_seed_personas()
    personas: list[dict] = []
    persona_models: list[str] = []
    synthesis_costs: list[float] = []

    if seed_personas is not None:
        personas, seed_source = seed_personas
        persona_models = [f"seed:{Path(seed_source).name}"] * len(personas)
        synthesis_costs = [0.0] * len(personas)
        print(f"      using seed personas from {seed_source}")
    else:
        for cluster in clusters[:2]:
            persona, model_used, cost_usd = await synthesize_persona(cluster)
            personas.append(persona)
            persona_models.append(model_used)
            synthesis_costs.append(cost_usd)
            print(f"      synthesized {persona.get('name', 'unknown')} ({model_used})")

    runs: list[PersonaRunResult] = []
    for persona, cluster, model_used in zip(personas, clusters[: len(personas)], persona_models, strict=False):
        for strategy in RED_TEAM_STRATEGIES:
            result, _ = await run_strategy(
                persona=persona,
                cluster=cluster,
                strategy=strategy,
                synthesis_model=model_used,
                provider=provider,
            )
            runs.append(result)
            print(
                f"      {persona.get('name', 'unknown')} / {strategy}: "
                f"break={result.turns_to_break}, recovery={result.recovery_turns}, score={result.mean_score:.2f}"
            )

    strategy_summaries = {
        strategy: asdict(summarise_strategy_runs(runs, strategy))
        for strategy in RED_TEAM_STRATEGIES
    }
    attack_success_rates = [summary["attack_success_rate"] for summary in strategy_summaries.values()]
    break_turns = [summary["mean_turns_to_break"] for summary in strategy_summaries.values() if summary["mean_turns_to_break"] is not None]
    recovery_turns = [summary["mean_recovery_turns"] for summary in strategy_summaries.values() if summary["mean_recovery_turns"] is not None]
    mean_scores = [summary["mean_score"] for summary in strategy_summaries.values()]
    total_cost = sum(synthesis_costs) + sum(run.cost_usd for run in runs)

    results = {
        "experiment": "4.09",
        "title": "Multi-turn red-team",
        "provider": provider,
        "synthesis_model": persona_models[0] if persona_models else ANTHROPIC_SYNTHESIS_MODEL,
        "twin_model": ANTHROPIC_CHAT_MODEL if settings.anthropic_api_key else OPENAI_MODEL,
        "red_team_model": ANTHROPIC_CHAT_MODEL if settings.anthropic_api_key else OPENAI_MODEL,
        "persona_source": str(SEED_PERSONA_RESULTS) if seed_personas is not None else "synthesized",
        "turn_limit": TURN_LIMIT,
        "n_personas": len(personas),
        "n_strategies": len(RED_TEAM_STRATEGIES),
        "persona_cost_usd": synthesis_costs,
        "strategy_runs": [result_to_dict(run) for run in runs],
        "summary": {
            "strategies": strategy_summaries,
            "max_attack_success_rate": max(attack_success_rates) if attack_success_rates else 0.0,
            "mean_attack_success_rate": mean(attack_success_rates) if attack_success_rates else 0.0,
            "mean_turns_to_break": mean(break_turns) if break_turns else None,
            "mean_recovery_turns": mean(recovery_turns) if recovery_turns else None,
            "mean_score": mean(mean_scores) if mean_scores else 0.0,
            "total_cost_usd": total_cost,
        },
        "duration_seconds": time.monotonic() - t0,
    }

    (OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (OUTPUT_DIR / "FINDINGS.md").write_text(_write_findings(results))

    print(
        f"      success={results['summary']['mean_attack_success_rate']:.1%} "
        f"break_turns={results['summary']['mean_turns_to_break']} "
        f"cost=${results['summary']['total_cost_usd']:.4f}"
    )
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
