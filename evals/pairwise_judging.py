"""Experiment 5.10: Pairwise vs absolute judging."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from statistics import mean, pstdev

DIMENSIONS = (
    "grounded",
    "distinctive",
    "coherent",
    "actionable",
    "voice_fidelity",
)


@dataclass
class AbsoluteScoreRow:
    judge_model: str
    persona_id: str
    overall: float
    dimensions: dict[str, float]


@dataclass
class PairwiseRow:
    judge_model: str
    persona_a: str
    persona_b: str
    winners: dict[str, str]
    rationale: str


PAIRWISE_SYSTEM_PROMPT = """You compare two customer personas.

For each dimension and for overall quality, select exactly one of:
- "A"
- "B"
- "TIE"

Dimensions:
- grounded
- distinctive
- coherent
- actionable
- voice_fidelity

Return JSON only:
{
  "grounded": "A",
  "distinctive": "B",
  "coherent": "TIE",
  "actionable": "A",
  "voice_fidelity": "B",
  "overall": "A",
  "rationale": "short reason"
}
"""


def _parse_pairwise_response(text: str) -> tuple[dict[str, str], str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            winners = {dim: "TIE" for dim in DIMENSIONS + ("overall",)}
            return winners, cleaned[:240]
        data = json.loads(match.group())
    winners = {dim: str(data.get(dim, "TIE")).upper() for dim in DIMENSIONS + ("overall",)}
    return winners, str(data.get("rationale", ""))


def _build_pairwise_prompt(persona_a: dict, persona_b: dict) -> str:
    return (
        "Compare PERSONA A vs PERSONA B.\n\n"
        "PERSONA A:\n"
        + json.dumps(persona_a, indent=2, default=str)
        + "\n\nPERSONA B:\n"
        + json.dumps(persona_b, indent=2, default=str)
    )


async def judge_pair(backend, persona_a: dict, persona_b: dict) -> tuple[dict[str, str], str]:
    response = await backend.score(
        system=PAIRWISE_SYSTEM_PROMPT,
        prompt=_build_pairwise_prompt(persona_a, persona_b),
    )
    return _parse_pairwise_response(response)


def _normalize_swapped(result: dict[str, str]) -> dict[str, str]:
    normalized = {}
    for key, value in result.items():
        if value == "A":
            normalized[key] = "B"
        elif value == "B":
            normalized[key] = "A"
        else:
            normalized[key] = "TIE"
    return normalized


async def judge_pair_bidirectional(backend, persona_a: dict, persona_b: dict) -> tuple[dict[str, str], str]:
    ab, rationale_ab = await judge_pair(backend, persona_a, persona_b)
    ba_raw, rationale_ba = await judge_pair(backend, persona_b, persona_a)
    ba = _normalize_swapped(ba_raw)

    combined = {}
    for key in DIMENSIONS + ("overall",):
        combined[key] = ab[key] if ab[key] == ba[key] else "TIE"
    return combined, f"AB: {rationale_ab} | BA: {rationale_ba}"


def _rank(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    return {persona_id: index + 1 for index, (persona_id, _) in enumerate(ordered)}


def _spearman(rank_a: dict[str, int], rank_b: dict[str, int]) -> float:
    persona_ids = sorted(rank_a)
    n = len(persona_ids)
    if n < 2:
        return float("nan")
    diffs_sq = sum((rank_a[persona_id] - rank_b[persona_id]) ** 2 for persona_id in persona_ids)
    return 1 - (6 * diffs_sq) / (n * (n**2 - 1))


def absolute_rankings(rows: list[AbsoluteScoreRow]) -> dict[str, dict[str, dict[str, int]]]:
    rankings: dict[str, dict[str, dict[str, int]]] = {}
    for model in sorted({row.judge_model for row in rows}):
        model_rows = [row for row in rows if row.judge_model == model]
        rankings[model] = {}
        for dim in DIMENSIONS:
            rankings[model][dim] = _rank({row.persona_id: row.dimensions[dim] for row in model_rows})
        rankings[model]["overall"] = _rank({row.persona_id: row.overall for row in model_rows})
    return rankings


def pairwise_rankings(rows: list[PairwiseRow]) -> dict[str, dict[str, dict[str, int]]]:
    rankings: dict[str, dict[str, dict[str, int]]] = {}
    for model in sorted({row.judge_model for row in rows}):
        tallies = {
            dimension: {}
            for dimension in DIMENSIONS + ("overall",)
        }
        for row in [pair for pair in rows if pair.judge_model == model]:
            for dimension, winner in row.winners.items():
                tallies[dimension].setdefault(row.persona_a, 0.0)
                tallies[dimension].setdefault(row.persona_b, 0.0)
                if winner == "A":
                    tallies[dimension][row.persona_a] += 1.0
                elif winner == "B":
                    tallies[dimension][row.persona_b] += 1.0
                else:
                    tallies[dimension][row.persona_a] += 0.5
                    tallies[dimension][row.persona_b] += 0.5
        rankings[model] = {dimension: _rank(scores) for dimension, scores in tallies.items()}
    return rankings


def agreement_report(rankings: dict[str, dict[str, dict[str, int]]]) -> dict[str, float]:
    models = sorted(rankings)
    if len(models) != 2:
        raise ValueError("agreement_report expects exactly two judge models")
    model_a, model_b = models
    report = {}
    for dimension in DIMENSIONS + ("overall",):
        report[dimension] = _spearman(rankings[model_a][dimension], rankings[model_b][dimension])
    return report


def distribution_tightness(rows: list[AbsoluteScoreRow], pair_rows: list[PairwiseRow]) -> dict[str, float]:
    absolute_overalls = [row.overall for row in rows]
    pairwise_overalls = []
    for model in sorted({row.judge_model for row in pair_rows}):
        model_rankings = pairwise_rankings([row for row in pair_rows if row.judge_model == model])[model]["overall"]
        pairwise_overalls.extend(model_rankings.values())
    return {
        "absolute_overall_stddev": pstdev(absolute_overalls) if len(absolute_overalls) > 1 else 0.0,
        "pairwise_rank_stddev": pstdev(pairwise_overalls) if len(pairwise_overalls) > 1 else 0.0,
    }


def summarize(
    absolute_rows: list[AbsoluteScoreRow],
    pairwise_rows: list[PairwiseRow],
) -> dict:
    absolute_agreement = agreement_report(absolute_rankings(absolute_rows))
    pairwise_agreement = agreement_report(pairwise_rankings(pairwise_rows))
    return {
        "absolute_agreement": absolute_agreement,
        "pairwise_agreement": pairwise_agreement,
        "mean_absolute_agreement": mean(
            value for value in absolute_agreement.values() if not math.isnan(value)
        ),
        "mean_pairwise_agreement": mean(
            value for value in pairwise_agreement.values() if not math.isnan(value)
        ),
        "distribution_tightness": distribution_tightness(absolute_rows, pairwise_rows),
    }


def results_to_dict(absolute_rows: list[AbsoluteScoreRow], pairwise_rows: list[PairwiseRow], summary: dict) -> dict:
    return {
        "absolute_rows": [asdict(row) for row in absolute_rows],
        "pairwise_rows": [asdict(row) for row in pairwise_rows],
        "summary": summary,
    }
