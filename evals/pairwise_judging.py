"""Pairwise persona judging for experiment 5.10.

Fills the `pairwise()` stub from evaluation/judges.py as a LOCAL HELPER.
Team rule: do not modify shared module defaults. This is an experiment-local
implementation so 5.10 can compare pairwise vs absolute scoring without
blocking the 5.13 owner.

Three pieces:
  - `pairwise_prompt` builder — dimension-aware comparison prompt
  - `pairwise_judge` — runs both orderings (a,b) and (b,a) to debias position
  - `win_count_ranking` — converts pairwise results to per-persona ranks
  - `bradley_terry_ranking` — a more principled ranker for when win counts tie
"""
from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass, field

from anthropic import AsyncAnthropic


DIMENSIONS = ("grounded", "distinctive", "coherent", "actionable", "voice_fidelity")


_PAIRWISE_SYSTEM = """\
You are an expert persona evaluator comparing two synthesized customer personas
side by side. You pick the better one on each quality dimension.

Dimensions:
- **grounded**: Are claims traceable to source evidence?
- **distinctive**: Does this feel like a real individual, not a generic average?
- **coherent**: Are demographics, vocabulary, and quotes internally consistent?
- **actionable**: Are goals/pains specific enough to drive product decisions?
- **voice_fidelity**: Do sample quotes sound like one consistent speaker?

For each dimension, pick A, B, or TIE. Only call a TIE if the two personas are
genuinely indistinguishable on that dimension.

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{
  "grounded": "A" | "B" | "TIE",
  "distinctive": "A" | "B" | "TIE",
  "coherent": "A" | "B" | "TIE",
  "actionable": "A" | "B" | "TIE",
  "voice_fidelity": "A" | "B" | "TIE",
  "rationale": "<brief justification>"
}
"""


def _persona_brief(p: dict) -> str:
    """Compress a persona to the fields the judge actually needs."""
    fields = ["name", "summary", "goals", "pains", "motivations", "objections",
              "vocabulary", "sample_quotes"]
    compact = {k: p.get(k) for k in fields if k in p}
    return json.dumps(compact, indent=2)


def build_pairwise_prompt(persona_a: dict, persona_b: dict) -> str:
    return f"""Compare these two personas.

=== PERSONA A ===
{_persona_brief(persona_a)}

=== PERSONA B ===
{_persona_brief(persona_b)}

Pick A, B, or TIE on each dimension. Return JSON only."""


@dataclass
class PairwiseVerdict:
    """One pair comparison result, with both orderings averaged."""
    persona_a_id: str
    persona_b_id: str
    # per-dimension: "a" | "b" | "tie"
    ab_order: dict[str, str] = field(default_factory=dict)
    ba_order: dict[str, str] = field(default_factory=dict)
    # debiased result per dimension (counts agreements; disagreement = tie)
    debiased: dict[str, str] = field(default_factory=dict)
    rationale_ab: str = ""
    rationale_ba: str = ""


def _parse_verdict(raw: str, dims: tuple[str, ...] = DIMENSIONS) -> dict:
    """Extract a dimension -> A/B/TIE dict from a judge response."""
    # Try to find a JSON block
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return {d: "TIE" for d in dims}
    try:
        data = json.loads(m.group(0))
    except Exception:
        return {d: "TIE" for d in dims}
    out = {}
    for d in dims:
        v = str(data.get(d, "TIE")).strip().upper()
        if v not in {"A", "B", "TIE"}:
            v = "TIE"
        out[d] = v
    out["_rationale"] = data.get("rationale", "")
    return out


async def pairwise_judge(
    client: AsyncAnthropic,
    model: str,
    persona_a: dict,
    persona_b: dict,
    persona_a_id: str,
    persona_b_id: str,
) -> PairwiseVerdict:
    """Run pairwise comparison in BOTH orderings to debias position."""
    prompt_ab = build_pairwise_prompt(persona_a, persona_b)
    prompt_ba = build_pairwise_prompt(persona_b, persona_a)

    resp_ab = await client.messages.create(
        model=model, max_tokens=1024, system=_PAIRWISE_SYSTEM,
        messages=[{"role": "user", "content": prompt_ab}],
    )
    resp_ba = await client.messages.create(
        model=model, max_tokens=1024, system=_PAIRWISE_SYSTEM,
        messages=[{"role": "user", "content": prompt_ba}],
    )

    ab_raw = resp_ab.content[0].text
    ba_raw = resp_ba.content[0].text

    ab_parsed = _parse_verdict(ab_raw)
    ba_parsed = _parse_verdict(ba_raw)

    verdict = PairwiseVerdict(
        persona_a_id=persona_a_id, persona_b_id=persona_b_id,
        ab_order={d: ab_parsed[d].lower() for d in DIMENSIONS},
        ba_order={d: ba_parsed[d].lower() for d in DIMENSIONS},
        rationale_ab=ab_parsed.get("_rationale", ""),
        rationale_ba=ba_parsed.get("_rationale", ""),
    )

    # Debias: flip ba_order (in second run, A and B were swapped so "a" there means our b)
    flipped_ba = {}
    for d in DIMENSIONS:
        v = verdict.ba_order[d]
        if v == "a":
            flipped_ba[d] = "b"
        elif v == "b":
            flipped_ba[d] = "a"
        else:
            flipped_ba[d] = "tie"

    # Final verdict: agree across orderings -> that winner. Disagree -> tie (position bias).
    for d in DIMENSIONS:
        if verdict.ab_order[d] == flipped_ba[d]:
            verdict.debiased[d] = verdict.ab_order[d]
        else:
            verdict.debiased[d] = "tie"

    return verdict


def win_count_ranking(
    persona_ids: list[str],
    verdicts: list[PairwiseVerdict],
    dimension: str,
) -> dict[str, float]:
    """Convert pairwise verdicts to per-persona win counts for one dimension.
    Returns {persona_id: win_count}. Ties add 0.5 to each side."""
    scores = {pid: 0.0 for pid in persona_ids}
    for v in verdicts:
        winner = v.debiased[dimension]
        if winner == "a":
            scores[v.persona_a_id] += 1
        elif winner == "b":
            scores[v.persona_b_id] += 1
        else:
            scores[v.persona_a_id] += 0.5
            scores[v.persona_b_id] += 0.5
    return scores


def bradley_terry_ranking(
    persona_ids: list[str],
    verdicts: list[PairwiseVerdict],
    dimension: str,
    iterations: int = 100,
) -> dict[str, float]:
    """Simple iterative Bradley-Terry ranker.

    BT solves for per-persona strength parameters p_i such that
    P(i beats j) = p_i / (p_i + p_j).
    Initialized uniform, iterated with the standard MM update.
    """
    n = len(persona_ids)
    idx = {pid: i for i, pid in enumerate(persona_ids)}

    # wins[i][j] = number of times i beat j
    wins = [[0.0] * n for _ in range(n)]
    for v in verdicts:
        ia = idx[v.persona_a_id]
        ib = idx[v.persona_b_id]
        w = v.debiased[dimension]
        if w == "a":
            wins[ia][ib] += 1
        elif w == "b":
            wins[ib][ia] += 1
        else:
            wins[ia][ib] += 0.5
            wins[ib][ia] += 0.5

    # Initial strengths uniform
    p = [1.0] * n

    for _ in range(iterations):
        new_p = [0.0] * n
        for i in range(n):
            num = sum(wins[i][j] for j in range(n) if j != i)
            den = sum(
                (wins[i][j] + wins[j][i]) / (p[i] + p[j])
                for j in range(n) if j != i
            )
            new_p[i] = num / den if den > 0 else p[i]
        # Normalize to avoid runaway
        total = sum(new_p)
        if total > 0:
            p = [x / total * n for x in new_p]
        else:
            break

    return {persona_ids[i]: round(p[i], 4) for i in range(n)}


def aggregate_dimensions(
    persona_ids: list[str],
    ranking_fn,
    verdicts: list[PairwiseVerdict],
    dimensions: tuple[str, ...] = DIMENSIONS,
) -> dict[str, float]:
    """Average rank across all dimensions -> one overall score per persona."""
    per_dim = {d: ranking_fn(persona_ids, verdicts, d) for d in dimensions}
    overall = {}
    for pid in persona_ids:
        overall[pid] = sum(per_dim[d][pid] for d in dimensions) / len(dimensions)
    return overall
