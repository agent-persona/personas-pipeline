"""Experiment 2.11: Multi-agent debate for persona synthesis.

Three-way loop:
  1. **Synthesizer** generates a persona (standard pipeline).
  2. **Adversary** challenges each evidenced claim, looking for hallucinations,
     unsupported inferences, or over-confident claims.
  3. **Judge** resolves each challenge: keep, revise, or drop the claim.
  4. If any claims are revised/dropped, re-synthesize with judge feedback.

The adversary and judge are separate LLM calls with adversarial/arbitration
system prompts. This adds cost but should improve groundedness and reduce
hallucination.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from typing import Sequence

from anthropic import AsyncAnthropic

from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1

from .groundedness import GroundednessReport, check_groundedness
from .model_backend import AnthropicBackend, LLMResult
from .prompt_builder import SYSTEM_PROMPT, build_messages, build_tool_definition
from .synthesizer import SynthesisResult, synthesize

logger = logging.getLogger(__name__)


# ── Prompts ──────────────────────────────────────────────────────────

ADVERSARY_SYSTEM_PROMPT = """\
You are a skeptical fact-checker reviewing a synthesized marketing persona. \
Your job is to challenge every claim that is not directly and strongly \
supported by the source data.

For each source_evidence entry in the persona, evaluate:
1. Does the claim match what the cited records actually show?
2. Is the confidence score justified, or is the synthesizer over-confident?
3. Is the claim a reasonable inference from the data, or a hallucination?

Be adversarial but fair. A claim supported by multiple records with clear \
behavioral signals should be kept. A claim citing one vague record with \
high confidence should be challenged.

Respond with ONLY a JSON array of challenge objects:
[
  {
    "field_path": "<the evidence field_path being challenged>",
    "claim": "<the claim text>",
    "challenge": "<why this claim is suspect>",
    "severity": "<high|medium|low>"
  }
]

If all claims are well-supported, return an empty array: []
Do NOT wrap in markdown fences.
"""

JUDGE_SYSTEM_PROMPT = """\
You are an impartial arbiter resolving disputes between a persona synthesizer \
and a fact-checking adversary. For each challenged claim, you have:
- The original claim and its evidence (record IDs, confidence)
- The adversary's challenge explaining why it's suspect

Your job is to issue a verdict for each challenge:
- **keep**: The claim is adequately supported. The adversary is being too strict.
- **revise**: The claim has merit but needs qualification (lower confidence, \
  softer language, or additional evidence).
- **drop**: The claim is not supported by the data and should be removed.

Respond with ONLY a JSON array of verdict objects:
[
  {
    "field_path": "<field_path of the challenged claim>",
    "verdict": "<keep|revise|drop>",
    "reasoning": "<brief explanation>"
  }
]

Do NOT wrap in markdown fences.
"""


# ── Data types ───────────────────────────────────────────────────────

@dataclass
class Challenge:
    field_path: str
    claim: str
    challenge: str
    severity: str  # high, medium, low


@dataclass
class Verdict:
    field_path: str
    verdict: str  # keep, revise, drop
    reasoning: str


@dataclass
class DebateRound:
    challenges: list[Challenge]
    verdicts: list[Verdict]
    kept: int = 0
    revised: int = 0
    dropped: int = 0
    adversary_cost_usd: float = 0.0
    judge_cost_usd: float = 0.0


@dataclass
class DebateResult:
    """Full result of a debate-enhanced synthesis."""
    persona: PersonaV1
    groundedness: GroundednessReport
    total_cost_usd: float
    model_used: str
    synthesis_attempts: int
    debate_rounds: list[DebateRound]
    total_challenges: int
    total_kept: int
    total_revised: int
    total_dropped: int


# ── Adversary ────────────────────────────────────────────────────────

async def run_adversary(
    client: AsyncAnthropic,
    model: str,
    persona_dict: dict,
    cluster_data_summary: str,
) -> tuple[list[Challenge], float]:
    """Have the adversary challenge claims in the persona."""
    user_msg = (
        "## Persona to review\n"
        + json.dumps(persona_dict, indent=2)
        + "\n\n## Source data summary\n"
        + cluster_data_summary
    )

    response = await client.messages.create(
        model=model,
        max_tokens=2048,
        system=ADVERSARY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

    in_tok = response.usage.input_tokens
    out_tok = response.usage.output_tokens
    if "opus" in model:
        cost = (in_tok * 15 + out_tok * 75) / 1_000_000
    elif "haiku" in model:
        cost = (in_tok * 1 + out_tok * 5) / 1_000_000
    else:
        cost = (in_tok * 3 + out_tok * 15) / 1_000_000

    parsed = json.loads(text)
    challenges = [
        Challenge(
            field_path=c.get("field_path", ""),
            claim=c.get("claim", ""),
            challenge=c.get("challenge", ""),
            severity=c.get("severity", "medium"),
        )
        for c in parsed
    ]
    return challenges, cost


# ── Judge ────────────────────────────────────────────────────────────

async def run_judge(
    client: AsyncAnthropic,
    model: str,
    challenges: list[Challenge],
    persona_dict: dict,
) -> tuple[list[Verdict], float]:
    """Have the judge resolve adversary challenges."""
    if not challenges:
        return [], 0.0

    # Build the dispute context
    evidence = persona_dict.get("source_evidence", [])
    evidence_by_path = {e.get("field_path", ""): e for e in evidence}

    disputes = []
    for c in challenges:
        ev = evidence_by_path.get(c.field_path, {})
        disputes.append({
            "field_path": c.field_path,
            "claim": c.claim,
            "evidence_record_ids": ev.get("record_ids", []),
            "evidence_confidence": ev.get("confidence", 0),
            "adversary_challenge": c.challenge,
            "adversary_severity": c.severity,
        })

    user_msg = json.dumps(disputes, indent=2)

    response = await client.messages.create(
        model=model,
        max_tokens=2048,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

    in_tok = response.usage.input_tokens
    out_tok = response.usage.output_tokens
    if "opus" in model:
        cost = (in_tok * 15 + out_tok * 75) / 1_000_000
    elif "haiku" in model:
        cost = (in_tok * 1 + out_tok * 5) / 1_000_000
    else:
        cost = (in_tok * 3 + out_tok * 15) / 1_000_000

    parsed = json.loads(text)
    verdicts = [
        Verdict(
            field_path=v.get("field_path", ""),
            verdict=v.get("verdict", "keep"),
            reasoning=v.get("reasoning", ""),
        )
        for v in parsed
    ]
    return verdicts, cost


# ── Debate orchestrator ──────────────────────────────────────────────

def _build_cluster_summary(cluster: ClusterData) -> str:
    """Short text summary of the cluster for adversary context."""
    lines = [
        f"Cluster: {cluster.cluster_id}",
        f"Size: {cluster.summary.cluster_size} records",
        f"Behaviors: {', '.join(cluster.summary.top_behaviors)}",
        f"Pages: {', '.join(cluster.summary.top_pages)}",
        f"Record IDs: {', '.join(cluster.all_record_ids)}",
    ]
    return "\n".join(lines)


def _apply_verdicts(
    persona_dict: dict,
    verdicts: list[Verdict],
) -> tuple[dict, int, int, int]:
    """Apply judge verdicts to a persona dict.

    For 'drop' verdicts: remove the evidence entry and lower confidence.
    For 'revise' verdicts: lower the confidence on that evidence entry.
    For 'keep' verdicts: no change.

    Returns (modified_dict, kept, revised, dropped).
    """
    kept = revised = dropped = 0
    verdict_map = {v.field_path: v.verdict for v in verdicts}

    evidence = persona_dict.get("source_evidence", [])
    new_evidence = []
    for ev in evidence:
        fp = ev.get("field_path", "")
        v = verdict_map.get(fp, "keep")
        if v == "drop":
            dropped += 1
            continue  # remove this evidence entry
        elif v == "revise":
            revised += 1
            ev = dict(ev)
            ev["confidence"] = max(0.3, ev.get("confidence", 0.5) - 0.2)
        else:
            kept += 1
        new_evidence.append(ev)

    result = dict(persona_dict)
    result["source_evidence"] = new_evidence
    return result, kept, revised, dropped


async def debate_synthesize(
    cluster: ClusterData,
    client: AsyncAnthropic,
    synth_model: str,
    adversary_model: str | None = None,
    judge_model: str | None = None,
    max_debate_rounds: int = 1,
) -> DebateResult:
    """Synthesize a persona with multi-agent debate refinement.

    1. Standard synthesis (synthesizer)
    2. Adversary challenges claims
    3. Judge resolves challenges
    4. Apply verdicts and optionally re-synthesize

    Args:
        adversary_model: Model for the adversary. Defaults to synth_model.
        judge_model: Model for the judge. Defaults to synth_model.
        max_debate_rounds: Number of adversary→judge rounds (default 1).
    """
    adversary_model = adversary_model or synth_model
    judge_model = judge_model or synth_model

    # Step 1: Standard synthesis
    backend = AnthropicBackend(client=client, model=synth_model)
    synth_result = await synthesize(cluster, backend)

    persona_dict = synth_result.persona.model_dump(mode="json")
    cluster_summary = _build_cluster_summary(cluster)
    total_cost = synth_result.total_cost_usd
    debate_rounds: list[DebateRound] = []
    total_challenges = 0
    total_kept = total_revised = total_dropped = 0

    # Step 2-3: Debate rounds
    for round_num in range(max_debate_rounds):
        # Adversary challenges
        try:
            challenges, adv_cost = await run_adversary(
                client, adversary_model, persona_dict, cluster_summary,
            )
        except Exception as e:
            logger.warning("Adversary failed in round %d: %s", round_num + 1, e)
            challenges, adv_cost = [], 0.0

        total_cost += adv_cost

        if not challenges:
            debate_rounds.append(DebateRound(
                challenges=[], verdicts=[],
                adversary_cost_usd=adv_cost,
            ))
            break  # No challenges = adversary approves

        # Judge resolves
        try:
            verdicts, judge_cost = await run_judge(
                client, judge_model, challenges, persona_dict,
            )
        except Exception as e:
            logger.warning("Judge failed in round %d: %s", round_num + 1, e)
            verdicts, judge_cost = [], 0.0

        total_cost += judge_cost

        # Apply verdicts
        persona_dict, kept, revised, dropped = _apply_verdicts(
            persona_dict, verdicts,
        )

        dr = DebateRound(
            challenges=challenges,
            verdicts=verdicts,
            kept=kept,
            revised=revised,
            dropped=dropped,
            adversary_cost_usd=adv_cost,
            judge_cost_usd=judge_cost,
        )
        debate_rounds.append(dr)
        total_challenges += len(challenges)
        total_kept += kept
        total_revised += revised
        total_dropped += dropped

    # Re-validate the final persona
    final_persona = PersonaV1.model_validate(persona_dict)
    final_groundedness = check_groundedness(final_persona, cluster)

    return DebateResult(
        persona=final_persona,
        groundedness=final_groundedness,
        total_cost_usd=total_cost,
        model_used=synth_model,
        synthesis_attempts=synth_result.attempts,
        debate_rounds=debate_rounds,
        total_challenges=total_challenges,
        total_kept=total_kept,
        total_revised=total_revised,
        total_dropped=total_dropped,
    )
