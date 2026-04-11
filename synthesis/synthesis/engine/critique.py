"""Experiment 2.02: Critique and reflexion loops.

Adds an optional critic pass after successful synthesis. The critic scores
the persona on the standard dimensions and provides specific improvement
suggestions. The synthesizer then re-generates with the critic's feedback
injected as context (reflexion).

Parameterized by revision_rounds (0-3):
  - 0 rounds: standard single-pass (control)
  - 1 round:  synthesize -> critic -> re-synthesize with feedback
  - 2 rounds: two critic-revision cycles
  - 3 rounds: three cycles (hypothesis: overfits and kills distinctiveness)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from anthropic import AsyncAnthropic

from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1

from .groundedness import GroundednessReport, check_groundedness
from .model_backend import AnthropicBackend
from .prompt_builder import SYSTEM_PROMPT, build_tool_definition, build_user_message
from .synthesizer import SynthesisResult, synthesize

logger = logging.getLogger(__name__)


# ── Critic prompt ────────────────────────────────────────────────────

CRITIC_SYSTEM_PROMPT = """\
You are a persona quality critic. Score the persona below on five dimensions \
(0.0-1.0) and provide specific, actionable improvement suggestions.

## Dimensions
- **grounded**: Claims traceable to source evidence with valid record IDs.
- **distinctive**: Feels like a real individual, not generic boilerplate.
- **coherent**: Internal consistency across demographics, goals, vocabulary, quotes.
- **actionable**: Goals/pains specific enough to drive real product decisions.
- **voice_fidelity**: Quotes and vocabulary sound like one consistent speaker.

## Output format
Respond with ONLY a JSON object (no markdown fences):
{
  "grounded": <float>,
  "distinctive": <float>,
  "coherent": <float>,
  "actionable": <float>,
  "voice_fidelity": <float>,
  "overall": <float>,
  "improvements": [
    "<specific actionable suggestion 1>",
    "<specific actionable suggestion 2>",
    "<specific actionable suggestion 3>"
  ]
}

Focus improvements on the LOWEST-scoring dimensions. Be specific — name \
the exact fields that need work and what to change.
"""


# ── Data types ───────────────────────────────────────────────────────

@dataclass
class CritiqueScore:
    overall: float
    dimensions: dict[str, float]
    improvements: list[str]
    cost_usd: float = 0.0


@dataclass
class RevisionRound:
    round_num: int
    pre_critique: CritiqueScore | None = None
    post_persona_name: str = ""
    post_groundedness: float = 0.0
    synthesis_cost_usd: float = 0.0
    critique_cost_usd: float = 0.0


@dataclass
class ReflexionResult:
    """Result of a critique-and-revise synthesis."""
    persona: PersonaV1
    groundedness: GroundednessReport
    total_cost_usd: float
    model_used: str
    initial_attempts: int
    revision_rounds: list[RevisionRound]
    initial_critique: CritiqueScore | None = None
    final_critique: CritiqueScore | None = None


# ── Critic ───────────────────────────────────────────────────────────

async def run_critic(
    client: AsyncAnthropic,
    model: str,
    persona_dict: dict,
) -> CritiqueScore:
    """Score a persona and suggest improvements."""
    stripped = {k: v for k, v in persona_dict.items() if not k.startswith("_")}
    prompt = json.dumps(stripped, indent=2)

    response = await client.messages.create(
        model=model,
        max_tokens=1024,
        system=CRITIC_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
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
    dims = {d: float(parsed.get(d, 0.0)) for d in
            ("grounded", "distinctive", "coherent", "actionable", "voice_fidelity")}

    return CritiqueScore(
        overall=float(parsed.get("overall", sum(dims.values()) / len(dims))),
        dimensions=dims,
        improvements=parsed.get("improvements", []),
        cost_usd=cost,
    )


# ── Reflexion synthesis ──────────────────────────────────────────────

REFLEXION_ADDENDUM = """

## Critic Feedback (Round {round})
A quality critic scored your previous attempt and found these issues:
{improvements}

Overall score: {overall:.2f}/1.00 (lowest: {worst_dim} at {worst_score:.2f})

Please address ALL improvement suggestions in your revised persona. \
Maintain distinctiveness — do not make the persona more generic in an \
attempt to be safe. Instead, make the specific improvements listed above \
while keeping the persona's unique voice and personality.
"""


async def reflexion_synthesize(
    cluster: ClusterData,
    client: AsyncAnthropic,
    model: str,
    revision_rounds: int = 1,
) -> ReflexionResult:
    """Synthesize with critique-and-revise loops.

    Args:
        revision_rounds: Number of critic-revision cycles (0 = control).
    """
    backend = AnthropicBackend(client=client, model=model)
    tool = build_tool_definition()

    # Initial synthesis
    synth_result = await synthesize(cluster, backend)
    persona = synth_result.persona
    persona_dict = persona.model_dump(mode="json")
    total_cost = synth_result.total_cost_usd
    rounds: list[RevisionRound] = []

    if revision_rounds == 0:
        # Control: no critique
        return ReflexionResult(
            persona=persona,
            groundedness=synth_result.groundedness,
            total_cost_usd=total_cost,
            model_used=model,
            initial_attempts=synth_result.attempts,
            revision_rounds=[],
        )

    # Initial critique (before any revisions)
    initial_critique = await run_critic(client, model, persona_dict)
    total_cost += initial_critique.cost_usd

    for round_num in range(1, revision_rounds + 1):
        # Critique current persona
        if round_num == 1:
            critique = initial_critique
        else:
            critique = await run_critic(client, model, persona_dict)
            total_cost += critique.cost_usd

        # Build reflexion prompt
        improvements_text = "\n".join(f"- {imp}" for imp in critique.improvements)
        worst_dim = min(critique.dimensions, key=critique.dimensions.get)
        worst_score = critique.dimensions[worst_dim]

        addendum = REFLEXION_ADDENDUM.format(
            round=round_num,
            improvements=improvements_text,
            overall=critique.overall,
            worst_dim=worst_dim,
            worst_score=worst_score,
        )

        user_msg = build_user_message(cluster) + addendum

        # Re-synthesize with critic feedback
        from pydantic import ValidationError

        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            tools=[tool],
            tool_choice={"type": "tool", "name": "create_persona"},
        )

        tool_block = next(b for b in response.content if b.type == "tool_use")

        in_tok = response.usage.input_tokens
        out_tok = response.usage.output_tokens
        if "opus" in model:
            synth_cost = (in_tok * 15 + out_tok * 75) / 1_000_000
        elif "haiku" in model:
            synth_cost = (in_tok * 1 + out_tok * 5) / 1_000_000
        else:
            synth_cost = (in_tok * 3 + out_tok * 15) / 1_000_000
        total_cost += synth_cost

        try:
            persona = PersonaV1.model_validate(tool_block.input)
            persona_dict = persona.model_dump(mode="json")
            groundedness = check_groundedness(persona, cluster)

            rounds.append(RevisionRound(
                round_num=round_num,
                pre_critique=critique,
                post_persona_name=persona.name,
                post_groundedness=groundedness.score,
                synthesis_cost_usd=synth_cost,
                critique_cost_usd=critique.cost_usd,
            ))
        except (ValidationError, Exception) as e:
            logger.warning("Revision round %d failed: %s", round_num, e)
            rounds.append(RevisionRound(
                round_num=round_num,
                pre_critique=critique,
                synthesis_cost_usd=synth_cost,
                critique_cost_usd=critique.cost_usd,
            ))
            break  # stop revising on failure

    # Final critique
    final_critique = await run_critic(client, model, persona_dict)
    total_cost += final_critique.cost_usd

    final_groundedness = check_groundedness(persona, cluster)

    return ReflexionResult(
        persona=persona,
        groundedness=final_groundedness,
        total_cost_usd=total_cost,
        model_used=model,
        initial_attempts=synth_result.attempts,
        revision_rounds=rounds,
        initial_critique=initial_critique,
        final_critique=final_critique,
    )
