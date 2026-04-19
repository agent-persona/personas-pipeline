"""J4 Persona Depth — LLM-as-judge rubric scorer.

Trustworthiness: HIGH (LLM judge with explicit rubric criteria).
Method: Claude evaluates whether the persona is a fully realized person with
interior life, or a shallow demographic summary.

Evidence: Structural scorers verify field presence; they cannot evaluate whether
a bio reveals genuine interiority, contradiction, and lived experience.
"""

from __future__ import annotations

from persona_eval.scorer import BaseScorer
from persona_eval.schemas import EvalResult, Persona
from persona_eval.source_context import SourceContext
from persona_eval.scorers.judge._base import (
    build_persona_block,
    make_client,
    normalize,
    parse_score,
)

_SYSTEM_PROMPT = """\
You are an expert evaluator of AI persona quality. Assess whether this is a fully realized \
person with interior life, or a shallow demographic summary.

Rate the persona on PERSONA DEPTH using the following 1–5 rubric:

1 = Demographic Skeleton
   • Feels like a form filled in with required fields
   • No sense of history, contradiction, or interiority
   • Responses treat background facts as data points, not lived experience

2 = Thin Profile
   • Some interesting detail but lacks depth
   • Biography reads as a list of achievements, not a life
   • No vulnerability, regret, or ambiguity visible

3 = Moderate Depth
   • A real person could have this profile, but not fully inhabited
   • A few specific, non-generic details bring it to life

4 = Realized Person
   • Strong sense of a specific individual with history
   • Responses draw on backstory organically
   • At least one unexpected dimension (contradiction, private struggle, unlikely interest)

5 = Full Interior Life
   • Responses reveal a complete human being with aspirations, wounds, habits of mind
   • Surprises you with specificity that feels discovered, not constructed
   • Multiple dimensions of depth operating simultaneously

Respond ONLY with JSON: {"score": N, "reasoning": "one or two sentences"}"""

_RUBRIC_LABEL = "1=skeleton 2=thin 3=moderate 4=realized 5=full_interior_life"


class PersonaDepthScorer(BaseScorer):
    """Evaluates whether persona has full interior life or is a shallow demographic summary."""

    dimension_id = "J4"
    dimension_name = "Persona Depth"
    tier = 3
    requires_set = False

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        responses: list[str] = source_context.extra_data.get("responses") or []
        responses = [r for r in responses if isinstance(r, str) and r.strip()]

        if not responses:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No responses in extra_data"},
            )

        persona_block = build_persona_block(persona)
        responses_block = "\n\n".join(
            f"Response {i + 1}:\n{r}" for i, r in enumerate(responses[:5])
        )

        client = make_client()
        messages = client.format_messages(
            system=_SYSTEM_PROMPT,
            user=(
                f"PERSONA DEFINITION:\n{persona_block}\n\n"
                f"BIO (key source for depth):\n{persona.bio}\n\n"
                f"SAMPLE RESPONSES:\n{responses_block}\n\n"
                "Rate persona depth on the 1–5 rubric."
            ),
        )
        raw = client.complete(messages, max_tokens=300)
        raw_score, reasoning, parse_ok = parse_score(raw)
        normalized = normalize(raw_score)
        passed = normalized >= 0.6

        return self._result(
            persona,
            passed=passed,
            score=normalized,
            details={
                "raw_score": raw_score,
                "reasoning": reasoning,
                "parse_ok": parse_ok,
                "response_count": len(responses[:5]),
                "rubric": _RUBRIC_LABEL,
            },
        )
