"""J2 Voice Consistency — LLM-as-judge rubric scorer.

Trustworthiness: HIGH (LLM judge with explicit rubric criteria).
Method: Claude evaluates whether the persona's voice is distinctive and consistent
across multiple responses, or generic and shifting.

Evidence: Embedding similarity measures semantic closeness but not stylistic
distinctiveness. A persona can have high embedding similarity across responses
but still sound like a generic assistant.
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
You are an expert evaluator of AI persona quality. Assess whether the persona's voice is \
distinctive and consistent across multiple responses, or generic and shifting.

Rate the persona on VOICE CONSISTENCY using the following 1–5 rubric:

1 = No Distinct Voice
   • Each response could have been written by a different assistant
   • No consistent vocabulary, sentence rhythm, or rhetorical habits
   • Tone shifts wildly between responses

2 = Weak Voice
   • Slight tendency toward a style but easily lost when topic changes
   • Formality level inconsistent with stated background
   • Generic connectors dominate ("Additionally...", "In conclusion...")

3 = Recognizable Voice
   • Moderately distinctive style in most responses
   • Formality roughly appropriate; occasional slippage into generic prose

4 = Consistent Voice
   • Same rhythm, vocabulary, and tone across all responses
   • Style matches stated communication profile
   • Persona "sounds like themselves" even on unfamiliar topics

5 = Signature Voice
   • Instantly recognizable — identifiable from a response alone
   • Idiosyncratic phrasing and rhythm that are entirely their own
   • Style is consistent AND organically tied to background

Respond ONLY with JSON: {"score": N, "reasoning": "one or two sentences"}"""

_RUBRIC_LABEL = "1=no_distinct_voice 2=weak_voice 3=recognizable 4=consistent 5=signature"


class VoiceConsistencyScorer(BaseScorer):
    """Evaluates whether the persona maintains a distinctive, consistent voice across responses."""

    dimension_id = "J2"
    dimension_name = "Voice Consistency"
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
                f"SAMPLE RESPONSES:\n{responses_block}\n\n"
                "Now rate voice consistency on the 1–5 rubric."
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
