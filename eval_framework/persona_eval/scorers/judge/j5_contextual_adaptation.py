"""J5 Contextual Adaptation — LLM-as-judge rubric scorer.

Trustworthiness: HIGH (LLM judge with explicit rubric criteria).
Method: Claude evaluates whether the persona adapts its behavior to different
situational contexts while staying unmistakably in character.

Evidence: Embedding similarity can detect tonal shifts but cannot evaluate
whether those shifts are contextually appropriate AND persona-consistent.
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
You are an expert evaluator of AI persona quality. Assess whether the persona adapts its \
behavior to different situational contexts while staying unmistakably in character.

Rate the persona on CONTEXTUAL ADAPTATION using the following 1–5 rubric:

1 = No Adaptation
   • Identical tone and vocabulary regardless of context
   • No contextual awareness — treats every prompt as the same interaction

2 = Minimal Adaptation
   • Slight shifts in length or formality, but not tied to the persona's profile
   • Adaptation feels mechanical rather than organic

3 = Adequate Adaptation
   • Noticeable adaptation to different contexts
   • Shifts feel appropriate but generic — any persona might do the same

4 = Good Adaptation
   • Adapts in a way consistent with the persona's specific background
   • Same person clearly present in both formal and casual contexts
   • Same values, different register

5 = Masterful Adaptation
   • Each context reveals a different facet of the same deeply consistent character
   • Persona uses contextual shifts to reveal more of themselves
   • Adaptation reinforces authenticity rather than threatening it

Respond ONLY with JSON: {"score": N, "reasoning": "one or two sentences"}"""

_RUBRIC_LABEL = "1=no_adaptation 2=minimal 3=adequate 4=good 5=masterful"


class ContextualAdaptationScorer(BaseScorer):
    """Evaluates whether persona adapts to contexts while maintaining consistent character."""

    dimension_id = "J5"
    dimension_name = "Contextual Adaptation"
    tier = 3
    requires_set = False

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        contextual = source_context.extra_data.get("contextual_responses") or []
        response_count: int

        if contextual and isinstance(contextual, list):
            items = [item for item in contextual[:5] if isinstance(item, dict)]
            responses_block = "\n\n".join(
                f"Context: {item.get('context', 'unspecified')}\nResponse:\n{item.get('response', '')}"
                for item in items
            )
            response_count = len(items)
        else:
            flat = [r for r in (source_context.extra_data.get("responses") or [])
                    if isinstance(r, str) and r.strip()]
            if not flat:
                return self._result(
                    persona, passed=True, score=1.0,
                    details={
                        "skipped": True,
                        "reason": "No contextual_responses or responses in extra_data",
                    },
                )
            responses_block = "\n\n".join(
                f"Response {i + 1}:\n{r}" for i, r in enumerate(flat[:5])
            )
            response_count = len(flat[:5])

        persona_block = build_persona_block(persona)
        client = make_client()
        messages = client.format_messages(
            system=_SYSTEM_PROMPT,
            user=(
                f"PERSONA DEFINITION:\n{persona_block}\n\n"
                f"RESPONSES ACROSS CONTEXTS:\n{responses_block}\n\n"
                "Rate contextual adaptation on the 1–5 rubric."
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
                "response_count": response_count,
                "rubric": _RUBRIC_LABEL,
            },
        )
