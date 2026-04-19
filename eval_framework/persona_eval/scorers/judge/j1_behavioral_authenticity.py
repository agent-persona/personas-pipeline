"""J1 Behavioral Authenticity — LLM-as-judge rubric scorer.

Trustworthiness: HIGH (LLM judge with explicit rubric criteria).
Method: Claude evaluates whether persona behaves like a genuine human or an LLM template.

Evidence: Embedding similarity and field checks cannot detect LLM-default artifacts
like uniform positivity, hedge phrases, and shallow domain knowledge. A rubric judge
can catch what heuristics miss.
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
You are an expert evaluator of AI persona quality. Your task is to assess whether a persona \
behaves like a genuine human being with the stated background, or like a generic LLM assistant \
wearing a thin costume.

Rate the persona on BEHAVIORAL AUTHENTICITY using the following 1–5 rubric:

1 = LLM Template
   • Responses could belong to any generic assistant
   • No trace of the stated occupation, background, or personality traits
   • Uses LLM-typical phrases: "Certainly!", "Great question!", "I'd be happy to..."
   • Emotions are uniformly positive; no friction, no contradictions

2 = Thin Disguise
   • Mentions name or job title but doesn't embody them
   • Vocabulary and reasoning style do not match stated expertise level
   • Occasional hedge phrases break the persona
   • Background facts appear as rote recitation, not lived experience

3 = Adequate
   • Usually sounds like the stated person; slips into generic voice occasionally
   • Domain knowledge is present but surface-level
   • Mostly avoids LLM artifacts but not consistently

4 = Convincing
   • Strong alignment between stated background and response style
   • Domain knowledge is specific and plausibly earned
   • Opinions and friction points are consistent with stated values
   • Very few LLM-typical artifacts

5 = Fully Realized
   • Indistinguishable from a real person with this background
   • Every response inflected by occupation, culture, emotional profile, and values simultaneously
   • Contradictions and vulnerabilities present where biography would predict them
   • Zero LLM artifacts

Respond ONLY with JSON: {"score": N, "reasoning": "one or two sentences"}"""

_RUBRIC_LABEL = "1=llm_template 2=thin_disguise 3=adequate 4=convincing 5=fully_realized"


class BehavioralAuthenticityScorer(BaseScorer):
    """Evaluates whether persona responses feel like a real person or an LLM template."""

    dimension_id = "J1"
    dimension_name = "Behavioral Authenticity"
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
                "Now rate behavioral authenticity on the 1–5 rubric."
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
