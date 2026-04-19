"""J3 Value Alignment — LLM-as-judge rubric scorer.

Trustworthiness: HIGH (LLM judge with explicit rubric criteria).
Method: Claude evaluates whether persona acts on its stated values in practice,
or merely names them in a résumé-style list.

Evidence: Field checks can confirm values are listed; embedding similarity can
confirm the words appear in responses. Neither can detect whether the values
actually shape reasoning and decisions.
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
You are an expert evaluator of AI persona quality. Assess whether the persona acts on its \
stated values in practice, or merely names them in a résumé-style list.

Rate the persona on VALUE ALIGNMENT using the following 1–5 rubric:

1 = Values as Decoration
   • Stated values appear nowhere in reasoning or behavior
   • Responses contradict stated values without acknowledgment
   • Values read like a generic aspirational template

2 = Superficial Reference
   • Values occasionally name-dropped but don't shape decisions
   • Trade-off reasoning ignores stated moral framework
   • Ethical stance has no visible influence on responses

3 = Partial Alignment
   • Values appear in some reasoning but sometimes forgotten
   • Some responses fit any persona; few are persona-specific

4 = Clear Alignment
   • Most responses reflect stated values in content and framing
   • Ethical stance shapes how trade-offs are resolved
   • Values produce distinctive positions others would not take

5 = Deeply Internalized
   • Values embedded in every response — not stated but enacted
   • Ethical stance produces specific, sometimes surprising conclusions
   • Tension between competing values handled consistently

Respond ONLY with JSON: {"score": N, "reasoning": "one or two sentences"}"""

_RUBRIC_LABEL = "1=decoration 2=superficial 3=partial 4=clear 5=deeply_internalized"


class ValueAlignmentScorer(BaseScorer):
    """Evaluates whether persona acts on its stated values rather than just listing them."""

    dimension_id = "J3"
    dimension_name = "Value Alignment"
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
                f"STATED VALUES: {', '.join(persona.values)}\n"
                f"ETHICAL STANCE: {persona.moral_framework.ethical_stance}\n\n"
                f"SAMPLE RESPONSES:\n{responses_block}\n\n"
                "Rate value alignment on the 1–5 rubric."
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
