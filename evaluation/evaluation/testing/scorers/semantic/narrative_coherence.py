"""D12 Narrative Coherence — LLM-as-judge with structured rubric.

Trustworthiness: LOW-MEDIUM (LLMs are poor judges of narrative quality).
Method: LLM-as-judge with narrative coherence rubric + pre-computed score mode.
"""

from __future__ import annotations

import json

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import Persona, EvalResult
from evaluation.testing.source_context import SourceContext

RUBRIC = """Rate this persona on narrative coherence (1-5 scale):

1 = Incoherent: Fields seem randomly assembled, no sense of a real person
2 = Weak: Some connections but major gaps or implausibilities
3 = Acceptable: Mostly coherent but feels somewhat artificial
4 = Strong: Reads like a real person with a plausible life story
5 = Excellent: Compelling, internally consistent narrative

Evaluate:
- Does the career trajectory match the skills and education?
- Does the communication style match the background?
- Do the goals and pain points fit the professional context?
- Does the emotional profile fit the personality traits?
- Does it feel like ONE person's life?

Respond ONLY with JSON: {"score": N, "reasoning": "..."}
"""


class NarrativeCoherenceScorer(BaseScorer):
    """Evaluates narrative coherence of a persona via LLM-as-judge or pre-computed score."""

    dimension_id = "D12"
    dimension_name = "Narrative Coherence"
    tier = 2

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        """Evaluate narrative coherence.

        Reads from source_context.extra_data:
        - "narrative_score": pre-computed 1-5 score (for testing)
        - "narrative_reasoning": optional reasoning string
        If no pre-computed score, reads "llm_client" for LLM-as-judge mode.
        """
        pre_computed_score = source_context.extra_data.get("narrative_score")

        if pre_computed_score is not None:
            score_normalized = max(0.0, min(1.0, pre_computed_score / 5.0))
            return self._result(
                persona, passed=score_normalized >= 0.6,
                score=round(score_normalized, 4),
                details={
                    "raw_score": pre_computed_score,
                    "reasoning": source_context.extra_data.get("narrative_reasoning", ""),
                    "source": "pre_computed",
                },
            )

        # LLM-as-judge mode: use llm_client from extra_data
        llm_client = source_context.extra_data.get("llm_client")
        if llm_client is None:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No LLM client or pre-computed score"},
            )

        persona_text = persona.model_dump_json(indent=2)
        messages = llm_client.format_messages(
            system=RUBRIC,
            user=f"Persona to evaluate:\n\n{persona_text}",
        )

        try:
            response = llm_client.complete(messages, max_tokens=500)
            parsed = json.loads(response)
            raw_score = parsed.get("score", 3)
            reasoning = parsed.get("reasoning", "")
            score_normalized = max(0.0, min(1.0, raw_score / 5.0))

            return self._result(
                persona, passed=score_normalized >= 0.6,
                score=round(score_normalized, 4),
                details={
                    "raw_score": raw_score,
                    "reasoning": reasoning,
                    "source": "llm_judge",
                },
            )
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            return self._result(
                persona, passed=False, score=0.5,
                details={"error": str(e), "source": "llm_judge"},
                errors=[str(e)],
            )
