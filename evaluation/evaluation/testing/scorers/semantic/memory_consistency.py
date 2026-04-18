"""D8 Memory Consistency — direct recall probes.

Trustworthiness: HIGH for direct recall, MEDIUM for indirect.
Method: Generate recall probes from persona attributes, compare answers to source.
"""

from __future__ import annotations

from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import Persona, EvalResult
from evaluation.testing.source_context import SourceContext


def _generate_recall_probes(persona: Persona) -> list[dict]:
    """Generate direct recall probes from persona attributes."""
    probes = []
    if persona.age is not None:
        probes.append({"question": "How old are you?", "expected": str(persona.age), "field": "age"})
    if persona.location:
        probes.append({"question": "Where do you live?", "expected": persona.location, "field": "location"})
    if persona.occupation:
        probes.append({"question": "What is your job title?", "expected": persona.occupation, "field": "occupation"})
    if persona.experience_years is not None:
        probes.append({"question": "How many years of experience do you have?", "expected": str(persona.experience_years), "field": "experience_years"})
    if persona.industry:
        probes.append({"question": "What industry do you work in?", "expected": persona.industry, "field": "industry"})
    if persona.education:
        probes.append({"question": "What is your education level?", "expected": persona.education, "field": "education"})
    return probes


class MemoryConsistencyScorer(BaseScorer):
    """Evaluates whether a persona's answers match its declared attributes."""

    dimension_id = "D8"
    dimension_name = "Memory Consistency"
    tier = 2

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        """Evaluate memory consistency using recall probes.

        Reads pre_computed answers from source_context.extra_data["answers"]:
        a dict mapping field name to the persona's response string.
        """
        probes = _generate_recall_probes(persona)
        if not probes:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No probes generated"},
            )

        pre_computed: dict[str, str] = source_context.extra_data.get("answers", {})
        probe_results = []

        for probe in probes:
            answer = pre_computed.get(probe["field"])
            if answer is not None:
                # Substring match — catches "Product Manager" in "I'm a Product Manager at..."
                # but won't match partial variants like "product management lead"
                expected_lower = probe["expected"].lower()
                answer_lower = answer.lower()
                correct = expected_lower in answer_lower
                probe_results.append({
                    "field": probe["field"],
                    "question": probe["question"],
                    "expected": probe["expected"],
                    "answer": answer,
                    "correct": correct,
                })

        if not probe_results:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No answers to check"},
            )

        correct_count = sum(1 for p in probe_results if p["correct"])
        score = correct_count / len(probe_results)
        passed = score >= 0.8

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details={
                "probes": probe_results,
                "correct": correct_count,
                "total": len(probe_results),
            },
        )
