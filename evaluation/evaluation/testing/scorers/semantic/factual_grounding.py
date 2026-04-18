"""D4 Factual Grounding — claim extraction + retrieval against source context.

Trustworthiness: MEDIUM (threshold-dependent).
Method: Extract claims from persona fields, retrieve against source text chunks,
measure semantic similarity.
"""

from __future__ import annotations

import re

from evaluation.testing.embeddings import Embedder
from evaluation.testing.scorer import BaseScorer
from evaluation.testing.schemas import Persona, EvalResult
from evaluation.testing.source_context import SourceContext

SIMILARITY_THRESHOLD = 0.2  # Minimum similarity to consider a claim grounded (tuned for MiniLM-L6-v2)
CHUNK_SIZE = 200  # Characters per source chunk


def _extract_claims(persona: Persona) -> list[str]:
    """Extract testable claims from persona fields."""
    claims = []
    if persona.occupation:
        claims.append(f"The person works as a {persona.occupation}")
    if persona.industry:
        claims.append(f"They work in {persona.industry}")
    if persona.experience_years is not None:
        claims.append(f"They have {persona.experience_years} years of experience")
    if persona.age is not None:
        claims.append(f"They are {persona.age} years old")
    if persona.location:
        claims.append(f"They are located in {persona.location}")
    for goal in persona.goals:
        if goal and goal.strip():
            claims.append(f"Their goal is to {goal}")
    for pp in persona.pain_points:
        if pp and pp.strip():
            claims.append(f"They experience: {pp}")
    return claims


def _chunk_source(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split source text into sentence-boundary chunks for retrieval."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


class FactualGroundingScorer(BaseScorer):
    """Evaluates whether persona claims are grounded in source text."""

    dimension_id = "D4"
    dimension_name = "Factual Grounding"
    tier = 2

    def __init__(self) -> None:
        self._embedder: Embedder | None = None  # Lazy init for test performance

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def score(self, persona: Persona, source_context: SourceContext) -> EvalResult:
        if not source_context.text.strip():
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No source text provided"},
            )

        claims = _extract_claims(persona)
        if not claims:
            return self._result(
                persona, passed=True, score=1.0,
                details={"skipped": True, "reason": "No claims extracted"},
            )

        embedder = self._get_embedder()
        chunks = source_context.chunks if source_context.chunks else _chunk_source(source_context.text)

        claim_scores = []
        for claim in claims:
            results = embedder.retrieval_score(claim, chunks, top_k=1)
            best_score = results[0][1] if results else 0.0
            claim_scores.append({
                "claim": claim,
                "best_match_score": round(best_score, 4),
                "grounded": best_score >= SIMILARITY_THRESHOLD,
            })

        grounded_count = sum(1 for cs in claim_scores if cs["grounded"])
        score = grounded_count / len(claim_scores) if claim_scores else 0.0
        passed = score >= 0.5

        return self._result(
            persona, passed=passed, score=round(score, 4),
            details={
                "claim_scores": claim_scores,
                "grounded_claims": grounded_count,
                "total_claims": len(claim_scores),
            },
        )
