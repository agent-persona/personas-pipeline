"""Local judge helper for experiment 3.05.

Per-claim entailment is the experiment-specific replacement for the missing
NLI model. The helper stays local to this branch so the main scaffold remains
untouched.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

ENTAILMENT_LABELS = ("entailed", "neutral", "contradicted")
OPENAI_FALLBACK_MODEL = "gpt-5-nano"

_JUDGE_SYSTEM_PROMPT = """\
You are an expert claim entailment judge for customer personas.

Your job is to decide whether a persona claim follows from the provided source
records. Use only the supplied records and do not assume unstated facts.

Label definitions:
- entailed: the claim is directly supported or is a very strong inference from
  the cited records.
- neutral: the claim is plausible but the evidence is insufficient to confirm
  it.
- contradicted: the cited records point the other way or clearly do not support
  the claim.

Return JSON only in this exact shape:
{
  "label": "entailed|neutral|contradicted",
  "confidence": 0.0,
  "rationale": "short explanation"
}
"""


@dataclass
class ClaimEntailmentScore:
    label: str
    confidence: float
    rationale: str
    judge_model: str


class JudgeBackend:
    def __init__(self, client: AsyncAnthropic, model: str) -> None:
        self.client = client
        self.model = model

    async def score(self, system: str, prompt: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenAIJudgeBackend:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self.client = client
        self.model = model

    async def score(self, system: str, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=1024,
        )
        return response.choices[0].message.content or ""


class FallbackJudgeBackend:
    def __init__(self, primary, fallback=None) -> None:
        self.primary = primary
        self.fallback = fallback

    async def score(self, system: str, prompt: str) -> str:
        try:
            return await self.primary.score(system=system, prompt=prompt)
        except Exception:
            if self.fallback is None:
                raise
            return await self.fallback.score(system=system, prompt=prompt)


def _parse_response(text: str, model: str) -> ClaimEntailmentScore:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            lowered = cleaned.lower()
            if "contradicted" in lowered:
                label = "contradicted"
            elif "entailed" in lowered:
                label = "entailed"
            else:
                label = "neutral"
            return ClaimEntailmentScore(
                label=label,
                confidence=0.0,
                rationale=cleaned[:240],
                judge_model=model,
            )

    label = str(data.get("label", "neutral")).strip().lower()
    if label not in ENTAILMENT_LABELS:
        lowered = json.dumps(data).lower()
        if "contradicted" in lowered:
            label = "contradicted"
        elif "entailed" in lowered:
            label = "entailed"
        else:
            label = "neutral"

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return ClaimEntailmentScore(
        label=label,
        confidence=max(0.0, min(1.0, confidence)),
        rationale=str(data.get("rationale", "")),
        judge_model=model,
    )


class LLMJudge:
    def __init__(self, backend: FallbackJudgeBackend | JudgeBackend | OpenAIJudgeBackend, model: str) -> None:
        self.backend = backend
        self.model = model

    async def score_claim(
        self,
        *,
        persona_name: str,
        field_path: str,
        claim: str,
        evidence_context: str,
    ) -> ClaimEntailmentScore:
        prompt = (
            f"Persona: {persona_name}\n"
            f"Field path: {field_path}\n"
            f"Claim: {claim}\n\n"
            "Cited source records:\n"
            f"{evidence_context}\n\n"
            "Decide whether the claim is entailed, neutral, or contradicted by the cited records."
        )
        response = await self.backend.score(_JUDGE_SYSTEM_PROMPT, prompt)
        return _parse_response(response, self.model)


def build_judge() -> tuple[LLMJudge, str, str]:
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if anthropic_key:
        primary = JudgeBackend(
            client=AsyncAnthropic(api_key=anthropic_key),
            model="claude-sonnet-4-20250514",
        )
        fallback = (
            OpenAIJudgeBackend(
                client=AsyncOpenAI(api_key=openai_key),
                model=OPENAI_FALLBACK_MODEL,
            )
            if openai_key
            else None
        )
        return (
            LLMJudge(
                backend=FallbackJudgeBackend(primary=primary, fallback=fallback),
                model="claude-sonnet-4-20250514",
            ),
            "anthropic" if fallback is None else "anthropic->openai",
            "claude-sonnet-4-20250514",
        )

    if openai_key:
        return (
            LLMJudge(
                backend=OpenAIJudgeBackend(
                    client=AsyncOpenAI(api_key=openai_key),
                    model=OPENAI_FALLBACK_MODEL,
                ),
                model=OPENAI_FALLBACK_MODEL,
            ),
            "openai",
            OPENAI_FALLBACK_MODEL,
        )

    raise RuntimeError("Missing ANTHROPIC_API_KEY and OPENAI_API_KEY")
