"""LLM-as-judge with optional few-shot calibration.

Problem space 5 of `PRD_LAB_RESEARCH.md` asks:

    "We can't trust any of the above experiments unless we trust the eval.
     LLM-as-judge has known failure modes (self-preference, position bias,
     sycophancy). We need to red-team the judge before using it as ground
     truth."

Experiment 5.13 adds few-shot calibration anchors (examples of a 1, 3,
and 5) to tighten the judge's score distribution and reduce variance.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

from anthropic import AsyncAnthropic

CalibrationMode = Literal["none", "few_shot"]


@dataclass
class JudgeScore:
    """One judge's score on one persona or twin transcript.

    `dimensions` maps rubric dimensions (e.g. "grounded", "distinctive",
    "voice_consistency") to a 1-5 score. `overall` is the weighted
    average. `rationale` is the judge's free-text justification, kept for
    debugging and for the human-correlation study.
    """

    overall: float
    dimensions: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    judge_model: str = ""


class JudgeBackend:
    """Anthropic-backed judge that calls the LLM to score personas."""

    def __init__(self, client: AsyncAnthropic, model: str) -> None:
        self.client = client
        self.model = model

    async def score(self, system: str, prompt: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


# ============================================================================
# Rubric and calibration anchors
# ============================================================================

DEFAULT_DIMENSIONS = (
    "grounded",
    "distinctive",
    "coherent",
    "actionable",
    "voice_fidelity",
)

_JUDGE_SYSTEM_PROMPT = """\
You are an expert persona evaluator. You score synthesized customer personas \
on a 1-5 scale across multiple quality dimensions.

Scoring scale:
  1 = Very poor — fails this dimension entirely
  2 = Weak — major gaps, mostly generic or inconsistent
  3 = Acceptable — meets minimum bar but unremarkable
  4 = Good — solid quality with minor issues
  5 = Excellent — publication-ready, specific, grounded, distinctive

Dimensions to score:
- **grounded**: Are claims traceable to source evidence? Do record IDs exist and make sense?
- **distinctive**: Does this feel like a real individual, or a generic average?
- **coherent**: Are demographics, firmographics, vocabulary, and quotes internally consistent?
- **actionable**: Are goals/pains specific enough to drive product decisions?
- **voice_fidelity**: Do sample quotes sound like one consistent speaker?

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{
  "grounded": <1-5>,
  "distinctive": <1-5>,
  "coherent": <1-5>,
  "actionable": <1-5>,
  "voice_fidelity": <1-5>,
  "overall": <1-5>,
  "rationale": "<brief justification>"
}
"""

# Anchor examples for few-shot calibration
_ANCHOR_SCORE_1 = """\
=== CALIBRATION EXAMPLE — Score: 1 (Very Poor) ===
{
  "name": "Tech User",
  "summary": "A person who uses technology.",
  "demographics": {"age_range": "18-65", "gender_distribution": "mixed", "location_signals": []},
  "firmographics": {"company_size": null, "industry": null, "role_titles": [], "tech_stack_signals": []},
  "goals": ["Be more productive", "Save time"],
  "pains": ["Things are hard", "Not enough features"],
  "motivations": ["Wants to do better"],
  "objections": ["Price"],
  "channels": ["Internet"],
  "vocabulary": ["tool", "software", "platform"],
  "decision_triggers": ["Recommendation"],
  "sample_quotes": ["I like good tools.", "This should be easier."],
  "journey_stages": [
    {"stage": "awareness", "mindset": "looking", "key_actions": ["search"], "content_preferences": ["articles"]}
  ],
  "source_evidence": [
    {"claim": "Uses technology", "record_ids": ["rec_001"], "field_path": "goals.0", "confidence": 0.5}
  ]
}
SCORES: grounded=1, distinctive=1, coherent=1, actionable=1, voice_fidelity=1, overall=1
WHY: Completely generic. No real grounding — single vague evidence entry. Goals/pains \
could describe anyone. No distinctive voice, vocabulary, or specificity. Unusable for \
any product decision."""

_ANCHOR_SCORE_3 = """\
=== CALIBRATION EXAMPLE — Score: 3 (Acceptable) ===
{
  "name": "Marketing Manager Maria",
  "summary": "A mid-level marketing manager at a SaaS company focused on lead generation.",
  "demographics": {"age_range": "28-35", "gender_distribution": "predominantly female", "location_signals": ["US"]},
  "firmographics": {"company_size": "50-200", "industry": "SaaS", "role_titles": ["Marketing Manager"], "tech_stack_signals": ["HubSpot"]},
  "goals": ["Increase lead conversion rate", "Build better attribution models", "Reduce cost per lead"],
  "pains": ["Difficulty tracking campaign performance across channels", "Manual reporting takes too long"],
  "motivations": ["Prove marketing ROI to leadership", "Get promoted to director"],
  "objections": ["Already invested in current stack"],
  "channels": ["LinkedIn", "Marketing blogs", "Slack communities"],
  "vocabulary": ["MQL", "pipeline", "attribution", "CAC"],
  "decision_triggers": ["Integration with HubSpot", "Case study from similar company"],
  "sample_quotes": ["I need to show the CEO that marketing drives revenue, not just leads.", "Our reporting is still mostly manual spreadsheets."],
  "journey_stages": [
    {"stage": "consideration", "mindset": "comparing options", "key_actions": ["requesting demos", "reading reviews"], "content_preferences": ["comparison guides", "ROI calculators"]},
    {"stage": "decision", "mindset": "building internal business case", "key_actions": ["pilot program", "stakeholder presentation"], "content_preferences": ["case studies", "implementation guides"]}
  ],
  "source_evidence": [
    {"claim": "Focused on lead conversion", "record_ids": ["rec_012", "rec_034"], "field_path": "goals.0", "confidence": 0.7},
    {"claim": "Manual reporting pain", "record_ids": ["rec_034"], "field_path": "pains.1", "confidence": 0.8},
    {"claim": "Wants to prove ROI", "record_ids": ["rec_012"], "field_path": "motivations.0", "confidence": 0.6}
  ]
}
SCORES: grounded=3, distinctive=3, coherent=3, actionable=3, voice_fidelity=3, overall=3
WHY: Functional but not sharp. Goals are reasonable but generic to any marketing manager. \
Some grounding but evidence is thin — only 3 entries covering a fraction of claims. \
Vocabulary is appropriate but standard. Quotes are decent but could be more distinctive. \
Usable as a starting point but needs refinement."""

_ANCHOR_SCORE_5 = """\
=== CALIBRATION EXAMPLE — Score: 5 (Excellent) ===
{
  "name": "DevOps Dana — The Automation-First Platform Lead",
  "summary": "A senior platform engineer at a 150-person fintech company who treats every manual process as a bug. Spends 60% of her week on CI/CD infrastructure and the rest fighting for engineering standards across 4 product teams.",
  "demographics": {"age_range": "32-38", "gender_distribution": "predominantly female", "location_signals": ["SF Bay Area", "remote-first company"]},
  "firmographics": {"company_size": "100-200", "industry": "fintech", "role_titles": ["Senior Platform Engineer", "DevOps Lead"], "tech_stack_signals": ["Terraform", "GitHub Actions", "Datadog", "AWS EKS"]},
  "goals": ["Automate deployment pipelines to achieve <10min deploy cycles across all 4 product teams", "Consolidate observability into a single Datadog dashboard that replaces 3 separate monitoring tools", "Reduce CI flakiness from 12% to under 2% by Q3"],
  "pains": ["Spends 5+ hours/week debugging CI failures caused by flaky integration tests that other teams won't fix", "Every team has its own deployment script with different conventions — no standardization", "Rate limits on the bulk export API break the nightly data pipeline at least twice a month"],
  "motivations": ["Wants to prove that platform engineering deserves headcount, not just her doing everything solo", "Believes infrastructure-as-code is non-negotiable — if it can't be Terraformed, it doesn't ship"],
  "objections": ["Won't adopt any tool that doesn't have a Terraform provider or well-documented API", "Resistant to per-seat pricing because her 'users' are CI bots, not humans"],
  "channels": ["HashiCorp community forums", "DevOps Weekly newsletter", "#platform-eng internal Slack", "KubeCon talks on YouTube"],
  "vocabulary": ["toil", "blast radius", "golden path", "paved road", "SLO", "error budget", "drift detection"],
  "decision_triggers": ["Terraform provider available on day one", "SOC 2 compliance documentation", "Bulk API with >10k row export support"],
  "sample_quotes": ["If I can't terraform it, it doesn't exist in my infrastructure.", "I spent three hours yesterday debugging a flaky test that a product team wrote in 2023 and never maintained. That's not engineering, that's archaeology.", "Show me the API rate limits before the demo. I've been burned too many times."],
  "journey_stages": [
    {"stage": "evaluation", "mindset": "Prove it works with my stack before I invest time", "key_actions": ["Read API docs", "Check Terraform registry", "Run a POC in staging"], "content_preferences": ["API reference", "Terraform module docs", "Architecture decision records"]},
    {"stage": "adoption", "mindset": "Standardize it across all teams before entropy sets in", "key_actions": ["Write internal RFC", "Build golden-path template", "Present ROI to VP Eng"], "content_preferences": ["Migration guides", "Cost comparison calculators", "Case studies from similar-size fintechs"]}
  ],
  "source_evidence": [
    {"claim": "Automate deployment to <10min cycles", "record_ids": ["ga4_003", "ga4_007", "intercom_001"], "field_path": "goals.0", "confidence": 0.9},
    {"claim": "Consolidate observability tools", "record_ids": ["ga4_004", "hubspot_002"], "field_path": "goals.1", "confidence": 0.85},
    {"claim": "Reduce CI flakiness", "record_ids": ["intercom_003", "ga4_009"], "field_path": "goals.2", "confidence": 0.8},
    {"claim": "Debugging flaky CI tests weekly", "record_ids": ["intercom_001", "ga4_007"], "field_path": "pains.0", "confidence": 0.95},
    {"claim": "No deployment standardization", "record_ids": ["ga4_003", "ga4_006"], "field_path": "pains.1", "confidence": 0.85},
    {"claim": "Bulk API rate limits break pipeline", "record_ids": ["intercom_003"], "field_path": "pains.2", "confidence": 1.0},
    {"claim": "Platform eng needs headcount", "record_ids": ["intercom_001", "hubspot_002"], "field_path": "motivations.0", "confidence": 0.7},
    {"claim": "IaC is non-negotiable", "record_ids": ["ga4_003", "ga4_008"], "field_path": "motivations.1", "confidence": 0.9},
    {"claim": "Requires Terraform provider", "record_ids": ["ga4_008", "ga4_003"], "field_path": "objections.0", "confidence": 0.95},
    {"claim": "Per-seat pricing doesn't fit CI bots", "record_ids": ["intercom_003"], "field_path": "objections.1", "confidence": 0.85}
  ]
}
SCORES: grounded=5, distinctive=5, coherent=5, actionable=5, voice_fidelity=5, overall=5
WHY: Every claim has strong evidence with multiple record IDs. Persona is unmistakably \
specific — you can picture this person. Goals have quantified targets. Pains reference \
concrete time costs. Vocabulary is domain-authentic. Quotes are vivid and individually \
voiced. Highly actionable for product decisions (API rate limits, Terraform support, \
pricing model). Internal consistency is flawless."""


def _build_judge_prompt(persona: dict) -> str:
    """Format a persona for the judge to score."""
    return (
        "Score the following persona on each dimension (1-5).\n\n"
        "PERSONA:\n"
        + json.dumps(persona, indent=2, default=str)
    )


def _build_calibrated_judge_prompt(persona: dict) -> str:
    """Format a persona with few-shot calibration anchors prepended."""
    return (
        "Below are three calibration examples showing what a 1, 3, and 5 look like. "
        "Use these as anchors when scoring the TARGET persona that follows.\n\n"
        + _ANCHOR_SCORE_1 + "\n\n"
        + _ANCHOR_SCORE_3 + "\n\n"
        + _ANCHOR_SCORE_5 + "\n\n"
        + "=== TARGET PERSONA TO SCORE ===\n"
        "Score the following persona on each dimension (1-5).\n\n"
        "PERSONA:\n"
        + json.dumps(persona, indent=2, default=str)
    )


def _parse_judge_response(text: str, model: str) -> JudgeScore:
    """Parse the judge's JSON response into a JudgeScore."""
    # Strip markdown fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            return JudgeScore(
                overall=float("nan"),
                dimensions={d: float("nan") for d in DEFAULT_DIMENSIONS},
                rationale=f"Failed to parse judge response: {text[:200]}",
                judge_model=model,
            )

    dimensions = {}
    for dim in DEFAULT_DIMENSIONS:
        val = data.get(dim)
        dimensions[dim] = float(val) if val is not None else float("nan")

    overall = data.get("overall", float("nan"))
    rationale = data.get("rationale", "")

    return JudgeScore(
        overall=float(overall),
        dimensions=dimensions,
        rationale=rationale,
        judge_model=model,
    )


class LLMJudge:
    """LLM-as-judge with optional few-shot calibration.

    Default rubric dimensions:
      - grounded       : claims traceable to source data
      - distinctive    : persona feels like an individual, not a generic average
      - coherent       : internal consistency across fields
      - actionable     : goals / pains sharp enough to drive product decisions
      - voice_fidelity : sample quotes sound like one consistent speaker

    Calibration modes:
      - "none"     : zero-shot — just the rubric and the persona (baseline)
      - "few_shot" : prepend 3 anchor examples (score 1, 3, 5) before the target
    """

    DEFAULT_DIMENSIONS = DEFAULT_DIMENSIONS

    def __init__(
        self,
        backend: JudgeBackend | None = None,
        model: str = "claude-opus-4-6",
        dimensions: tuple[str, ...] = DEFAULT_DIMENSIONS,
        calibration: CalibrationMode = "none",
    ) -> None:
        self.backend = backend
        self.model = model
        self.dimensions = dimensions
        self.calibration = calibration

    async def score_persona(self, persona: dict) -> JudgeScore:
        """Score a single persona JSON against the rubric."""
        if self.backend is None:
            return JudgeScore(
                overall=float("nan"),
                dimensions={d: float("nan") for d in self.dimensions},
                rationale="No backend configured",
                judge_model=self.model,
            )

        if self.calibration == "few_shot":
            prompt = _build_calibrated_judge_prompt(persona)
        else:
            prompt = _build_judge_prompt(persona)

        response = await self.backend.score(
            system=_JUDGE_SYSTEM_PROMPT,
            prompt=prompt,
        )

        return _parse_judge_response(response, self.model)

    async def score_transcript(
        self,
        persona: dict,
        transcript: list[dict],
    ) -> JudgeScore:
        """Score a twin chat transcript against the persona it was seeded with."""
        # TODO(space-5): implement for the twin-drift experiments in space 4.
        return JudgeScore(
            overall=float("nan"),
            dimensions={d: float("nan") for d in self.dimensions},
            rationale="TODO: implement in evaluation/judges.py",
            judge_model=self.model,
        )

    async def pairwise(
        self,
        persona_a: dict,
        persona_b: dict,
    ) -> tuple[str, str]:
        """Pairwise A/B preference judging.

        Returns (winner, rationale). Always run in both orders (a,b) and
        (b,a) and average — position bias is one of the known failure
        modes per experiment 5.4.

        TODO(space-5): implement.
        """
        return ("tie", "TODO: implement pairwise judging")
