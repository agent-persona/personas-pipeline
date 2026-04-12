"""Experiment 5.22: Eval contamination check.

Probes LLM judges for memorization of public persona datasets. If a judge
has seen similar personas during training, it might rate them higher from
familiarity rather than actual quality — contaminating all eval results.

Three probe types:
  1. **Completion probe**: Give the model the first half of a known public
     persona and ask it to complete. High completion accuracy = memorization.
  2. **Attribution probe**: Ask "Have you seen a persona like this in your
     training data?" for known public vs our novel synthetic personas.
  3. **Familiarity scoring**: Score known public personas and novel synthetic
     personas with the same judge rubric. If public ones score systematically
     higher, familiarity bias exists.

Known public persona sources tested:
  - persona-hub (Tao et al., 2024) — large persona dataset
  - TinyTroupe (Microsoft) — simulation personas
  - HubSpot buyer personas — marketing examples
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field

from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


# ── Known public persona fragments ───────────────────────────────────
# Fragments from well-known public persona datasets that an LLM may have
# seen during training. These are paraphrased to avoid exact reproduction.

PUBLIC_PERSONA_PROBES = [
    {
        "source": "persona-hub",
        "fragment": "A 35-year-old software engineer at a mid-size tech company in San Francisco who specializes in backend systems and is passionate about open-source contributions",
        "completion_seed": "A 35-year-old software engineer at a mid-size tech company in San Francisco who specializes in",
        "expected_keywords": ["backend", "open-source", "contributions", "engineer"],
    },
    {
        "source": "persona-hub",
        "fragment": "A retired schoolteacher in rural Iowa who spends her time gardening, volunteering at the local library, and writing letters to her grandchildren",
        "completion_seed": "A retired schoolteacher in rural Iowa who spends her time gardening, volunteering at",
        "expected_keywords": ["library", "grandchildren", "letters", "volunteering"],
    },
    {
        "source": "TinyTroupe",
        "fragment": "Oscar is a 30-year-old architect from Brazil who works at a large multinational firm and is known for his innovative sustainable designs",
        "completion_seed": "Oscar is a 30-year-old architect from Brazil who works at a large multinational firm and is known for",
        "expected_keywords": ["sustainable", "designs", "innovative", "architect"],
    },
    {
        "source": "TinyTroupe",
        "fragment": "Lisa is a 28-year-old data scientist living in Berlin who is interested in machine learning applications in healthcare and ethical AI",
        "completion_seed": "Lisa is a 28-year-old data scientist living in Berlin who is interested in",
        "expected_keywords": ["machine learning", "healthcare", "ethical", "AI"],
    },
    {
        "source": "HubSpot examples",
        "fragment": "Marketing Mary is a VP of Marketing at a B2B SaaS company with 200 employees. She is responsible for lead generation and is frustrated by the inability to prove ROI on marketing spend",
        "completion_seed": "Marketing Mary is a VP of Marketing at a B2B SaaS company with 200 employees. She is responsible for",
        "expected_keywords": ["lead generation", "ROI", "marketing spend", "frustrated"],
    },
]

# Novel synthetic fragments that should NOT be memorized
NOVEL_PERSONA_PROBES = [
    {
        "source": "novel",
        "fragment": "Kazimir is a 42-year-old apiary technologist in rural Lithuania who develops IoT sensors for beehive health monitoring and sells artisanal mead at farmers markets",
        "completion_seed": "Kazimir is a 42-year-old apiary technologist in rural Lithuania who develops IoT sensors for",
        "expected_keywords": ["beehive", "mead", "sensors", "monitoring"],
    },
    {
        "source": "novel",
        "fragment": "Priya is a 31-year-old fermentation scientist at a synthetic biology startup in Pune who optimizes bioreactor yield curves and moonlights as a competitive badminton player",
        "completion_seed": "Priya is a 31-year-old fermentation scientist at a synthetic biology startup in Pune who optimizes",
        "expected_keywords": ["bioreactor", "yield curves", "badminton", "fermentation"],
    },
    {
        "source": "novel",
        "fragment": "Torben is a 55-year-old decommissioning specialist for offshore wind farms in the North Sea who navigates complex environmental regulations and runs a podcast about marine ecology",
        "completion_seed": "Torben is a 55-year-old decommissioning specialist for offshore wind farms in the North Sea who navigates",
        "expected_keywords": ["decommissioning", "wind farms", "marine ecology", "regulations"],
    },
]


# ── Data types ───────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    source: str
    probe_type: str  # completion, attribution
    fragment: str
    model_response: str
    keywords_matched: int
    keywords_total: int
    hit_rate: float  # keywords_matched / keywords_total
    is_memorized: bool  # hit_rate > threshold
    cost_usd: float = 0.0


@dataclass
class ContaminationReport:
    model: str
    public_probes: list[ProbeResult] = field(default_factory=list)
    novel_probes: list[ProbeResult] = field(default_factory=list)
    public_hit_rate: float = 0.0
    novel_hit_rate: float = 0.0
    contamination_delta: float = 0.0  # public - novel
    is_contaminated: bool = False
    total_cost_usd: float = 0.0


# ── Probe runners ────────────────────────────────────────────────────

COMPLETION_PROMPT = """\
Complete the following persona description in 1-2 sentences. Continue \
naturally from where it left off. Do NOT say you cannot complete it — \
just write the most likely continuation.

"""

ATTRIBUTION_PROMPT = """\
Read the persona description below. On a scale of 0-10, how familiar \
does this feel — as if you may have encountered similar examples in \
training data, textbooks, or well-known persona frameworks?

Respond with ONLY a JSON object:
{"familiarity_score": <int 0-10>, "reasoning": "<brief explanation>"}

Persona:
"""


async def run_completion_probe(
    client: AsyncAnthropic,
    model: str,
    probe: dict,
    memorization_threshold: float = 0.5,
) -> ProbeResult:
    """Test if the model can complete a known persona fragment."""
    prompt = COMPLETION_PROMPT + probe["completion_seed"]

    response = await client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip().lower()
    in_tok = response.usage.input_tokens
    out_tok = response.usage.output_tokens
    if "haiku" in model:
        cost = (in_tok * 1 + out_tok * 5) / 1_000_000
    elif "opus" in model:
        cost = (in_tok * 15 + out_tok * 75) / 1_000_000
    else:
        cost = (in_tok * 3 + out_tok * 15) / 1_000_000

    expected = probe["expected_keywords"]
    matched = sum(1 for kw in expected if kw.lower() in text)
    hit_rate = matched / len(expected) if expected else 0.0

    return ProbeResult(
        source=probe["source"],
        probe_type="completion",
        fragment=probe["completion_seed"][:60],
        model_response=text[:120],
        keywords_matched=matched,
        keywords_total=len(expected),
        hit_rate=hit_rate,
        is_memorized=hit_rate >= memorization_threshold,
        cost_usd=cost,
    )


async def run_attribution_probe(
    client: AsyncAnthropic,
    model: str,
    probe: dict,
) -> ProbeResult:
    """Ask the model how familiar a persona feels."""
    prompt = ATTRIBUTION_PROMPT + probe["fragment"]

    response = await client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

    in_tok = response.usage.input_tokens
    out_tok = response.usage.output_tokens
    if "haiku" in model:
        cost = (in_tok * 1 + out_tok * 5) / 1_000_000
    elif "opus" in model:
        cost = (in_tok * 15 + out_tok * 75) / 1_000_000
    else:
        cost = (in_tok * 3 + out_tok * 15) / 1_000_000

    try:
        parsed = json.loads(text)
        fam_score = int(parsed.get("familiarity_score", 0))
    except (json.JSONDecodeError, ValueError):
        fam_score = 0

    # familiarity > 6 = likely memorized
    hit_rate = fam_score / 10.0

    return ProbeResult(
        source=probe["source"],
        probe_type="attribution",
        fragment=probe["fragment"][:60],
        model_response=text[:120],
        keywords_matched=fam_score,
        keywords_total=10,
        hit_rate=hit_rate,
        is_memorized=fam_score >= 7,
        cost_usd=cost,
    )


# ── Full contamination check ────────────────────────────────────────

async def check_contamination(
    client: AsyncAnthropic,
    model: str,
    memorization_threshold: float = 0.5,
) -> ContaminationReport:
    """Run all probes and produce a contamination report."""
    import asyncio

    report = ContaminationReport(model=model)

    # Run completion probes on public fragments
    public_tasks = [
        run_completion_probe(client, model, p, memorization_threshold)
        for p in PUBLIC_PERSONA_PROBES
    ]
    novel_tasks = [
        run_completion_probe(client, model, p, memorization_threshold)
        for p in NOVEL_PERSONA_PROBES
    ]

    # Run attribution probes
    public_attr_tasks = [
        run_attribution_probe(client, model, p)
        for p in PUBLIC_PERSONA_PROBES
    ]
    novel_attr_tasks = [
        run_attribution_probe(client, model, p)
        for p in NOVEL_PERSONA_PROBES
    ]

    all_results = await asyncio.gather(
        *public_tasks, *novel_tasks, *public_attr_tasks, *novel_attr_tasks,
        return_exceptions=True,
    )

    n_pub = len(PUBLIC_PERSONA_PROBES)
    n_nov = len(NOVEL_PERSONA_PROBES)

    # Split results
    pub_completion = all_results[:n_pub]
    nov_completion = all_results[n_pub:n_pub + n_nov]
    pub_attribution = all_results[n_pub + n_nov:n_pub + n_nov + n_pub]
    nov_attribution = all_results[n_pub + n_nov + n_pub:]

    for r in pub_completion + pub_attribution:
        if isinstance(r, ProbeResult):
            report.public_probes.append(r)
            report.total_cost_usd += r.cost_usd
        elif isinstance(r, Exception):
            logger.warning("Public probe failed: %s", r)

    for r in nov_completion + nov_attribution:
        if isinstance(r, ProbeResult):
            report.novel_probes.append(r)
            report.total_cost_usd += r.cost_usd
        elif isinstance(r, Exception):
            logger.warning("Novel probe failed: %s", r)

    # Compute aggregate rates
    pub_rates = [p.hit_rate for p in report.public_probes]
    nov_rates = [p.hit_rate for p in report.novel_probes]

    report.public_hit_rate = statistics.mean(pub_rates) if pub_rates else 0.0
    report.novel_hit_rate = statistics.mean(nov_rates) if nov_rates else 0.0
    report.contamination_delta = report.public_hit_rate - report.novel_hit_rate
    report.is_contaminated = report.contamination_delta > 0.15

    return report
