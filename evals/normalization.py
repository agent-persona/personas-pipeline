#!/usr/bin/env python3
"""Experiment 5.21 — Per-tenant eval normalization.

Goal: Compute per-tenant baseline scores, then report judge scores relative
to that baseline.  Test whether normalization makes model-variant rankings
more stable.

Method:
1. Synthesize personas from the standard tenant (tenant_acme_corp, 2 clusters)
2. Judge each persona with an LLM judge (score 1-5 on grounded, distinctive,
   coherent, actionable, voice_fidelity)
3. Compute the tenant baseline: the mean score across all personas for each
   dimension
4. Report both absolute scores and normalized scores
   (score - baseline_mean) / baseline_std
5. Simulate a "variant ranking" scenario: re-judge with slightly different
   prompt instructions, see if the ranking of personas is more stable when
   using normalized vs absolute scores
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from statistics import mean, stdev

# ── path setup ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic
from synthesis.engine.model_backend import AnthropicBackend
from synthesis.engine.synthesizer import synthesize
from synthesis.models.cluster import (
    ClusterData,
    ClusterSummary,
    EnrichmentPayload,
    SampleRecord,
    TenantContext,
)
from evaluation.judges import JudgeScore, LLMJudge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("exp-5.21")

# ── constants ───────────────────────────────────────────────────────────
DIMENSIONS = ("grounded", "distinctive", "coherent", "actionable", "voice_fidelity")
JUDGE_MODEL = "claude-haiku-4-5-20251001"
SYNTH_MODEL = "claude-haiku-4-5-20251001"

# ── test data: two clusters for tenant_acme_corp ────────────────────────

TENANT = TenantContext(
    tenant_id="tenant_acme_corp",
    industry="B2B SaaS — project management",
    product_description="Cloud-based project management platform for mid-market teams",
    existing_persona_names=[],
)

CLUSTERS = [
    ClusterData(
        cluster_id="cluster_acme_001",
        tenant=TENANT,
        summary=ClusterSummary(
            cluster_size=128,
            top_behaviors=["creates_projects_weekly", "invites_team_members", "uses_gantt_view"],
            top_pages=["/dashboard", "/projects/new", "/gantt"],
            conversion_rate=0.12,
            avg_session_duration_seconds=420.0,
            top_referrers=["google", "linkedin"],
        ),
        sample_records=[
            SampleRecord(
                record_id="rec_a001",
                source="ga4",
                timestamp="2025-11-15T10:23:00Z",
                payload={
                    "page_path": "/projects/new",
                    "session_duration": 380,
                    "events": "create_project,invite_member,set_deadline",
                },
            ),
            SampleRecord(
                record_id="rec_a002",
                source="hubspot",
                payload={
                    "company_size": "50-200",
                    "role": "Project Manager",
                    "industry": "Technology",
                    "lead_score": 82,
                },
            ),
            SampleRecord(
                record_id="rec_a003",
                source="ga4",
                timestamp="2025-11-16T14:05:00Z",
                payload={
                    "page_path": "/gantt",
                    "session_duration": 540,
                    "events": "view_gantt,export_pdf,share_view",
                },
            ),
            SampleRecord(
                record_id="rec_a004",
                source="hubspot",
                payload={
                    "company_size": "50-200",
                    "role": "Team Lead",
                    "deal_stage": "evaluation",
                    "notes": "Needs Gantt charts and resource allocation views",
                },
            ),
        ],
        enrichment=EnrichmentPayload(
            firmographic={"avg_company_size": "50-200", "primary_industry": "Technology"},
            intent_signals=["project management comparison", "gantt chart software"],
            technographic={"current_tools": "Jira, Slack, Google Workspace"},
        ),
    ),
    ClusterData(
        cluster_id="cluster_acme_002",
        tenant=TENANT,
        summary=ClusterSummary(
            cluster_size=95,
            top_behaviors=["views_pricing_page", "reads_case_studies", "downloads_whitepaper"],
            top_pages=["/pricing", "/case-studies", "/resources/whitepaper"],
            conversion_rate=0.07,
            avg_session_duration_seconds=210.0,
            top_referrers=["google", "capterra"],
        ),
        sample_records=[
            SampleRecord(
                record_id="rec_b001",
                source="ga4",
                timestamp="2025-11-14T09:12:00Z",
                payload={
                    "page_path": "/pricing",
                    "session_duration": 180,
                    "events": "view_pricing,compare_plans,click_enterprise",
                },
            ),
            SampleRecord(
                record_id="rec_b002",
                source="hubspot",
                payload={
                    "company_size": "200-1000",
                    "role": "VP Operations",
                    "industry": "Financial Services",
                    "lead_score": 65,
                },
            ),
            SampleRecord(
                record_id="rec_b003",
                source="ga4",
                timestamp="2025-11-15T16:30:00Z",
                payload={
                    "page_path": "/case-studies/fintech-co",
                    "session_duration": 290,
                    "events": "read_case_study,download_pdf",
                },
            ),
            SampleRecord(
                record_id="rec_b004",
                source="capterra",
                payload={
                    "review_rating": 4,
                    "review_snippet": "Good for mid-size teams but enterprise features are lacking",
                    "reviewer_role": "Director of PMO",
                },
            ),
        ],
        enrichment=EnrichmentPayload(
            firmographic={"avg_company_size": "200-1000", "primary_industry": "Financial Services"},
            intent_signals=["enterprise project management", "project portfolio management"],
            technographic={"current_tools": "MS Project, Teams, Salesforce"},
        ),
    ),
]


# ── LLM judge implementation ───────────────────────────────────────────

JUDGE_RUBRIC = """\
You are an expert persona evaluator. You will be given a synthesized persona
(as JSON) and must rate it on these five dimensions using a 1-5 scale:

1. **grounded** (1-5): Are the persona's claims traceable to the source data?
   1 = fabricated, 5 = every claim cites real records.
2. **distinctive** (1-5): Does this feel like a real individual vs a generic average?
   1 = bland template, 5 = vivid and specific.
3. **coherent** (1-5): Is the persona internally consistent across all fields?
   1 = contradictory, 5 = perfectly consistent.
4. **actionable** (1-5): Are the goals/pains sharp enough to drive product decisions?
   1 = vague platitudes, 5 = directly usable in a product brief.
5. **voice_fidelity** (1-5): Do the sample quotes sound like one consistent person?
   1 = robotic / inconsistent, 5 = natural and unified voice.

{extra_instruction}

Return your response as a JSON object with this exact structure:
{{
  "grounded": <int 1-5>,
  "distinctive": <int 1-5>,
  "coherent": <int 1-5>,
  "actionable": <int 1-5>,
  "voice_fidelity": <int 1-5>,
  "rationale": "<brief explanation for each score>"
}}

Return ONLY the JSON object, no markdown fences or other text.
"""

VARIANT_EXTRA = (
    "Pay extra attention to specificity of language. "
    "Personas that use concrete numbers, dates, or tool names should score "
    "higher on grounded and actionable."
)


async def judge_persona(
    client: AsyncAnthropic,
    persona_dict: dict,
    model: str = JUDGE_MODEL,
    extra_instruction: str = "",
) -> JudgeScore:
    """Score a persona dict using an LLM judge. Returns a JudgeScore."""
    rubric = JUDGE_RUBRIC.format(extra_instruction=extra_instruction)
    persona_json = json.dumps(persona_dict, indent=2)

    response = await client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{rubric}\n\nHere is the persona to evaluate:\n\n{persona_json}"
                ),
            }
        ],
    )

    raw_text = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
        if raw_text.endswith("```"):
            raw_text = raw_text[: raw_text.rfind("```")]
    parsed = json.loads(raw_text)

    dims = {d: float(parsed[d]) for d in DIMENSIONS}
    overall = mean(dims.values())

    return JudgeScore(
        overall=overall,
        dimensions=dims,
        rationale=parsed.get("rationale", ""),
        judge_model=model,
    )


# ── normalization helpers ───────────────────────────────────────────────

def compute_baseline(scores_list: list[JudgeScore]) -> dict[str, tuple[float, float]]:
    """Return {dimension: (mean, std)} across all personas in the tenant."""
    baseline: dict[str, tuple[float, float]] = {}
    for dim in DIMENSIONS:
        vals = [s.dimensions[dim] for s in scores_list]
        mu = mean(vals)
        sd = stdev(vals) if len(vals) > 1 else 1.0  # avoid div-by-zero
        baseline[dim] = (mu, sd)
    return baseline


def normalize_scores(
    score: JudgeScore,
    baseline: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Return normalized scores: (raw - mean) / std for each dimension."""
    normed: dict[str, float] = {}
    for dim in DIMENSIONS:
        mu, sd = baseline[dim]
        if sd == 0:
            normed[dim] = 0.0
        else:
            normed[dim] = (score.dimensions[dim] - mu) / sd
    return normed


def rank_personas(
    scores: dict[str, float | dict[str, float]],
) -> list[tuple[str, float]]:
    """Rank persona names by their overall (or mean-normalized) score, descending."""
    items: list[tuple[str, float]] = []
    for name, val in scores.items():
        if isinstance(val, dict):
            items.append((name, mean(val.values())))
        else:
            items.append((name, val))
    return sorted(items, key=lambda x: x[1], reverse=True)


def ranking_stability(
    rank_a: list[tuple[str, float]],
    rank_b: list[tuple[str, float]],
) -> float:
    """Compute ranking stability as Kendall-tau-like agreement.

    Returns fraction of pairwise orderings that agree between the two rankings.
    1.0 = identical ordering, 0.0 = fully reversed.
    """
    names_a = [name for name, _ in rank_a]
    names_b = [name for name, _ in rank_b]
    n = len(names_a)
    if n < 2:
        return 1.0
    concordant = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            pos_a_i = names_a.index(rank_a[i][0])
            pos_a_j = names_a.index(rank_a[j][0])
            pos_b_i = names_b.index(rank_a[i][0])
            pos_b_j = names_b.index(rank_a[j][0])
            if (pos_a_i - pos_a_j) * (pos_b_i - pos_b_j) > 0:
                concordant += 1
            elif (pos_a_i - pos_a_j) * (pos_b_i - pos_b_j) == 0:
                concordant += 0.5  # tie
            total += 1
    return concordant / total if total > 0 else 1.0


# ── main experiment ─────────────────────────────────────────────────────

async def run_experiment() -> dict:
    """Run the full experiment 5.21 pipeline."""
    client = AsyncAnthropic()
    backend = AnthropicBackend(client=client, model=SYNTH_MODEL)

    # ── Step 1: synthesize personas ─────────────────────────────────────
    logger.info("Step 1: Synthesizing personas from %d clusters", len(CLUSTERS))
    personas: dict[str, dict] = {}
    for cluster in CLUSTERS:
        result = await synthesize(cluster, backend)
        persona_dict = result.persona.model_dump()
        name = persona_dict.get("name", cluster.cluster_id)
        personas[name] = persona_dict
        logger.info("  Synthesized persona: %s (cost=$%.4f)", name, result.total_cost_usd)

    # ── Step 2: judge each persona (default prompt) ─────────────────────
    logger.info("Step 2: Judging personas (default rubric)")
    default_scores: dict[str, JudgeScore] = {}
    for name, persona_dict in personas.items():
        score = await judge_persona(client, persona_dict)
        default_scores[name] = score
        logger.info("  %s — overall=%.2f  dims=%s", name, score.overall,
                     {d: f"{v:.1f}" for d, v in score.dimensions.items()})

    # ── Step 3: compute tenant baseline ─────────────────────────────────
    logger.info("Step 3: Computing tenant baseline")
    baseline = compute_baseline(list(default_scores.values()))
    for dim, (mu, sd) in baseline.items():
        logger.info("  %s: mean=%.2f  std=%.2f", dim, mu, sd)

    # ── Step 4: report absolute vs normalized ───────────────────────────
    logger.info("Step 4: Absolute vs normalized scores")
    default_normalized: dict[str, dict[str, float]] = {}
    for name, score in default_scores.items():
        normed = normalize_scores(score, baseline)
        default_normalized[name] = normed
        logger.info("  %s absolute: %s", name,
                     {d: f"{v:.1f}" for d, v in score.dimensions.items()})
        logger.info("  %s normalized: %s", name,
                     {d: f"{v:.2f}" for d, v in normed.items()})

    # ── Step 5: variant ranking stability ───────────────────────────────
    logger.info("Step 5: Variant ranking stability test")
    logger.info("  Re-judging with variant prompt instruction...")
    variant_scores: dict[str, JudgeScore] = {}
    for name, persona_dict in personas.items():
        score = await judge_persona(
            client, persona_dict, extra_instruction=VARIANT_EXTRA,
        )
        variant_scores[name] = score
        logger.info("  %s (variant) — overall=%.2f  dims=%s", name, score.overall,
                     {d: f"{v:.1f}" for d, v in score.dimensions.items()})

    variant_baseline = compute_baseline(list(variant_scores.values()))
    variant_normalized: dict[str, dict[str, float]] = {}
    for name, score in variant_scores.items():
        normed = normalize_scores(score, variant_baseline)
        variant_normalized[name] = normed

    # Absolute rankings
    abs_rank_default = rank_personas(
        {n: s.overall for n, s in default_scores.items()}
    )
    abs_rank_variant = rank_personas(
        {n: s.overall for n, s in variant_scores.items()}
    )
    abs_stability = ranking_stability(abs_rank_default, abs_rank_variant)

    # Normalized rankings
    norm_rank_default = rank_personas(default_normalized)
    norm_rank_variant = rank_personas(variant_normalized)
    norm_stability = ranking_stability(norm_rank_default, norm_rank_variant)

    logger.info("=== RESULTS ===")
    logger.info("Absolute ranking (default): %s",
                [(n, f"{s:.2f}") for n, s in abs_rank_default])
    logger.info("Absolute ranking (variant): %s",
                [(n, f"{s:.2f}") for n, s in abs_rank_variant])
    logger.info("Absolute ranking stability: %.2f", abs_stability)
    logger.info("")
    logger.info("Normalized ranking (default): %s",
                [(n, f"{s:.2f}") for n, s in norm_rank_default])
    logger.info("Normalized ranking (variant): %s",
                [(n, f"{s:.2f}") for n, s in norm_rank_variant])
    logger.info("Normalized ranking stability: %.2f", norm_stability)
    logger.info("")
    logger.info(
        "Normalization %s ranking stability (abs=%.2f, norm=%.2f)",
        "IMPROVED" if norm_stability >= abs_stability else "DID NOT IMPROVE",
        abs_stability,
        norm_stability,
    )

    results = {
        "tenant": TENANT.tenant_id,
        "num_clusters": len(CLUSTERS),
        "personas": list(personas.keys()),
        "baseline": {d: {"mean": mu, "std": sd} for d, (mu, sd) in baseline.items()},
        "default_scores": {
            n: {"overall": s.overall, "dimensions": s.dimensions}
            for n, s in default_scores.items()
        },
        "variant_scores": {
            n: {"overall": s.overall, "dimensions": s.dimensions}
            for n, s in variant_scores.items()
        },
        "default_normalized": default_normalized,
        "variant_normalized": variant_normalized,
        "absolute_ranking_stability": abs_stability,
        "normalized_ranking_stability": norm_stability,
        "normalization_helps": norm_stability >= abs_stability,
    }

    # Write results to file
    output_path = Path(__file__).resolve().parent / "normalization_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s", output_path)

    return results


if __name__ == "__main__":
    asyncio.run(run_experiment())
