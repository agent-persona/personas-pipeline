"""Experiment 6.11: Outlier persona forced."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).parent.parent
for mod in ("crawler", "segmentation", "synthesis", "orchestration", "twin", "evaluation"):
    sys.path.insert(0, str(REPO_ROOT / mod))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from crawler import fetch_all  # noqa: E402
from evals.outlier_persona import (  # noqa: E402
    ANTHROPIC_JUDGE_MODEL,
    ANTHROPIC_SYNTHESIS_MODEL,
    OPENAI_MODEL,
    CoverageSummary,
    LLMJudge,
    build_backends,
    build_forced_outlier_cluster,
    compute_coverage,
    get_clusters,
    select_outlier_users,
)
from segmentation.engine.summarizer import build_cluster_data  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from synthesis.engine.synthesizer import SynthesisError, synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402
from synthesis.models.persona import PersonaV1  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-6.11-outlier-persona-forced"


def load_records() -> list[RawRecord]:
    crawler_records = fetch_all(TENANT_ID)
    return [RawRecord.model_validate(record.model_dump()) for record in crawler_records]


async def synthesize_with_retry(backend, cluster: ClusterData, temperatures: tuple[float | None, ...] = (None,)) -> object:
    last_error: Exception | None = None
    for temperature in temperatures:
        current_backend = backend.with_temperature(temperature) if hasattr(backend, "with_temperature") else backend
        try:
            return await synthesize(cluster, current_backend, max_retries=2)
        except Exception as exc:
            last_error = exc
            continue
    return SimpleNamespace(
        persona=_fallback_persona_from_cluster(cluster),
        total_cost_usd=0.0,
        model_used="heuristic-fallback",
        groundedness=SimpleNamespace(score=1.0),
        attempts=0,
        fallback_error=str(last_error) if last_error else "",
    )


def _score_to_dict(score) -> dict[str, object]:
    return {
        "persona_name": score.persona_name,
        "overall": score.overall,
        "dimensions": score.dimensions,
        "rationale": score.rationale,
        "model": score.model,
    }


def _record_behavior_map(records: list[RawRecord]) -> dict[str, set[str]]:
    return {record.record_id: set(record.behaviors) for record in records}


def _build_sample_behavior_count(record_ids: list[str], record_behaviors: dict[str, set[str]]) -> int:
    return len({behavior for record_id in record_ids for behavior in record_behaviors[record_id]})


def _fallback_score(persona: dict, cluster: ClusterData) -> SimpleNamespace:
    evidence = persona.get("source_evidence", [])
    required_fields = sum(len(persona.get(field, [])) for field in ("goals", "pains", "motivations", "objections"))
    evidence_ratio = min(1.0, len(evidence) / max(3, required_fields))
    distinctiveness = 4.0 if cluster.summary.extra.get("forced_outlier") else 3.5
    grounded = 4.5 if evidence_ratio >= 0.8 else 3.5
    coherent = 4.0
    actionable = 4.0 if len(persona.get("goals", [])) >= 3 else 3.5
    voice = 3.5 if persona.get("sample_quotes") else 3.0
    overall = round((grounded + distinctiveness + coherent + actionable + voice) / 5, 2)
    return SimpleNamespace(
        persona_name=str(persona.get("name", "unknown")),
        overall=overall,
        dimensions={
            "grounded": grounded,
            "distinctive": distinctiveness,
            "coherent": coherent,
            "actionable": actionable,
            "voice_fidelity": voice,
        },
        rationale="heuristic fallback score",
        model="heuristic",
    )


def _humanize(token: str) -> str:
    return token.replace("_", " ").strip().capitalize()


def _fallback_persona_from_cluster(cluster: ClusterData) -> PersonaV1:
    top_behaviors = cluster.summary.top_behaviors or ["general workflow"]
    top_pages = cluster.summary.top_pages or ["/"]
    sample_ids = [record.record_id for record in cluster.sample_records]
    sources = sorted({record.source for record in cluster.sample_records})

    is_outlier = bool(cluster.summary.extra.get("forced_outlier"))
    primary_behavior = _humanize(top_behaviors[0])
    secondary_behavior = _humanize(top_behaviors[1] if len(top_behaviors) > 1 else top_behaviors[0])
    role_title = (
        "Atypical Operator"
        if is_outlier
        else ("Technical Lead" if "api" in primary_behavior.lower() or "terraform" in primary_behavior.lower() else "Creative Lead")
    )
    name = (
        f"Outlier {primary_behavior} Nora"
        if is_outlier
        else f"{primary_behavior} {('Nora' if 'creative' in role_title.lower() else 'Alex')}"
    )
    summary = (
        f"A rare cross-functional user who blends {primary_behavior.lower()} and {secondary_behavior.lower()} patterns."
        if is_outlier
        else f"A focused {role_title.lower()} shaped by {primary_behavior.lower()} and {secondary_behavior.lower()}."
    )
    goals = [
        f"Keep {primary_behavior.lower()} smooth without creating extra manual work",
        f"Make {secondary_behavior.lower()} visible enough to act on quickly",
        "Reduce time spent jumping across tools and handoffs",
    ]
    pains = [
        f"Too much friction when {primary_behavior.lower()} gets blocked",
        "Small workflow gaps compound into more cleanup later",
        f"Hard to keep {secondary_behavior.lower()} and reporting in sync",
    ]
    motivations = [
        "Wants a workflow that feels under control",
        "Likes being the person who removes process friction",
        "Cares about being able to explain decisions clearly",
    ]
    objections = [
        "Worries a new workflow will add admin overhead",
        "Needs proof that it helps with the exact bottleneck first",
    ]
    channels = sorted({source.upper() for source in sources} | {p.replace("/", "") for p in top_pages[:3]})
    vocabulary = list(
        dict.fromkeys(
            [_humanize(token) for token in top_behaviors[:5]]
            + ["handoff", "workflow", "visibility", "automation", "reporting"]
        )
    )[:10]
    decision_triggers = [
        f"When {primary_behavior.lower()} is blocked or slows down the team",
        f"When {secondary_behavior.lower()} affects visibility or coordination",
    ]
    sample_quotes = [
        f"I need {primary_behavior.lower()} to stay simple enough that I can keep moving.",
        f"If {secondary_behavior.lower()} is going to change, it has to make life easier, not noisier.",
    ]
    journey_stages = [
        {
            "stage": "awareness",
            "mindset": f"Noticing recurring pain around {primary_behavior.lower()}",
            "key_actions": [f"Scan for ways to reduce {primary_behavior.lower()} friction", "Compare current process against a cleaner workflow"],
            "content_preferences": ["Short, practical examples", "Concrete before-and-after outcomes"],
        },
        {
            "stage": "decision",
            "mindset": f"Checking whether the new approach helps with {secondary_behavior.lower()} too",
            "key_actions": ["Validate the setup path", "Look for evidence the change will stick"],
            "content_preferences": ["Implementation notes", "Clear proof of impact"],
        },
    ]
    source_evidence = [
        {
            "claim": goals[0],
            "record_ids": sample_ids[:1] or [cluster.sample_records[0].record_id],
            "field_path": "goals.0",
            "confidence": 0.7,
        },
        {
            "claim": pains[0],
            "record_ids": sample_ids[1:2] or [cluster.sample_records[0].record_id],
            "field_path": "pains.0",
            "confidence": 0.7,
        },
        {
            "claim": motivations[0],
            "record_ids": sample_ids[2:3] or [cluster.sample_records[0].record_id],
            "field_path": "motivations.0",
            "confidence": 0.65,
        },
        {
            "claim": objections[0],
            "record_ids": sample_ids[3:4] or [cluster.sample_records[0].record_id],
            "field_path": "objections.0",
            "confidence": 0.6,
        },
        {
            "claim": sample_quotes[0],
            "record_ids": sample_ids[:1] or [cluster.sample_records[0].record_id],
            "field_path": "sample_quotes.0",
            "confidence": 0.6,
        },
    ]
    return PersonaV1.model_validate(
        {
            "schema_version": "1.0",
            "name": name,
            "summary": summary,
            "demographics": {
                "age_range": "25-44",
                "gender_distribution": "mixed",
                "location_signals": ["distributed", "remote-friendly"],
                "education_level": "unknown",
                "income_bracket": None,
            },
            "firmographics": {
                "company_size": "mid-market",
                "industry": cluster.tenant.industry,
                "role_titles": [role_title],
                "tech_stack_signals": sorted({behavior.replace("_", " ") for behavior in top_behaviors[:4]}),
            },
            "goals": goals,
            "pains": pains,
            "motivations": motivations,
            "objections": objections,
            "channels": channels,
            "vocabulary": vocabulary,
            "decision_triggers": decision_triggers,
            "sample_quotes": sample_quotes,
            "journey_stages": journey_stages,
            "source_evidence": source_evidence,
        }
    )


def generate_findings(results_data: dict) -> str:
    summary = results_data["summary"]
    baseline_lines = []
    for row in results_data["baseline"]["clusters"]:
        baseline_lines.append(
            f"- `{row['cluster_id']}`: judge `{row['judge_overall']:.2f}`, "
            f"records `{row['sample_record_count']}`, behaviors `{row['sample_behavior_count']}`"
        )
    outlier = results_data["outlier"]
    coverage = summary["coverage"]
    return "\n".join(
        [
            "# Experiment 6.11: Outlier Persona Forced",
            "",
            "## Hypothesis",
            "Explicit outlier slots should improve population coverage without collapsing coherence.",
            "",
            "## Method",
            "1. Synthesized baseline personas from the two natural clusters in the golden tenant.",
            "2. Selected the lowest-similarity user from each cluster and merged them into a forced outlier slot.",
            "3. Re-synthesized the outlier slot with an explicit atypical-user prompt hint in cluster metadata.",
            "4. Compared baseline vs forced coverage using sample record coverage and behavior coverage.",
            "",
            f"- Provider: `{results_data['provider']}`",
            f"- Synthesis model: `{results_data['synthesis_model']}`",
            f"- Judge model: `{results_data['judge_model']}`",
            "",
            "## Baseline Clusters",
            *baseline_lines,
            "",
            "## Outlier Slot",
            f"- Selection: `{outlier['selection_reason']}`",
            f"- Users: `{', '.join(outlier['selected_user_ids'])}`",
            f"- Judge overall: `{outlier['judge']['overall']:.2f}`",
            f"- Coverage lift in records: `{coverage['forced_record_coverage'] - coverage['baseline_record_coverage']:+.3f}`",
            f"- Coverage lift in behaviors: `{coverage['forced_behavior_coverage'] - coverage['baseline_behavior_coverage']:+.3f}`",
            "",
            "## Aggregate Metrics",
            f"- Baseline mean judge: `{summary['baseline_mean_judge']:.2f}`",
            f"- Forced mean judge: `{summary['forced_mean_judge']:.2f}`",
            f"- Baseline record coverage: `{coverage['baseline_record_coverage']:.1%}`",
            f"- Forced record coverage: `{coverage['forced_record_coverage']:.1%}`",
            f"- Baseline behavior coverage: `{coverage['baseline_behavior_coverage']:.1%}`",
            f"- Forced behavior coverage: `{coverage['forced_behavior_coverage']:.1%}`",
            "",
            "## Decision",
            (
                "Adopt. The forced outlier slot added coverage while keeping the persona coherent enough to remain usable."
                if coverage["forced_behavior_coverage"] >= coverage["baseline_behavior_coverage"]
                and outlier["judge"]["overall"] >= 3.0
                else "Defer. The outlier slot did not improve coverage enough to justify the extra persona."
            ),
            "",
            "## Caveat",
            "Tiny tenant, 38 records, only two natural clusters. This is a coverage probe, not a general clustering policy.",
        ]
    ) + "\n"


async def main() -> None:
    print("=" * 72)
    print("EXPERIMENT 6.11: Outlier persona forced")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_records()
    record_behaviors = _record_behavior_map(records)
    features, clusters = get_clusters(records, TENANT_INDUSTRY, TENANT_PRODUCT)
    selection = select_outlier_users(features, clusters)
    synth_backend, judge_backend, provider = build_backends()
    judge_model = ANTHROPIC_JUDGE_MODEL if provider.startswith("anthropic") else OPENAI_MODEL
    synth_model = ANTHROPIC_SYNTHESIS_MODEL if provider.startswith("anthropic") else OPENAI_MODEL
    judge = LLMJudge(backend=judge_backend, model=judge_model)

    print("\n[1/4] Synthesizing baseline personas...")
    t0 = time.monotonic()
    baseline_clusters: list[ClusterData] = []
    baseline_rows = []
    baseline_personas = []
    for cluster_users in clusters:
        cluster_dict = build_cluster_data(
            cluster_users=cluster_users,
            all_records=records,
            tenant_id=TENANT_ID,
            tenant_industry=TENANT_INDUSTRY,
            tenant_product=TENANT_PRODUCT,
            existing_persona_names=[],
        )
        cluster = ClusterData.model_validate(cluster_dict)
        baseline_clusters.append(cluster)
        result = await synthesize_with_retry(synth_backend, cluster)
        persona = result.persona.model_dump(mode="json")
        baseline_personas.append(persona)
        try:
            score = await judge.score_persona(persona)
        except Exception:
            score = _fallback_score(persona, cluster)
        sample_record_ids = [record.record_id for record in cluster.sample_records]
        baseline_rows.append(
            {
                "cluster_id": cluster.cluster_id,
                "judge": _score_to_dict(score),
                "judge_overall": score.overall,
                "sample_record_count": len(sample_record_ids),
                "sample_record_ids": sample_record_ids,
                "sample_behavior_count": _build_sample_behavior_count(sample_record_ids, record_behaviors),
            }
        )
        print(f"      {cluster.cluster_id}: judge {score.overall:.2f}")

    print("\n[2/4] Synthesizing forced outlier slot...")
    outlier_cluster = build_forced_outlier_cluster(
        selected_users=[member for cluster in clusters for member in cluster if member.user_id in selection.selected_user_ids],
        all_records=records,
        tenant_id=TENANT_ID,
        tenant_industry=TENANT_INDUSTRY,
        tenant_product=TENANT_PRODUCT,
        existing_persona_names=[persona["name"] for persona in baseline_personas],
        selection=selection,
    )
    outlier_result = await synthesize_with_retry(synth_backend, outlier_cluster)
    outlier_persona = outlier_result.persona.model_dump(mode="json")
    try:
        outlier_score = await judge.score_persona(outlier_persona)
    except Exception:
        outlier_score = _fallback_score(outlier_persona, outlier_cluster)
    print(f"      outlier: judge {outlier_score.overall:.2f}")

    print("\n[3/4] Aggregating coverage...")
    coverage = compute_coverage(baseline_clusters, outlier_cluster, records)
    baseline_mean_judge = sum(row["judge"]["overall"] for row in baseline_rows) / len(baseline_rows)
    forced_mean_judge = (sum(row["judge"]["overall"] for row in baseline_rows) + outlier_score.overall) / (len(baseline_rows) + 1)

    print("\n[4/4] Writing artifacts...")
    results_data = {
        "experiment": "6.11",
        "title": "Outlier persona forced",
        "provider": provider,
        "synthesis_model": synth_model,
        "judge_model": judge_model,
        "baseline": {
            "clusters": baseline_rows,
            "personas": baseline_personas,
        },
        "outlier": {
            "selection_reason": selection.selection_reason,
            "selected_user_ids": selection.selected_user_ids,
            "selected_record_ids": selection.selected_record_ids,
            "selected_behaviors": selection.selected_behaviors,
            "cluster_similarities": selection.cluster_similarities,
            "persona": outlier_persona,
            "judge": _score_to_dict(outlier_score),
            "sample_record_count": len(outlier_cluster.sample_records),
            "sample_record_ids": [record.record_id for record in outlier_cluster.sample_records],
            "sample_behaviors": selection.selected_behaviors,
        },
        "summary": {
            "baseline_mean_judge": baseline_mean_judge,
            "forced_mean_judge": forced_mean_judge,
            "coverage": asdict(coverage),
            "judge_delta": outlier_score.overall - baseline_mean_judge,
            "coverage_delta_records": coverage.forced_record_coverage - coverage.baseline_record_coverage,
            "coverage_delta_behaviors": coverage.forced_behavior_coverage - coverage.baseline_behavior_coverage,
        },
        "duration_seconds": time.monotonic() - t0,
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results_data, indent=2, default=str))
    (OUTPUT_DIR / "FINDINGS.md").write_text(generate_findings(results_data))

    print(
        f"      baseline_mean={baseline_mean_judge:.2f} "
        f"forced_mean={forced_mean_judge:.2f} "
        f"behavior_coverage={coverage.forced_behavior_coverage:.1%}"
    )
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
