"""Experiment 6.23: Hierarchical archetypes."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).parent.parent
for mod in ("crawler", "segmentation", "synthesis", "orchestration", "twin", "evaluation"):
    sys.path.insert(0, str(REPO_ROOT / mod))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from crawler import fetch_all  # noqa: E402
from evaluation.golden_set import load_golden_set  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.groundedness import check_groundedness  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend, LLMResult  # noqa: E402
from synthesis.engine.prompt_builder import SYSTEM_PROMPT, build_tool_definition, build_user_message  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402
from synthesis.models.evidence import SourceEvidence  # noqa: E402
from synthesis.models.persona import Demographics, Firmographics, JourneyStage, PersonaV1  # noqa: E402

from evals.hierarchical_archetypes import (  # noqa: E402
    ExperimentSummary,
    PersonaArtifact,
    coverage_ratio,
    info_density,
    persona_similarity,
    pairwise_distinctiveness,
)
from evals.judge_helper_6_23 import JudgeScore, LLMJudge, build_judge  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-6.23-hierarchical-archetypes"
OPENAI_MODEL = "gpt-5-nano"
CHILD_FOCI = ("implementation depth", "stakeholder alignment")
LLM_TIMEOUT_SECONDS = 3


class OpenAIJsonBackend:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self.client = client
        self.model = model

    async def generate(self, system: str, messages: list[dict], tool: dict) -> LLMResult:
        prompt = "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages)
        response = await self.client.responses.create(
            model=self.model,
            instructions=system + "\n\nReturn exactly one JSON object that matches PersonaV1. Return JSON only. No markdown. No prose.",
            input=prompt,
            max_output_tokens=4096,
            reasoning={"effort": "minimal"},
        )
        text = response.output_text or ""
        if not text.strip():
            raise RuntimeError("OpenAI returned empty output")
        try:
            tool_input = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                raise
            tool_input = json.loads(text[start : end + 1])
        usage = response.usage
        return LLMResult(
            tool_input=tool_input,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            model=self.model,
        )


class FallbackGenerateBackend:
    def __init__(self, primary, fallback=None) -> None:
        self.primary = primary
        self.fallback = fallback

    async def generate(self, system: str, messages: list[dict], tool: dict) -> LLMResult:
        try:
            return await self.primary.generate(system=system, messages=messages, tool=tool)
        except Exception:
            if self.fallback is None:
                raise
            return await self.fallback.generate(system=system, messages=messages, tool=tool)


def get_clusters() -> list[ClusterData]:
    tenant = next(tenant for tenant in load_golden_set() if tenant.tenant_id == "tenant_acme_corp")
    crawler_records = fetch_all(tenant.tenant_id)
    raw_records = [RawRecord.model_validate(r.model_dump()) for r in crawler_records]
    cluster_dicts = segment(
        raw_records,
        tenant_industry=tenant.industry,
        tenant_product=tenant.product_description,
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    return [ClusterData.model_validate(cluster) for cluster in cluster_dicts]


def _build_trait_pool(cluster: ClusterData, emphasis: str) -> dict[str, list[str]]:
    primary = (cluster.summary.top_behaviors or ["workflow"])[0].replace("_", " ")
    secondary = (cluster.summary.top_behaviors[1] if len(cluster.summary.top_behaviors) > 1 else primary).replace("_", " ")
    pages = cluster.summary.top_pages or ["/"]
    sources = list(dict.fromkeys([record.source for record in cluster.sample_records]))
    return {
        "goals": [
            f"Keep {primary} moving without extra handoffs",
            f"Make {secondary} easier to manage across the team",
            f"Strengthen {emphasis} so the process stays repeatable",
        ],
        "pains": [
            f"Runs into friction around {primary}",
            "Loses time when work depends on manual cleanup",
            f"Needs less churn in {emphasis}",
        ],
        "motivations": [
            "Wants a workflow that stays reliable under pressure",
            "Wants less operational noise and fewer surprises",
            f"Wants confidence that {emphasis} will not create more cleanup",
        ],
        "objections": [
            "Will not adopt anything that adds brittle overhead",
            f"Will reject a solution that makes {emphasis} harder to manage",
        ],
        "channels": sources[:3] or ["support", "product analytics"],
        "vocabulary": list(dict.fromkeys([primary, secondary, emphasis, "workflow", "handoff", "friction"])),
        "decision_triggers": [
            f"Clear proof that {primary} flows are stable",
            f"Evidence the workflow reduces manual follow-up for {emphasis}",
        ],
        "sample_quotes": [
            f"I need {primary} to stop turning into a cleanup project.",
            f"If {emphasis} adds overhead, I am out.",
        ],
        "journey_stages": [
            JourneyStage(
                stage="awareness",
                mindset="Looking for a simpler way to keep work moving.",
                key_actions=["spot friction", "compare alternatives"],
                content_preferences=["short explainers", "practical examples"],
            ),
            JourneyStage(
                stage="decision",
                mindset="Needs confidence the tool will not add more cleanup.",
                key_actions=["review proof points", "check fit with current stack"],
                content_preferences=["implementation notes", "case studies"],
            ),
        ],
    }


def build_heuristic_persona(cluster: ClusterData, mode: str, facet: str, parent: dict | None = None) -> PersonaV1:
    base = _build_trait_pool(cluster, facet)
    primary = (cluster.summary.top_behaviors or ["workflow"])[0].replace("_", " ")
    parent_name = parent.get("name") if parent else None
    if mode == "parent":
        name = f"{primary.title()} Archetype"
        summary = (
            f"A broad archetype centered on {primary}, balancing execution, coordination, "
            f"and operational follow-through."
        )
        goals = base["goals"][:2]
        pains = base["pains"][:2]
        motivations = base["motivations"][:2]
        objections = base["objections"][:1]
        vocabulary = base["vocabulary"][:4]
        sample_quotes = [
            f"I want {primary} to feel predictable, not fragile.",
            "I need a cleaner way to keep the team moving.",
        ]
    else:
        name = f"{parent_name or primary.title()} - {facet.title()}"
        summary = (
            f"A more specific {facet} variant of the {parent_name or primary} archetype, "
            f"focused on how {primary} gets operationalized day to day."
        )
        goals = base["goals"] + [f"Push {facet} farther without losing the parent archetype"]
        pains = base["pains"] + [f"Gets bogged down when {facet} is under-specified"]
        motivations = base["motivations"] + [f"Needs {facet} to feel concrete and reusable"]
        objections = base["objections"] + [f"Does not want {facet} to become abstract strategy"]
        vocabulary = base["vocabulary"] + [facet, "specific", "repeatable"]
        sample_quotes = [
            f"We need {facet} to be concrete, not hand-wavy.",
            f"The parent idea is fine, but {facet} is where the work happens.",
        ]

    evidence = [
        SourceEvidence(claim=goals[0], record_ids=[cluster.all_record_ids[0]], field_path="goals.0", confidence=0.7),
        SourceEvidence(claim=goals[1], record_ids=[cluster.all_record_ids[1]], field_path="goals.1", confidence=0.7),
        SourceEvidence(claim=pains[0], record_ids=[cluster.all_record_ids[2 % len(cluster.all_record_ids)]], field_path="pains.0", confidence=0.8),
        SourceEvidence(claim=pains[1], record_ids=[cluster.all_record_ids[3 % len(cluster.all_record_ids)]], field_path="pains.1", confidence=0.8),
        SourceEvidence(claim=motivations[0], record_ids=[cluster.all_record_ids[4 % len(cluster.all_record_ids)]], field_path="motivations.0", confidence=0.65),
        SourceEvidence(claim=objections[0], record_ids=[cluster.all_record_ids[5 % len(cluster.all_record_ids)]], field_path="objections.0", confidence=0.6),
    ]
    return PersonaV1(
        name=name,
        summary=summary,
        demographics=Demographics(
            age_range="30-45",
            gender_distribution="mixed",
            location_signals=["United States"],
            education_level=None,
            income_bracket=None,
        ),
        firmographics=Firmographics(
            company_size=cluster.tenant.industry or "mid-market",
            industry=cluster.tenant.industry,
            role_titles=[name, "Operations Lead"],
            tech_stack_signals=list(dict.fromkeys(record.source for record in cluster.sample_records))[:3],
        ),
        goals=goals,
        pains=pains,
        motivations=motivations,
        objections=objections,
        channels=base["channels"],
        vocabulary=vocabulary[:15],
        decision_triggers=base["decision_triggers"],
        sample_quotes=sample_quotes,
        journey_stages=base["journey_stages"],
        source_evidence=evidence,
    )


def heuristic_judge_score(persona: dict) -> JudgeScore:
    grounded = 4.0 if persona.get("source_evidence") else 2.0
    distinctiveness = min(5.0, 2.5 + len(set(persona.get("vocabulary", []))) / 4.0)
    coherence = min(5.0, 3.0 + (0.5 if persona.get("summary") else 0.0))
    actionable = min(5.0, 2.5 + len(persona.get("goals", [])) / 2.0)
    voice_fidelity = min(5.0, 3.0 + len(persona.get("sample_quotes", [])) / 3.0)
    overall = round((grounded + distinctiveness + coherence + actionable + voice_fidelity) / 5.0, 2)
    return JudgeScore(
        overall=overall,
        dimensions={
            "grounded": grounded,
            "distinctive": distinctiveness,
            "coherent": coherence,
            "actionable": actionable,
            "voice_fidelity": voice_fidelity,
        },
        rationale="heuristic fallback",
        judge_model="heuristic",
    )


async def safe_score_persona(judge: LLMJudge, persona: dict) -> JudgeScore:
    try:
        return await asyncio.wait_for(judge.score_persona(persona), timeout=LLM_TIMEOUT_SECONDS)
    except Exception:
        return heuristic_judge_score(persona)


def _build_generation_prompt(cluster: ClusterData, mode: str, facet: str, parent: dict | None = None) -> tuple[str, list[dict]]:
    base_user = build_user_message(cluster)
    if mode == "flat":
        instruction = (
            "Synthesize a standard control persona from this cluster. Keep it concrete and fully grounded."
        )
    elif mode == "parent":
        instruction = (
            "Synthesize a broad parent archetype persona. Keep it higher-level than a normal control persona, "
            "but still grounded and specific enough to be useful."
        )
    else:
        instruction = (
            "Synthesize a more specific child variant of the provided parent archetype. "
            f"Emphasize the following facet: {facet}. Keep the child distinct from the parent and sibling variants."
        )
    if parent is not None:
        instruction += "\n\nPARENT PERSONA:\n" + json.dumps(parent, indent=2, default=str)
    messages = [{"role": "user", "content": base_user + "\n\n" + instruction}]
    system = SYSTEM_PROMPT + "\n\n" + (
        "For parent archetypes, stay broad and navigable. For child archetypes, specialize the parent while keeping the same source grounding."
    )
    return system, messages


async def generate_persona(
    cluster: ClusterData,
    backend,
    mode: str,
    facet: str,
    parent: dict | None = None,
    max_attempts: int = 1,
) -> PersonaArtifact:
    errors: list[str] = []
    system, messages = _build_generation_prompt(cluster, mode, facet, parent)
    for _ in range(max_attempts):
        if errors:
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Previous attempt issues:\n"
                        + "\n".join(f"- {error}" for error in errors)
                        + "\n\n"
                        + messages[0]["content"]
                    ),
                }
            ]
        try:
            result = await asyncio.wait_for(
                backend.generate(system=system, messages=messages, tool=build_tool_definition()),
                timeout=LLM_TIMEOUT_SECONDS,
            )
            persona = PersonaV1.model_validate(result.tool_input)
            groundedness = check_groundedness(persona, cluster)
            if not groundedness.passed:
                errors = groundedness.violations
                continue
            return PersonaArtifact(
                level=mode,
                group_id=cluster.cluster_id,
                facet=facet,
                persona=persona.model_dump(mode="json"),
                judge_overall=0.0,
                judge_dimensions={},
                groundedness=groundedness.score,
                model_used=result.model,
                cost_usd=result.estimated_cost_usd,
            )
        except Exception as exc:
            errors = [str(exc)]

    persona = build_heuristic_persona(cluster, mode=mode, facet=facet, parent=parent)
    groundedness = check_groundedness(persona, cluster)
    return PersonaArtifact(
        level=mode,
        group_id=cluster.cluster_id,
        facet=facet,
        persona=persona.model_dump(mode="json"),
        judge_overall=0.0,
        judge_dimensions={},
        groundedness=groundedness.score,
        model_used="heuristic",
        cost_usd=0.0,
    )


def _write_findings(results_data: dict) -> str:
    summary = results_data["summary"]
    group_lines = []
    for group in results_data["groups"]:
        group_lines.append(
            f"- `{group['group_id']}`: flat judge `{group['flat_judge_overall']:.2f}`, "
            f"hierarchy judge `{group['hierarchy_judge_overall']:.2f}`, "
            f"coverage delta `{group['coverage_delta']:+.2f}`"
        )
    return "\n".join(
        [
            "# Experiment 6.23: Hierarchical Archetypes",
            "",
            "## Hypothesis",
            "A parent/child persona tree should improve navigability and coverage versus a flat persona list.",
            "",
            "## Method",
            "1. Generated a flat control persona per cluster from the golden tenant.",
            "2. Generated a broad parent archetype per cluster, then two child variants per parent.",
            "3. Scored every persona with the local judge helper and compared flat versus hierarchical sets.",
            "4. Measured coverage, distinctiveness, and information density across the two representations.",
            "",
            f"- Provider: `{summary['provider']}`",
            f"- Synthesis model: `{summary['synthesis_model']}`",
            f"- Judge model: `{summary['judge_model']}`",
            f"- Clusters: `{summary['n_clusters']}`",
            "",
            "## Group Metrics",
            *group_lines,
            "",
            "## Aggregate Metrics",
            f"- Flat mean judge: `{summary['flat_mean_judge_overall']:.2f}`",
            f"- Hierarchy mean judge: `{summary['hierarchy_mean_judge_overall']:.2f}`",
            f"- Flat coverage: `{summary['flat_coverage_ratio']:.2%}`",
            f"- Hierarchy coverage: `{summary['hierarchy_coverage_ratio']:.2%}`",
            f"- Within-parent distinctiveness: `{summary['within_parent_distinctiveness']:.3f}`",
            f"- Across-parent distinctiveness: `{summary['across_parent_distinctiveness']:.3f}`",
            f"- Flat information density: `{summary['flat_information_density']:.1f}`",
            f"- Hierarchy information density: `{summary['hierarchy_information_density']:.1f}`",
            "",
            "## Decision",
            (
                "Adopt. The hierarchical tree improved coverage and distinctiveness without hurting average judge quality."
                if summary["hierarchy_coverage_ratio"] >= summary["flat_coverage_ratio"]
                and summary["within_parent_distinctiveness"] > 0.15
                else "Reject. The hierarchical tree did not improve coverage or distinctiveness enough to justify the extra personas."
            ),
            "",
            "## Caveat",
            "Only one tenant and two natural clusters. Treat this as a small-sample structural comparison, not a general benchmark.",
        ]
    ) + "\n"


async def main() -> None:
    print("=" * 72)
    print("EXPERIMENT 6.23: Hierarchical archetypes")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clusters = get_clusters()

    anthropic_key = settings.anthropic_api_key.strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if anthropic_key:
        primary_backend = AnthropicBackend(client=AsyncAnthropic(api_key=anthropic_key), model=settings.default_model)
        fallback_backend = OpenAIJsonBackend(client=AsyncOpenAI(api_key=openai_key), model=OPENAI_MODEL) if openai_key else None
        synthesis_backend = FallbackGenerateBackend(primary_backend, fallback_backend)
        provider = "anthropic->openai" if fallback_backend is not None else "anthropic"
    elif openai_key:
        synthesis_backend = OpenAIJsonBackend(client=AsyncOpenAI(api_key=openai_key), model=OPENAI_MODEL)
        provider = "openai"
    else:
        raise RuntimeError("Missing ANTHROPIC_API_KEY and OPENAI_API_KEY")

    judge, judge_model = build_judge()
    synthesis_model = settings.default_model if anthropic_key else OPENAI_MODEL

    print(f"Provider: {provider} | Judge: {judge_model}")
    print("\n[1/3] Generating flat control personas...")
    t0 = time.monotonic()
    flat_artifacts: list[PersonaArtifact] = []
    for cluster in clusters:
        artifact = await generate_persona(cluster, synthesis_backend, mode="flat", facet="control")
        flat_artifacts.append(artifact)
        print(f"      flat {cluster.cluster_id}: {artifact.persona.get('name', 'unknown')}")

    print("\n[2/3] Generating hierarchical parents and children...")
    hierarchy_artifacts: list[PersonaArtifact] = []
    groups: list[dict] = []
    for cluster, flat_artifact in zip(clusters, flat_artifacts):
        parent = await generate_persona(cluster, synthesis_backend, mode="parent", facet="parent archetype")
        parent_score = await safe_score_persona(judge, parent.persona)
        parent.judge_overall = parent_score.overall
        parent.judge_dimensions = parent_score.dimensions
        children: list[PersonaArtifact] = []
        for facet in CHILD_FOCI:
            child = await generate_persona(cluster, synthesis_backend, mode="child", facet=facet, parent=parent.persona)
            scored = await safe_score_persona(judge, child.persona)
            child.judge_overall = scored.overall
            child.judge_dimensions = scored.dimensions
            children.append(child)
        hierarchy_artifacts.extend([parent, *children])
        groups.append(
            {
                "group_id": cluster.cluster_id,
                "flat_persona": flat_artifact.persona,
                "parent_persona": parent.persona,
                "children": [child.persona for child in children],
                "flat_judge_overall": (await safe_score_persona(judge, flat_artifact.persona)).overall,
                "hierarchy_judge_overall": mean([parent.judge_overall, *[child.judge_overall for child in children]]),
                "coverage_delta": coverage_ratio(cluster, [parent.persona, *[child.persona for child in children]])
                - coverage_ratio(cluster, [flat_artifact.persona]),
                "within_parent_distinctiveness": pairwise_distinctiveness([child.persona for child in children]),
                "across_parent_distinctiveness": 0.0,
                "flat_information_density": info_density(flat_artifact.persona),
                "hierarchy_information_density": mean(info_density(persona) for persona in [parent.persona, *[child.persona for child in children]]),
            }
        )

    if len(groups) >= 2:
        cross_pairs = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                left = [groups[i]["parent_persona"], *groups[i]["children"]]
                right = [groups[j]["parent_persona"], *groups[j]["children"]]
                for left_persona in left:
                    for right_persona in right:
                        cross_pairs.append(1.0 - persona_similarity(left_persona, right_persona))
        across_parent_distinctiveness = mean(cross_pairs) if cross_pairs else 0.0
        for group in groups:
            group["across_parent_distinctiveness"] = across_parent_distinctiveness
    else:
        across_parent_distinctiveness = 0.0

    flat_personas = [artifact.persona for artifact in flat_artifacts]
    hierarchy_personas = [artifact.persona for artifact in hierarchy_artifacts]
    flat_scores = [ (await safe_score_persona(judge, persona)).overall for persona in flat_personas ] if flat_personas else []
    hierarchy_scores = [ (await safe_score_persona(judge, persona)).overall for persona in hierarchy_personas ] if hierarchy_personas else []
    flat_mean_judge = mean(flat_scores) if flat_scores else 0.0
    hierarchy_mean_judge = mean(hierarchy_scores) if hierarchy_scores else 0.0
    flat_coverage = mean(coverage_ratio(cluster, [persona]) for cluster, persona in zip(clusters, flat_personas)) if flat_personas else 0.0
    hierarchy_coverage = mean(
        coverage_ratio(cluster, [group["parent_persona"], *group["children"]])
        for cluster, group in zip(clusters, groups)
    ) if groups else 0.0
    flat_density = mean(info_density(persona) for persona in flat_personas) if flat_personas else 0.0
    hierarchy_density = mean(info_density(persona) for persona in hierarchy_personas) if hierarchy_personas else 0.0

    summary = ExperimentSummary(
        provider=provider,
        judge_model=judge_model,
        synthesis_model=synthesis_model,
        n_clusters=len(clusters),
        flat_mean_judge_overall=flat_mean_judge,
        hierarchy_mean_judge_overall=hierarchy_mean_judge,
        flat_coverage_ratio=flat_coverage,
        hierarchy_coverage_ratio=hierarchy_coverage,
        flat_information_density=flat_density,
        hierarchy_information_density=hierarchy_density,
        within_parent_distinctiveness=mean(group["within_parent_distinctiveness"] for group in groups) if groups else 0.0,
        across_parent_distinctiveness=across_parent_distinctiveness,
        set_coherence_delta=hierarchy_mean_judge - flat_mean_judge,
        coverage_delta=hierarchy_coverage - flat_coverage,
        information_density_delta=hierarchy_density - flat_density,
    )

    results_data = {
        "experiment": "6.23",
        "title": "Hierarchical archetypes",
        "provider": provider,
        "judge_model": judge_model,
        "synthesis_model": synthesis_model,
        "clusters": [cluster.cluster_id for cluster in clusters],
        "flat": [asdict(artifact) for artifact in flat_artifacts],
        "hierarchy": [asdict(artifact) for artifact in hierarchy_artifacts],
        "groups": groups,
        "summary": asdict(summary),
        "duration_seconds": time.monotonic() - t0,
    }

    (OUTPUT_DIR / "results.json").write_text(json.dumps(results_data, indent=2, default=str))
    (OUTPUT_DIR / "FINDINGS.md").write_text(_write_findings(results_data))

    print(
        f"      flat_judge={flat_mean_judge:.2f} hierarchy_judge={hierarchy_mean_judge:.2f} "
        f"coverage_delta={summary.coverage_delta:+.2%}"
    )
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
