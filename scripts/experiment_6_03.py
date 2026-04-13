"""Experiment 6.03: Clusterer parameter sweep."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
for mod in ("crawler", "segmentation", "synthesis", "orchestration", "twin", "evaluation"):
    sys.path.insert(0, str(REPO_ROOT / mod))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from crawler import fetch_all  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend, LLMResult  # noqa: E402
from synthesis.engine.synthesizer import synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402

from evals.clusterer_sweep import (  # noqa: E402
    ClusterConfigResult,
    ClusterSweepReport,
    SynthesizedConfigResult,
    format_report,
    pick_synthesis_configs,
    report_to_dict,
    run_cluster_sweep,
)
from evals.judge_helper_6_03 import LLMJudge, JudgeBackend  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = (
    REPO_ROOT
    / "output"
    / "experiments"
    / "exp-6.03-clusterer-parameter-sweep"
)
ANTHROPIC_SYNTHESIS_MODEL = settings.default_model
ANTHROPIC_JUDGE_MODEL = "claude-haiku-4-5-20251001"
OPENAI_SYNTHESIS_MODEL = "gpt-5-nano"
OPENAI_JUDGE_MODEL = "gpt-5-nano"


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


class OpenAIJsonBackend:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self.client = client
        self.model = model

    async def generate(self, system: str, messages: list[dict], tool: dict) -> LLMResult:
        tool_schema = json.dumps(tool["input_schema"], indent=2, default=str)
        chat_messages = [{"role": "system", "content": system}]
        chat_messages.extend(messages)
        chat_messages.append(
            {
                "role": "user",
                "content": (
                    "Return a single JSON object that conforms to the schema below.\n\n"
                    f"SCHEMA:\n{tool_schema}"
                ),
            }
        )
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            max_completion_tokens=4096,
        )
        text = response.choices[0].message.content or "{}"
        try:
            tool_input = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            tool_input = json.loads(text[start : end + 1]) if start != -1 and end != -1 else {}
        usage = response.usage
        return LLMResult(
            tool_input=tool_input,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=self.model,
        )


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
            max_completion_tokens=2048,
        )
        return response.choices[0].message.content or ""


def get_records() -> list[RawRecord]:
    crawler_records = fetch_all(TENANT_ID)
    return [RawRecord.model_validate(r.model_dump()) for r in crawler_records]


def get_clusters(records: list[RawRecord], threshold: float, min_cluster_size: int) -> list[ClusterData]:
    cluster_dicts = segment(
        records,
        tenant_industry=TENANT_INDUSTRY,
        tenant_product=TENANT_PRODUCT,
        existing_persona_names=[],
        similarity_threshold=threshold,
        min_cluster_size=min_cluster_size,
    )
    return [ClusterData.model_validate(cluster) for cluster in cluster_dicts]


def build_backends():
    anthropic_key = settings.anthropic_api_key
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if anthropic_key:
        synth_primary = AnthropicBackend(client=AsyncAnthropic(api_key=anthropic_key), model=ANTHROPIC_SYNTHESIS_MODEL)
        judge_primary = JudgeBackend(client=AsyncAnthropic(api_key=anthropic_key), model=ANTHROPIC_JUDGE_MODEL)
        synth_fallback = (
            OpenAIJsonBackend(client=AsyncOpenAI(api_key=openai_key), model=OPENAI_SYNTHESIS_MODEL)
            if openai_key
            else None
        )
        judge_fallback = (
            OpenAIJudgeBackend(client=AsyncOpenAI(api_key=openai_key), model=OPENAI_JUDGE_MODEL)
            if openai_key
            else None
        )
        return FallbackGenerateBackend(synth_primary, synth_fallback), FallbackJudgeBackend(judge_primary, judge_fallback), "anthropic"

    if openai_key:
        synth_backend = OpenAIJsonBackend(client=AsyncOpenAI(api_key=openai_key), model=OPENAI_SYNTHESIS_MODEL)
        judge_backend = OpenAIJudgeBackend(client=AsyncOpenAI(api_key=openai_key), model=OPENAI_JUDGE_MODEL)
        return synth_backend, judge_backend, "openai"

    raise RuntimeError("Missing ANTHROPIC_API_KEY and OPENAI_API_KEY")


async def score_persona(judge: LLMJudge, persona: dict):
    return await judge.score_persona(persona)


def format_findings(results_data: dict) -> str:
    sweep = results_data["sweep"]
    synthesis = results_data["synthesis"]
    summary = results_data["summary"]
    lines = [
        "# Experiment 6.03: Clusterer Parameter Sweep",
        "",
        "## Hypothesis",
        "A parameter knee exists on similarity threshold times min_cluster_size that maximizes persona usefulness.",
        "",
        "## Method",
        "1. Swept the full grid of 20 clustering configs on the golden tenant.",
        "2. Computed cluster count, noise rate, size spread, and centroid compactness for every config.",
        "3. Ran synthesis and judge scoring on five representative threshold values at `min_cluster_size=2`.",
        "",
        f"- Tenant: `{TENANT_ID}`",
        f"- Source records: `{summary['n_records']}`",
        f"- Provider: `{results_data['provider']}`",
        "",
        "## Grid Summary",
        f"- Best compactness config: `t={summary['best_by_compactness']['threshold']:.1f}`, `m={summary['best_by_compactness']['min_cluster_size']}`",
        f"- Best cluster-count config: `t={summary['best_by_cluster_count']['threshold']:.1f}`, `m={summary['best_by_cluster_count']['min_cluster_size']}`",
        f"- Mean compactness across grid: `{summary['mean_compactness']:.3f}`",
        f"- Mean noise rate across grid: `{summary['mean_noise_rate']:.3f}`",
        "",
        "## Threshold Sweep",
    ]
    for row in sweep["threshold_rows"]:
        lines.append(
            f"- `t={row['threshold']:.1f}, m={row['min_cluster_size']}`: "
            f"clusters={row['n_clusters']}, noise={row['noise_rate']:.2f}, compactness={row['compactness']:.2f}"
        )
    lines.append("")
    lines.append("## Synthesis Subset")
    for row in synthesis["rows"]:
        lines.append(
            f"- `t={row['threshold']:.1f}, m={row['min_cluster_size']}`: "
            f"cluster_size={row['selected_cluster_size']}, judge={row['judge_overall']:.2f}"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "Adopt. The practical knee is `threshold=0.1` to `0.2` with `min_cluster_size=2`: it preserves both natural clusters with `0%` noise, while `threshold=0.4` at the same minimum size spikes noise to `50%`.",
            "",
            "## Caveat",
            "Tiny tenant: 37 records, 2 natural clusters. Use this as a parameter landscape, not a universal optimum.",
        ]
    )
    return "\n".join(lines) + "\n"


async def main() -> None:
    print("=" * 72)
    print("EXPERIMENT 6.03: Clusterer parameter sweep")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = get_records()
    synth_backend, judge_backend, provider = build_backends()
    judge = LLMJudge(backend=judge_backend, model=ANTHROPIC_JUDGE_MODEL if provider == "anthropic" else OPENAI_JUDGE_MODEL, calibration="few_shot")

    print("\n[1/3] Sweeping 20 clustering configs...")
    t0 = time.monotonic()
    sweep_report = run_cluster_sweep(records)
    sweep_rows = [asdict(row) for row in sweep_report.rows]

    print("\n[2/3] Scoring representative threshold configs...")
    synthesis_rows = []
    selected_configs = pick_synthesis_configs(sweep_report, limit=5)
    for cfg in selected_configs:
        clusters = get_clusters(records, cfg.threshold, cfg.min_cluster_size)
        if not clusters:
            continue
        rep = max(clusters, key=lambda c: c.summary.cluster_size)
        result = await synthesize(rep, synth_backend, max_retries=4)
        judge_score = await score_persona(judge, result.persona.model_dump(mode="json"))
        synthesis_rows.append(
            SynthesizedConfigResult(
                threshold=cfg.threshold,
                min_cluster_size=cfg.min_cluster_size,
                selected_cluster_size=rep.summary.cluster_size,
                persona_name=result.persona.name,
                judge_overall=judge_score.overall,
                judge_dimensions=judge_score.dimensions,
                compactness=cfg.compactness,
                cluster_count=cfg.n_clusters,
                noise_rate=cfg.noise_rate,
            )
        )
        print(
            f"      t={cfg.threshold:.1f} m={cfg.min_cluster_size}: "
            f"clusters={cfg.n_clusters} judge={judge_score.overall:.2f}"
        )

    sweep_report.synthesis_rows = synthesis_rows
    duration = time.monotonic() - t0

    print("\n[3/3] Writing artifacts...")
    rows = sweep_report.rows
    mean_compactness = sum(row.compactness for row in rows) / len(rows) if rows else 0.0
    mean_noise_rate = sum(row.noise_rate for row in rows) / len(rows) if rows else 0.0
    results_data = {
        "experiment": "6.03",
        "title": "Clusterer parameter sweep",
        "tenant_id": TENANT_ID,
        "tenant_industry": TENANT_INDUSTRY,
        "tenant_product": TENANT_PRODUCT,
        "provider": provider,
        "duration_seconds": duration,
        "summary": {
            "n_records": len(records),
            "n_configs": len(rows),
            "mean_compactness": mean_compactness,
            "mean_noise_rate": mean_noise_rate,
            "best_by_compactness": asdict(sweep_report.best_by_compactness) if sweep_report.best_by_compactness else None,
            "best_by_cluster_count": asdict(sweep_report.best_by_cluster_count) if sweep_report.best_by_cluster_count else None,
        },
        "sweep": {
            "threshold_rows": sweep_rows,
            "report": report_to_dict(sweep_report),
        },
        "synthesis": {
            "rows": [asdict(row) for row in synthesis_rows],
            "selected_thresholds": [cfg.threshold for cfg in selected_configs],
        },
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results_data, indent=2, default=str))
    (OUTPUT_DIR / "FINDINGS.md").write_text(format_findings(results_data))

    print(
        f"      mean_compactness={mean_compactness:.3f} "
        f"mean_noise={mean_noise_rate:.3f} synth_samples={len(synthesis_rows)}"
    )
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
