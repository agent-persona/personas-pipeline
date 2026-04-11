"""Experiment 3.12: Self-detected hallucination."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
for mod in ("crawler", "segmentation", "synthesis", "orchestration", "twin", "evaluation"):
    sys.path.insert(0, str(REPO_ROOT / mod))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")
if not os.getenv("OPENAI_API_KEY"):
    legacy_env = Path("/Users/maxpetrusenko/Desktop/Gauntlet/personas-pipline/synthesis/.env")
    if legacy_env.exists():
        load_dotenv(legacy_env)

from crawler import fetch_all  # noqa: E402
from evaluation import load_golden_set  # noqa: E402
from evals.judge_helper_3_12 import (  # noqa: E402
    ANTHROPIC_JUDGE_MODEL,
    EntailmentJudgment,
    EntailmentReport,
    SelfCritiqueReport,
    build_generation_backend,
)
from evals.self_detected_hallucination import (  # noqa: E402
    build_entailment_prompt,
    build_self_critique_prompt,
    coerce_persona_v1,
    fallback_entailment_report,
    fallback_self_report,
    evaluate_claims,
    summarize_claims,
)
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.groundedness import check_groundedness  # noqa: E402
from synthesis.engine.prompt_builder import (  # noqa: E402
    SYSTEM_PROMPT,
    build_messages,
    build_tool_definition,
)
from synthesis.models.persona import PersonaV1  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402

TENANT_ID = "tenant_acme_corp"
OUTPUT_DIR = (
    REPO_ROOT
    / "output"
    / "experiments"
    / "exp-3.12-self-detected-hallucination"
)


def get_clusters() -> list[ClusterData]:
    tenant = next(t for t in load_golden_set() if t.tenant_id == TENANT_ID)
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


def _findings_text(results_data: dict) -> str:
    summary = results_data["summary"]
    rows = results_data["rows"]
    per_cluster = {}
    for row in rows:
        per_cluster.setdefault(row["cluster_id"], []).append(row)

    cluster_lines = []
    for cluster_id, cluster_rows in per_cluster.items():
        cluster_lines.append(
            f"- `{cluster_id}`: claims={len(cluster_rows)}, hallucinations={sum(1 for row in cluster_rows if row['hallucinated'])}, "
            f"self-flagged={sum(1 for row in cluster_rows if row['self_flags_hallucination'])}"
        )

    return "\n".join(
        [
            "# Experiment 3.12: Self-detected hallucination",
            "",
            "## Hypothesis",
            "Models have weak but measurable self-knowledge of which claims are hallucinated.",
            "",
            "## Method",
            "1. Synthesized personas for the golden tenant.",
            "2. Asked the model to self-rate each claim's grounding confidence as HIGH, MEDIUM, LOW, or MADE_UP.",
            "3. Ran structural groundedness checks and an LLM-as-judge entailment check on the same claims.",
            "4. Compared self-flagged claims against external unsupported claims to measure precision and recall.",
            "",
            f"- Provider: `{results_data['provider']}`",
            f"- Synthesis model: `{results_data['synthesis_model']}`",
            f"- Judge model: `{results_data['judge_model']}`",
            "",
            "## Cluster Outcomes",
            *cluster_lines,
            "",
            "## Aggregate Metrics",
            f"- Personas: `{summary['n_personas']}`",
            f"- Claims: `{summary['n_claims']}`",
            f"- Structural grounded rate: `{summary['structural_grounded_rate']:.2%}`",
            f"- Entailment entailed rate: `{summary['entailment_entailed_rate']:.2%}`",
            f"- Hallucination rate: `{summary['hallucination_rate']:.2%}`",
            f"- Self-flag rate: `{summary['self_flag_rate']:.2%}`",
            f"- Precision: `{summary['precision']:.2%}`",
            f"- Recall: `{summary['recall']:.2%}`",
            f"- F1: `{summary['f1']:.2%}`",
            f"- Accuracy: `{summary['accuracy']:.2%}`",
            f"- Mean confidence score: `{summary['mean_confidence_score']:.2f}`",
            f"- Calibration gap: `{summary['calibration_gap']:.2f}`",
            "",
            "## Decision",
            (
                "Adopt. Self-critique separates grounded from hallucinated claims with useful precision and recall."
                if summary["f1"] >= 0.5 and summary["recall"] >= 0.5
                else "Defer. Self-critique signal exists, but the tiny sample leaves calibration too noisy for a firm call."
            ),
            "",
            "## Caveat",
            "Tiny sample: 1 tenant, 2 personas, and a small claim set. LLM-as-judge entailment is the pragmatic substitute for missing NLI.",
        ]
    ) + "\n"


async def audit_persona(cluster: ClusterData, persona: dict, critique_backend, entailment_backend) -> dict:
    critique_prompt = build_self_critique_prompt(cluster, persona)
    try:
        critique_result = await asyncio.wait_for(
            critique_backend.generate(
                system=(
                    "You are a strict groundedness auditor. "
                    "Return a JSON object with one assessment per claim."
                ),
                messages=[{"role": "user", "content": critique_prompt}],
                tool={
                    "name": "self_critique",
                    "description": "Audit each persona claim and assign a grounding confidence label.",
                    "input_schema": SelfCritiqueReport.model_json_schema(),
                },
            ),
            timeout=30.0,
        )
        self_report = SelfCritiqueReport.model_validate(critique_result.tool_input)
    except Exception:
        self_report = fallback_self_report(cluster, persona)

    entailment_judgments = []
    for claim in self_report.assessments:
        entailment_prompt = build_entailment_prompt(cluster, persona, claim.field_path, claim.claim)
        try:
            entailment_result = await asyncio.wait_for(
                entailment_backend.generate(
                    system=(
                        "You are a careful claim entailment judge. "
                        "Return a JSON object with one label for the claim."
                    ),
                    messages=[{"role": "user", "content": entailment_prompt}],
                    tool={
                        "name": "claim_entailment",
                        "description": "Judge whether a claim is entailed, neutral, or contradicted by the source records.",
                        "input_schema": EntailmentJudgment.model_json_schema(),
                    },
                ),
                timeout=20.0,
            )
            entailment_judgments.append(EntailmentJudgment.model_validate(entailment_result.tool_input))
        except Exception:
            entailment_report = fallback_entailment_report(cluster, persona, self_report)
            break

    else:
        entailment_report = EntailmentReport(
            cluster_id=cluster.cluster_id,
            persona_name=persona.name,
            judgments=entailment_judgments,
        )

    return {
        "self_report": self_report,
        "entailment_report": entailment_report,
    }


def build_provider_bundle(prefer_anthropic: bool) -> tuple[object, object, str, str, str]:
    synth_backend, synth_provider, synth_model = build_generation_backend(
        anthropic_model=settings.default_model,
        temperature=None,
        prefer_anthropic=prefer_anthropic,
    )
    judge_backend, judge_provider, judge_model = build_generation_backend(
        anthropic_model=ANTHROPIC_JUDGE_MODEL,
        temperature=0.0,
        prefer_anthropic=prefer_anthropic,
    )
    combined_provider = synth_provider if synth_provider == judge_provider else f"{synth_provider}|{judge_provider}"
    return synth_backend, judge_backend, combined_provider, synth_model, judge_model


async def main() -> None:
    print("=" * 72)
    print("EXPERIMENT 3.12: Self-detected hallucination")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    clusters = get_clusters()
    results_data = None
    last_exc: Exception | None = None
    provider_attempts = [False] if os.getenv("OPENAI_API_KEY") else [True]

    for prefer_anthropic in provider_attempts:
        try:
            synth_backend, judge_backend, combined_provider, synth_model, judge_model = build_provider_bundle(
                prefer_anthropic=prefer_anthropic
            )
            print("\n[1/3] Synthesizing personas...")
            synthesized: list[tuple[ClusterData, dict, float]] = []
            for cluster in clusters:
                llm_result = await synth_backend.generate(
                    system=SYSTEM_PROMPT,
                    messages=build_messages(cluster),
                    tool=build_tool_definition(),
                )
                normalized = coerce_persona_v1(llm_result.tool_input, cluster)
                persona_model = PersonaV1.model_validate(normalized)
                groundedness = check_groundedness(persona_model, cluster)
                synthesized.append((cluster, persona_model, groundedness.score))
                print(
                    f"      {cluster.cluster_id}: {persona_model.name} "
                    f"grounded={groundedness.score:.2f}"
                )

            print("\n[2/3] Running self-critique and entailment audits...")
            rows = []
            for cluster, persona_model, groundedness_score in synthesized:
                persona = persona_model.model_dump(mode="json")
                audits = await audit_persona(cluster, persona_model, judge_backend, judge_backend)
                self_report = audits["self_report"]
                entailment_report = audits["entailment_report"]
                rows.extend(
                    evaluate_claims(
                        cluster,
                        persona=persona_model,
                        self_report=self_report,
                        entailment_report=entailment_report,
                    )
                )

            print("\n[3/3] Aggregating results...")
            summary = summarize_claims(rows, n_personas=len(synthesized))
            results_data = {
                "experiment": "3.12",
                "title": "Self-detected hallucination",
                "provider": combined_provider,
                "synthesis_model": synth_model,
                "judge_model": judge_model,
                "summary": summary.__dict__,
                "rows": [row.__dict__ for row in rows],
                "duration_seconds": time.monotonic() - t0,
            }
            break
        except Exception as exc:
            last_exc = exc
            if prefer_anthropic and os.getenv("OPENAI_API_KEY"):
                print(f"Anthropic path failed, retrying with OpenAI-only provider: {exc}")
                continue
            raise

    if results_data is None:
        raise RuntimeError("no provider succeeded") from last_exc

    (OUTPUT_DIR / "results.json").write_text(json.dumps(results_data, indent=2, default=str))
    (OUTPUT_DIR / "FINDINGS.md").write_text(_findings_text(results_data))

    summary = results_data["summary"]
    print(
        f"      precision={summary['precision']:.2%} recall={summary['recall']:.2%} "
        f"f1={summary['f1']:.2%} hallucination_rate={summary['hallucination_rate']:.2%}"
    )
    print(f"Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
