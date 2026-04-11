"""exp-2.07 — Order-of-fields effect.

A/B runs the persona synthesis pipeline with two schema variants that are
IDENTICAL except for field declaration order:

  - PersonaV1           (baseline: demographics first)
  - PersonaV1VoiceFirst (treatment: vocabulary & sample_quotes first)

Both are run against the same mock crawler output in the same session so
that segmentation/cluster stochasticity is held constant.

Metrics:
  - vocab_jaccard: Jaccard similarity between the two personas' vocabularies.
    Lower = more distinctive. Reported per schema.
  - stereotyping_rate: fraction of each persona's vocabulary that is a
    generic-business-English word (from a small curated stoplist).
  - groundedness, cost, retries — unchanged from the baseline pipeline.

Usage:
    python scripts/run_exp_2_07.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "orchestration"))
sys.path.insert(0, str(REPO_ROOT / "twin"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic  # noqa: E402

from crawler import fetch_all  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.synthesizer import SynthesisError, synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402
from synthesis.models.persona import PersonaV1, PersonaV1VoiceFirst  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-2.07"

# Generic-business-English stoplist for stereotyping detection. These are
# words that show up in nearly every SaaS persona regardless of cluster.
GENERIC_STOPLIST = {
    "collaboration", "collaborate", "efficient", "efficiency", "workflow",
    "workflows", "solution", "solutions", "productivity", "productive",
    "scalable", "scalability", "streamline", "streamlined", "optimize",
    "optimization", "strategic", "strategy", "innovative", "innovation",
    "seamless", "robust", "synergy", "leverage", "enable", "empower",
    "team", "teams", "teamwork", "stakeholder", "stakeholders", "user",
    "users", "customer", "customers", "solution-oriented", "best-in-class",
    "cutting-edge", "value-add", "holistic", "ecosystem", "bandwidth",
    "roadmap", "kpi", "alignment", "buy-in", "touchpoint", "onboarding",
}


def _normalize(word: str) -> str:
    return word.lower().strip().strip(".,;:'\"()-").replace("_", " ")


def vocab_jaccard(vocab_a: list[str], vocab_b: list[str]) -> float:
    a = {_normalize(w) for w in vocab_a}
    b = {_normalize(w) for w in vocab_b}
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def stereotyping_rate(vocab: list[str]) -> float:
    if not vocab:
        return 0.0
    norm = [_normalize(w) for w in vocab]
    hits = sum(1 for w in norm if w in GENERIC_STOPLIST)
    return hits / len(norm)


async def run_schema(
    clusters: list[ClusterData],
    schema_cls: type,
    backend: AnthropicBackend,
) -> list[dict]:
    """Synthesize one persona per cluster under the given schema.

    Captures SynthesisError per-cluster so the A/B run can complete even if
    one variant fails groundedness retries. A failure is itself a datum.
    """
    results = []
    for i, cluster in enumerate(clusters):
        print(f"  [{i + 1}/{len(clusters)}] {schema_cls.__name__}: {cluster.cluster_id}")
        try:
            result = await synthesize(cluster, backend, schema_cls=schema_cls)
            results.append({
                "cluster_id": cluster.cluster_id,
                "schema": schema_cls.__name__,
                "status": "ok",
                "persona": result.persona.model_dump(mode="json"),
                "cost_usd": result.total_cost_usd,
                "groundedness": result.groundedness.score,
                "attempts": result.attempts,
            })
            print(
                f"      [OK] {result.persona.name}  "
                f"cost=${result.total_cost_usd:.4f}  "
                f"grounded={result.groundedness.score:.2f}  "
                f"attempts={result.attempts}"
            )
        except SynthesisError as e:
            best_grounded = max(
                (a.groundedness_violations for a in e.attempts if a.groundedness_violations),
                default=[],
                key=len,
            )
            attempt_scores = []
            for a in e.attempts:
                attempt_scores.append({
                    "attempt": a.attempt,
                    "groundedness_violations": a.groundedness_violations,
                    "validation_errors": a.validation_errors,
                    "cost_usd": a.cost_usd,
                })
            total_cost = sum(a.cost_usd for a in e.attempts)
            results.append({
                "cluster_id": cluster.cluster_id,
                "schema": schema_cls.__name__,
                "status": "failed",
                "error": str(e),
                "attempts_detail": attempt_scores,
                "total_cost_usd": total_cost,
            })
            print(
                f"      [FAIL] {e}  "
                f"cost=${total_cost:.4f}  "
                f"attempts={len(e.attempts)}  "
                f"last_violations={len(best_grounded)}"
            )
    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute distinctiveness + stereotyping from a list of persona results.

    Gracefully handles failed-synthesis entries (status == 'failed').
    """
    ok_results = [r for r in results if r.get("status") == "ok"]
    failed_results = [r for r in results if r.get("status") == "failed"]

    vocabs = [r["persona"]["vocabulary"] for r in ok_results]
    jaccard = vocab_jaccard(vocabs[0], vocabs[1]) if len(vocabs) >= 2 else None
    stereo = [stereotyping_rate(v) for v in vocabs]
    grounded = [r["groundedness"] for r in ok_results]

    total_cost = sum(r.get("cost_usd", 0.0) for r in ok_results)
    total_cost += sum(r.get("total_cost_usd", 0.0) for r in failed_results)

    return {
        "n_personas_attempted": len(results),
        "n_personas_success": len(ok_results),
        "n_personas_failed": len(failed_results),
        "success_rate": len(ok_results) / len(results) if results else 0.0,
        "vocab_jaccard_pairwise": jaccard,
        "stereotyping_rate_per_persona": stereo,
        "stereotyping_rate_mean": sum(stereo) / len(stereo) if stereo else None,
        "groundedness_mean": sum(grounded) / len(grounded) if grounded else None,
        "total_cost_usd": total_cost,
    }


async def main() -> None:
    print("=" * 72)
    print("exp-2.07 — Order-of-fields effect")
    print("=" * 72)

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in synthesis/.env")

    # ---- Ingest + segment (shared across both conditions) ----
    print("\n[1/3] Fetching and clustering mock records...")
    raw_records = fetch_all(TENANT_ID)
    records = [RawRecord.model_validate(r.model_dump()) for r in raw_records]
    cluster_dicts = segment(
        records,
        tenant_industry=TENANT_INDUSTRY,
        tenant_product=TENANT_PRODUCT,
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    clusters = sorted(
        [ClusterData.model_validate(c) for c in cluster_dicts],
        key=lambda c: c.cluster_id,
    )
    print(f"  Got {len(clusters)} clusters: {[c.cluster_id for c in clusters]}")

    # ---- Synthesize under both schemas ----
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    print("\n[2/3] Baseline schema (PersonaV1)...")
    baseline_results = await run_schema(clusters, PersonaV1, backend)

    print("\n[3/3] Treatment schema (PersonaV1VoiceFirst)...")
    treatment_results = await run_schema(clusters, PersonaV1VoiceFirst, backend)

    # ---- Compute metrics ----
    baseline_metrics = compute_metrics(baseline_results)
    treatment_metrics = compute_metrics(treatment_results)

    summary = {
        "experiment_id": "2.07",
        "branch": "exp-2.07-order-of-fields",
        "model": settings.default_model,
        "n_clusters": len(clusters),
        "baseline": {"schema": "PersonaV1", **baseline_metrics},
        "treatment": {"schema": "PersonaV1VoiceFirst", **treatment_metrics},
    }

    # Deltas: treatment - baseline. Lower jaccard/stereotyping = BETTER.
    def _delta(key):
        b = baseline_metrics.get(key)
        t = treatment_metrics.get(key)
        if b is None or t is None:
            return None
        return t - b

    summary["delta_vocab_jaccard"] = _delta("vocab_jaccard_pairwise")
    summary["delta_stereotyping_rate_mean"] = _delta("stereotyping_rate_mean")
    summary["delta_groundedness_mean"] = _delta("groundedness_mean")
    summary["delta_success_rate"] = _delta("success_rate")

    # ---- Persist ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "baseline_personas.json").write_text(
        json.dumps(baseline_results, indent=2, default=str)
    )
    (OUTPUT_DIR / "treatment_personas.json").write_text(
        json.dumps(treatment_results, indent=2, default=str)
    )
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- Print summary ----
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
