"""exp-6.06 — Cross-Tenant Leakage.

Measures whether personas synthesized for different tenants (with different
industries and products) are meaningfully different, or whether the pipeline
produces generic personas that leak across tenant boundaries.

Approach:
  1. Run the full pipeline (fetch -> segment -> synthesize) for two tenants
     with distinct industry/product contexts.
  2. Compute within-tenant and cross-tenant Jaccard similarity on persona text.
  3. Leakage ratio = cross / within. Target: <= 0.5

Hypothesis: Tenant-specific context (industry, product) drives meaningfully
different personas. Leakage ratio should be well below 0.5.

Usage:
    python scripts/run_exp_6_06.py
"""

from __future__ import annotations

import asyncio
import json
import string
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "orchestration"))

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

TENANT_A = {
    "tenant_id": "tenant_acme_saas",
    "industry": "B2B SaaS",
    "product": "Project management tool for engineering teams",
}
TENANT_B = {
    "tenant_id": "tenant_medflow_health",
    "industry": "Healthcare technology",
    "product": "Patient scheduling and clinical workflow platform for hospital systems",
}
OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-6.06"

N_CLUSTERS_TO_SYNTHESIZE = 3


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def persona_to_text(p: dict) -> str:
    """Concatenate key persona fields into text."""
    parts = [
        p.get("summary", ""),
        " ".join(p.get("goals", [])),
        " ".join(p.get("pains", [])),
        " ".join(p.get("motivations", [])),
        " ".join(p.get("vocabulary", [])),
        " ".join(p.get("sample_quotes", [])),
    ]
    return " ".join(parts)


def tokenize(text: str) -> set[str]:
    """Tokenize to word set: lowercase, strip punct, filter len > 3."""
    words = text.lower().split()
    cleaned = {w.strip(string.punctuation) for w in words}
    return {w for w in cleaned if len(w) > 3}


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def mean_pairwise_jaccard(personas: list[dict]) -> float:
    """Mean pairwise Jaccard within a set of personas."""
    token_sets = [tokenize(persona_to_text(p)) for p in personas]
    pairs = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            pairs.append(jaccard(token_sets[i], token_sets[j]))
    return sum(pairs) / len(pairs) if pairs else 0.0


def cross_tenant_jaccard(personas_a: list[dict], personas_b: list[dict]) -> float:
    """Mean pairwise Jaccard between all A-vs-B pairs."""
    tokens_a = [tokenize(persona_to_text(p)) for p in personas_a]
    tokens_b = [tokenize(persona_to_text(p)) for p in personas_b]
    pairs = []
    for ta in tokens_a:
        for tb in tokens_b:
            pairs.append(jaccard(ta, tb))
    return sum(pairs) / len(pairs) if pairs else 0.0


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

async def run_tenant_pipeline(
    tenant: dict,
    backend: AnthropicBackend,
    n_clusters: int,
) -> tuple[list[dict], list[ClusterData]]:
    """Fetch, segment, and synthesize for a single tenant."""
    tenant_id = tenant["tenant_id"]
    industry = tenant["industry"]
    product = tenant["product"]

    print(f"\n  Fetching records for {tenant_id}...")
    raw_records = fetch_all(tenant_id)
    records = [RawRecord.model_validate(r.model_dump()) for r in raw_records]

    print(f"  Segmenting ({len(records)} records, industry={industry})...")
    cluster_dicts = segment(
        records,
        tenant_industry=industry,
        tenant_product=product,
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    clusters = sorted(
        [ClusterData.model_validate(c) for c in cluster_dicts],
        key=lambda c: c.cluster_id,
    )
    print(f"  Got {len(clusters)} clusters: {[c.cluster_id for c in clusters]}")

    # Synthesize first n_clusters
    synth_clusters = clusters[:n_clusters]
    results = []
    for i, cluster in enumerate(synth_clusters):
        print(f"    [{i + 1}/{len(synth_clusters)}] {cluster.cluster_id}...")
        try:
            r = await synthesize(cluster, backend)
            p_dict = r.persona.model_dump(mode="json")
            results.append({
                "cluster_id": cluster.cluster_id,
                "tenant_id": tenant_id,
                "status": "ok",
                "persona": p_dict,
                "cost_usd": r.total_cost_usd,
                "groundedness": r.groundedness.score,
                "attempts": r.attempts,
            })
            print(
                f"      [OK] {p_dict['name']}  "
                f"cost=${r.total_cost_usd:.4f}  "
                f"grounded={r.groundedness.score:.2f}"
            )
        except SynthesisError as e:
            total_cost = sum(a.cost_usd for a in e.attempts)
            results.append({
                "cluster_id": cluster.cluster_id,
                "tenant_id": tenant_id,
                "status": "failed",
                "error": str(e),
                "cost_usd": total_cost,
            })
            print(f"      [FAIL] {e}  cost=${total_cost:.4f}")

    return results, clusters


async def main() -> None:
    print("=" * 72)
    print("exp-6.06 — Cross-Tenant Leakage")
    print("=" * 72)

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in synthesis/.env")

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    # ---- F0 check: verify we get enough clusters ----
    print("\n[F0] Feasibility check — verifying >= 4 clusters from Tenant A data...")
    raw_records = fetch_all(TENANT_A["tenant_id"])
    records = [RawRecord.model_validate(r.model_dump()) for r in raw_records]
    f0_clusters = segment(
        records,
        tenant_industry=TENANT_A["industry"],
        tenant_product=TENANT_A["product"],
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    n_f0 = len(f0_clusters)
    print(f"  F0 cluster count: {n_f0}")
    assert n_f0 >= 4, f"F0 FAIL: expected >= 4 clusters, got {n_f0}"
    print("  F0 PASS")

    # ---- Run Tenant A ----
    print("\n[1/4] Running pipeline for Tenant A (B2B SaaS)...")
    results_a, clusters_a = await run_tenant_pipeline(
        TENANT_A, backend, N_CLUSTERS_TO_SYNTHESIZE
    )

    # ---- Run Tenant B ----
    print("\n[2/4] Running pipeline for Tenant B (Healthcare tech)...")
    results_b, clusters_b = await run_tenant_pipeline(
        TENANT_B, backend, N_CLUSTERS_TO_SYNTHESIZE
    )

    # ---- Compute similarity metrics ----
    print("\n[3/4] Computing similarity metrics...")

    ok_a = [r for r in results_a if r["status"] == "ok"]
    ok_b = [r for r in results_b if r["status"] == "ok"]
    personas_a = [r["persona"] for r in ok_a]
    personas_b = [r["persona"] for r in ok_b]

    within_a = mean_pairwise_jaccard(personas_a) if len(personas_a) >= 2 else None
    within_b = mean_pairwise_jaccard(personas_b) if len(personas_b) >= 2 else None
    cross = cross_tenant_jaccard(personas_a, personas_b) if personas_a and personas_b else None

    # Within = mean of within_a and within_b
    within_vals = [v for v in [within_a, within_b] if v is not None]
    within_mean = sum(within_vals) / len(within_vals) if within_vals else None

    leakage_ratio = (cross / within_mean) if (cross is not None and within_mean and within_mean > 0) else None
    leakage_pass = leakage_ratio is not None and leakage_ratio <= 0.5

    print(f"  Within-A Jaccard: {within_a}")
    print(f"  Within-B Jaccard: {within_b}")
    print(f"  Within mean:      {within_mean}")
    print(f"  Cross-tenant:     {cross}")
    print(f"  Leakage ratio:    {leakage_ratio}")
    print(f"  Leakage pass:     {leakage_pass}")

    # ---- Write outputs ----
    print("\n[4/4] Writing results...")

    total_cost = sum(r.get("cost_usd", 0) for r in results_a + results_b)

    summary = {
        "experiment_id": "6.06",
        "branch": "exp-6.06-cross-tenant-leakage",
        "model": settings.default_model,
        "tenant_a": TENANT_A,
        "tenant_b": TENANT_B,
        "n_clusters_a": len(clusters_a),
        "n_clusters_b": len(clusters_b),
        "n_synthesized_a": len(ok_a),
        "n_synthesized_b": len(ok_b),
        "n_failed_a": len(results_a) - len(ok_a),
        "n_failed_b": len(results_b) - len(ok_b),
        "within_jaccard_a": within_a,
        "within_jaccard_b": within_b,
        "within_jaccard_mean": within_mean,
        "cross_tenant_jaccard": cross,
        "leakage_ratio": leakage_ratio,
        "leakage_target": "<=0.5",
        "leakage_pass": leakage_pass,
        "total_cost_usd": total_cost,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUTPUT_DIR / "personas_a.json").write_text(
        json.dumps(results_a, indent=2, default=str)
    )
    (OUTPUT_DIR / "personas_b.json").write_text(
        json.dumps(results_b, indent=2, default=str)
    )

    # ---- Print summary ----
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
