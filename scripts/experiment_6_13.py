"""Experiment 6.13: Persona overlap heatmap.

Hypothesis: Personas synthesized from the same tenant may overlap excessively
on generic fields (channels, demographics) while remaining distinct on
identity-defining fields (goals, vocabulary, sample_quotes). The diagonal
density metric quantifies population-level distinctiveness.

Setup:
  1. Synthesize multiple personas from the golden tenant's clusters.
  2. Compute field-by-field similarity between all persona pairs.
  3. Build the NxN similarity matrix and render as heatmap.
  4. Compute diagonal density (mean off-diagonal similarity).
  5. Identify which fields drive the most overlap.

Metrics:
  - Diagonal density (mean off-diagonal similarity; lower = more distinct)
  - Per-field density breakdown
  - Max-overlap pair

Usage:
    python scripts/experiment_6_13.py
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "orchestration"))
sys.path.insert(0, str(REPO_ROOT / "twin"))
sys.path.insert(0, str(REPO_ROOT / "evals"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic  # noqa: E402

from crawler import fetch_all  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.synthesizer import synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402

from overlap_heatmap import (  # noqa: E402
    ALL_COMPARED_FIELDS,
    SimilarityMatrix,
    compute_similarity_matrix,
    diagonal_density,
    max_overlap_pair,
    per_field_density,
    render_field_heatmap,
    render_heatmap,
)

# ── Constants ─────────────────────────────────────────────────────────

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"


# ── Pipeline ──────────────────────────────────────────────────────────

def get_clusters(
    similarity_threshold: float = 0.15,
    min_cluster_size: int = 2,
) -> list[ClusterData]:
    crawler_records = fetch_all(TENANT_ID)
    raw_records = [RawRecord.model_validate(r.model_dump()) for r in crawler_records]
    cluster_dicts = segment(
        raw_records,
        tenant_industry=TENANT_INDUSTRY,
        tenant_product=TENANT_PRODUCT,
        existing_persona_names=[],
        similarity_threshold=similarity_threshold,
        min_cluster_size=min_cluster_size,
    )
    return [ClusterData.model_validate(c) for c in cluster_dicts]


async def synthesize_personas(
    clusters: list[ClusterData],
    backend: AnthropicBackend,
) -> list[dict]:
    """Synthesize one persona per cluster."""
    personas = []
    for cluster in clusters:
        try:
            result = await synthesize(cluster, backend)
            persona_dict = result.persona.model_dump(mode="json")
            persona_dict["_meta"] = {
                "cluster_id": cluster.cluster_id,
                "groundedness": result.groundedness.score,
                "cost_usd": result.total_cost_usd,
                "attempts": result.attempts,
            }
            personas.append(persona_dict)
            print(f"    {result.persona.name} "
                  f"(groundedness={result.groundedness.score:.2f}, "
                  f"cost=${result.total_cost_usd:.4f})")
        except Exception as e:
            print(f"    FAILED cluster {cluster.cluster_id}: {e}")
    return personas


# ── Reporting ─────────────────────────────────────────────────────────

def print_results(
    matrix: SimilarityMatrix,
    density: float,
    field_densities: dict[str, float],
    max_pair: tuple[str, str, float],
    synth_cost: float,
) -> str:
    lines: list[str] = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("\n" + "=" * 100)
    p("EXPERIMENT 6.13 — PERSONA OVERLAP HEATMAP — RESULTS")
    p("=" * 100)

    # ── Similarity matrix ──
    p(f"\n── SIMILARITY MATRIX ({matrix.n} personas) ──")

    # Short names
    shorts = []
    for name in matrix.names:
        words = name.split(",")[0].split()
        short = words[0] if words else name
        if len(short) > 12:
            short = short[:11] + "."
        shorts.append(short)

    header = f"  {'':>14}"
    for s in shorts:
        header += f"{s:>14}"
    p(header)
    p("  " + "-" * (14 + 14 * matrix.n))

    for i in range(matrix.n):
        row = f"  {shorts[i]:>14}"
        for j in range(matrix.n):
            val = matrix.matrix[i][j]
            marker = " *" if i == j else ""
            row += f"{val:>12.3f}{marker}"
        p(row)
    p("  (* = self-similarity, always 1.000)")

    # ── Diagonal density ──
    p(f"\n── DIAGONAL DENSITY ──")
    p(f"  Mean off-diagonal similarity: {density:.4f}")
    if density < 0.20:
        quality = "EXCELLENT — personas are highly distinct"
    elif density < 0.35:
        quality = "GOOD — moderate distinctiveness"
    elif density < 0.50:
        quality = "FAIR — some concerning overlap"
    else:
        quality = "POOR — personas are too similar"
    p(f"  Assessment: {quality}")

    # ── Max overlap pair ──
    p(f"\n── MOST SIMILAR PAIR ──")
    p(f"  {max_pair[0]}  <-->  {max_pair[1]}")
    p(f"  Similarity: {max_pair[2]:.4f}")

    # ── Per-field density ──
    p(f"\n── PER-FIELD OVERLAP DENSITY (lower = more distinct) ──")
    sorted_fields = sorted(field_densities.items(), key=lambda x: x[1], reverse=True)
    for f, d in sorted_fields:
        bar_len = int(d * 40)
        bar = "#" * bar_len + "." * (40 - bar_len)
        p(f"  {f:<22} {d:.3f}  [{bar}]")

    # ── High/low overlap fields ──
    high_overlap = [(f, d) for f, d in sorted_fields if d > 0.35]
    low_overlap = [(f, d) for f, d in sorted_fields if d < 0.15]

    p(f"\n  High-overlap fields (>0.35): "
      f"{', '.join(f'{f}({d:.2f})' for f, d in high_overlap) or 'none'}")
    p(f"  Low-overlap fields (<0.15):  "
      f"{', '.join(f'{f}({d:.2f})' for f, d in low_overlap) or 'none'}")

    # ── Signal assessment ──
    p("\n── SIGNAL ASSESSMENT ──")
    if high_overlap and low_overlap:
        strength = "STRONG FINDING"
        detail = (
            f"Clear differentiation pattern: {len(high_overlap)} fields show high "
            f"overlap while {len(low_overlap)} show low overlap. Persona synthesis "
            f"is producing distinct identities but sharing structural similarities "
            f"in {', '.join(f for f, _ in high_overlap)}."
        )
    elif density < 0.25:
        strength = "MODERATE FINDING"
        detail = (
            f"Diagonal density {density:.3f} indicates good persona distinctiveness. "
            f"The pipeline produces meaningfully different personas from different clusters."
        )
    elif density > 0.45:
        strength = "STRONG FINDING (negative)"
        detail = (
            f"Diagonal density {density:.3f} is high — personas overlap too much. "
            f"The synthesis prompt or clustering may need adjustment."
        )
    else:
        strength = "WEAK FINDING"
        detail = (
            f"Diagonal density {density:.3f} is moderate. Personas are somewhat "
            f"distinct but could benefit from stronger differentiation signals."
        )

    p(f"\n  Signal: {strength}")
    p(f"  {detail}")
    p(f"\n  Synthesis cost: ${synth_cost:.4f}")

    p("\n" + "=" * 100)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("EXPERIMENT 6.13: Persona overlap heatmap")
    print("Hypothesis: Personas overlap on structural fields but diverge")
    print("  on identity-defining fields (goals, vocabulary, quotes)")
    print("=" * 72)

    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    # Step 1: Ingest + segment
    print("\n[1/4] Running ingest + segmentation...")
    clusters = get_clusters()
    print(f"      Got {len(clusters)} clusters")

    # Step 2: Synthesize personas
    print("\n[2/4] Synthesizing personas...")
    personas = await synthesize_personas(clusters, backend)
    if len(personas) < 2:
        print("ERROR: Need at least 2 personas for overlap analysis")
        sys.exit(1)
    print(f"      Generated {len(personas)} personas")

    # Step 3: Compute similarity
    print("\n[3/4] Computing similarity matrix...")
    stripped = [{k: v for k, v in p.items() if not k.startswith("_")} for p in personas]
    matrix = compute_similarity_matrix(stripped)
    density = diagonal_density(matrix)
    field_densities = per_field_density(matrix)
    max_pair = max_overlap_pair(matrix)

    # Step 4: Render heatmaps and report
    print("\n[4/4] Rendering heatmaps and generating report...")
    output_dir = REPO_ROOT / "output" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path = render_heatmap(
        matrix, output_dir / "exp_6_13_heatmap.png",
        title=f"Persona Overlap (density={density:.3f})",
    )
    print(f"      Heatmap: {heatmap_path}")

    field_heatmap_path = render_field_heatmap(
        matrix, output_dir / "exp_6_13_field_density.png",
    )
    print(f"      Field density chart: {field_heatmap_path}")

    synth_cost = sum(p.get("_meta", {}).get("cost_usd", 0) for p in personas)
    report = print_results(matrix, density, field_densities, max_pair, synth_cost)

    # Save JSON results
    results_data = {
        "experiment": "6.13",
        "title": "Persona overlap heatmap",
        "hypothesis": (
            "Personas overlap on structural fields but diverge on "
            "identity-defining fields"
        ),
        "synthesis_model": settings.default_model,
        "n_personas": len(personas),
        "persona_names": matrix.names,
        "diagonal_density": density,
        "similarity_matrix": matrix.matrix,
        "per_field_density": field_densities,
        "max_overlap_pair": {
            "persona_a": max_pair[0],
            "persona_b": max_pair[1],
            "similarity": max_pair[2],
        },
        "personas_meta": [
            {
                "name": p.get("name", "?"),
                "cluster_id": p.get("_meta", {}).get("cluster_id", ""),
                "groundedness": p.get("_meta", {}).get("groundedness", 0),
                "cost_usd": p.get("_meta", {}).get("cost_usd", 0),
            }
            for p in personas
        ],
        "heatmap_path": str(heatmap_path),
        "field_density_path": str(field_heatmap_path),
        "synthesis_cost_usd": synth_cost,
    }

    results_path = output_dir / "exp_6_13_results.json"
    results_path.write_text(json.dumps(results_data, indent=2))
    report_path = output_dir / "exp_6_13_report.txt"
    report_path.write_text(report)

    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to:  {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
