"""LinkedIn → personas experiment for shrutijn profile crawl."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic  # noqa: E402

from crawler import fetch_from_run  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.synthesizer import synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402

TENANT_ID = "tenant_shrutijn_experiment"
RUN_DIR = REPO_ROOT / "crawler/data/linkedin/shrutijn/2026-04-15"
OUTPUT_DIR = REPO_ROOT / "output/experiments/linkedin_shrutijn"


async def main() -> None:
    print(f"[1/4] Loading records from {RUN_DIR}")
    crawler_records = fetch_from_run(RUN_DIR, TENANT_ID)
    print(f"      Loaded {len(crawler_records)} flat records")
    for r in crawler_records:
        print(f"      - user={r.user_id} behaviors={r.behaviors[:5]} pages={r.pages[:3]}")

    raw_records = [RawRecord.model_validate(r.model_dump()) for r in crawler_records]

    print("\n[2/4] Segmenting (min_cluster_size=1 since single profile)")
    cluster_dicts = segment(
        raw_records,
        tenant_industry="Tech / AI tooling",
        tenant_product="Persona synthesis pipeline",
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=1,
    )
    print(f"      Got {len(cluster_dicts)} cluster(s)")

    if not cluster_dicts:
        print("      No clusters produced — nothing to synthesize. Exiting.")
        return

    clusters = [ClusterData.model_validate(c) for c in cluster_dicts]
    for c in clusters:
        print(
            f"      - {c.cluster_id} size={c.summary.cluster_size} "
            f"top_behaviors={c.summary.top_behaviors[:5]}"
        )

    print("\n[3/4] Synthesizing personas")
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not found in synthesis/.env")
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    from synthesis.engine.synthesizer import SynthesisError  # noqa: E402

    personas: list[dict] = []
    for i, cluster in enumerate(clusters):
        print(f"      [{i + 1}/{len(clusters)}] {cluster.cluster_id}")
        try:
            result = await synthesize(cluster, backend)
        except SynthesisError as exc:
            print(f"          FAILED: {exc}")
            for j, a in enumerate(exc.attempts):
                print(f"          attempt {j + 1} errors: {a.validation_errors or a.groundedness_violations}")
            raise
        personas.append(
            {
                "cluster_id": cluster.cluster_id,
                "persona": result.persona.model_dump(mode="json"),
                "cost_usd": result.total_cost_usd,
                "groundedness": result.groundedness.score,
                "attempts": result.attempts,
            }
        )
        print(
            f"          -> {result.persona.name} "
            f"(cost=${result.total_cost_usd:.4f}, groundedness={result.groundedness.score:.2f})"
        )

    print(f"\n[4/4] Writing {len(personas)} persona(s) to {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, entry in enumerate(personas):
        (OUTPUT_DIR / f"persona_{i:02d}.json").write_text(
            json.dumps(entry, indent=2, default=str)
        )
    print("      Done.")


if __name__ == "__main__":
    asyncio.run(main())
