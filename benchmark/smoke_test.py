"""Quick smoke test after each merge — verifies imports work and a
single small tenant can synthesize successfully.

Runs only bench_sparse_30 (30 records, ~2 clusters, ~45s).
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "twin"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / "synthesis" / ".env")


def test_imports() -> bool:
    """Verify core modules import cleanly."""
    try:
        from synthesis.engine.synthesizer import synthesize  # noqa
        from synthesis.engine.prompt_builder import build_messages, SYSTEM_PROMPT  # noqa
        from synthesis.engine.groundedness import check_groundedness  # noqa
        from synthesis.engine.model_backend import AnthropicBackend  # noqa
        from synthesis.models.persona import PersonaV1  # noqa
        from synthesis.models.cluster import ClusterData  # noqa
        from evaluation.judges import LLMJudge  # noqa
        from twin import TwinChat  # noqa
        from segmentation.pipeline import segment  # noqa
        print("  [ok] imports")
        return True
    except Exception as e:
        print(f"  [FAIL] imports: {type(e).__name__}: {e}")
        return False


async def test_synthesis() -> bool:
    """Run a single tenant benchmark."""
    try:
        sys.path.insert(0, str(REPO_ROOT / "benchmark"))
        from tenants import load_tenant
        from segmentation.models.record import RawRecord
        from segmentation.pipeline import segment
        from synthesis.engine.model_backend import AnthropicBackend
        from synthesis.engine.synthesizer import synthesize
        from synthesis.models.cluster import ClusterData
        from synthesis.config import Settings
        from anthropic import AsyncAnthropic

        settings = Settings()
        if not settings.anthropic_api_key:
            print("  [FAIL] no API key in synthesis/.env")
            return False

        _, records, meta = load_tenant("bench_sparse_30")
        raw = [RawRecord.model_validate(r.model_dump()) for r in records]
        clusters_raw = segment(
            raw,
            tenant_industry=meta.get("industry"),
            tenant_product=meta.get("product"),
            existing_persona_names=[],
            similarity_threshold=0.15,
            min_cluster_size=2,
        )
        clusters = [ClusterData.model_validate(c) for c in clusters_raw]
        print(f"  [ok] segmented {len(raw)} records into {len(clusters)} clusters")

        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        backend = AnthropicBackend(client=client, model=settings.default_model)

        t0 = time.monotonic()
        successes = 0
        for c in clusters:
            try:
                r = await synthesize(c, backend)
                successes += 1
                print(f"    -> {r.persona.name} (g={r.groundedness.score:.2f}, ${r.total_cost_usd:.4f})")
            except Exception as e:
                print(f"    -> FAILED: {type(e).__name__}: {str(e)[:100]}")

        elapsed = time.monotonic() - t0
        if successes == 0:
            print(f"  [FAIL] synthesis: 0/{len(clusters)} in {elapsed:.1f}s")
            return False
        print(f"  [ok] synthesis: {successes}/{len(clusters)} in {elapsed:.1f}s")
        return True
    except Exception as e:
        import traceback
        print(f"  [FAIL] synthesis: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


async def main() -> int:
    print("SMOKE TEST")
    print("=" * 60)
    if not test_imports():
        return 1
    ok = await test_synthesis()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
