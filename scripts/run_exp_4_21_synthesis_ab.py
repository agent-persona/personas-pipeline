"""exp-4.21 synthesis A/B — humanize=False vs humanize=True on the same cluster.

What this runner does:
  1. Builds a minimal ClusterData fixture (tenant_acme_corp-style records).
  2. Calls synthesize() once with humanize=False (the pre-4.21 default).
  3. Calls synthesize() again with humanize=True (the new addendum path).
  4. Prints the vocabulary + sample_quotes side-by-side so the text delta
     is visible. Also reports groundedness scores so a regression is obvious.

Run from the pipeline repo root:
    ANTHROPIC_API_KEY=... python3 scripts/run_exp_4_21_synthesis_ab.py

Cost: 2 synthesis calls on Haiku, ~$0.02 total.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "synthesis"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / "synthesis" / ".env")

from anthropic import AsyncAnthropic  # noqa: E402

from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.synthesizer import synthesize  # noqa: E402
from synthesis.models.cluster import (  # noqa: E402
    ClusterData,
    ClusterSummary,
    EnrichmentPayload,
    SampleRecord,
    TenantContext,
)


def build_fixture_cluster() -> ClusterData:
    """Minimal cluster patterned on the acme_corp golden tenant — enough
    records to satisfy schema evidence minimums, but small enough to run
    cheaply twice.
    """
    return ClusterData(
        cluster_id="clust_ab_001",
        tenant=TenantContext(
            tenant_id="tenant_acme_corp",
            industry="B2B SaaS",
            product_description=(
                "Project management platform with API-first integrations"
            ),
            existing_persona_names=[],
        ),
        summary=ClusterSummary(
            cluster_size=6,
            top_behaviors=[
                "opens API docs",
                "submits webhook support tickets",
                "uses CLI more than UI",
            ],
            top_pages=["/api/docs", "/api/reference", "/settings/webhooks"],
            conversion_rate=0.35,
            avg_session_duration_seconds=2280.0,
            top_referrers=["github.com", "news.ycombinator.com"],
            extra={},
        ),
        sample_records=[
            SampleRecord(
                record_id="intercom_000",
                source="intercom",
                timestamp="2026-03-02T14:22:00Z",
                payload={
                    "message": (
                        "Your webhook retry logic is not idempotent. "
                        "We're getting duplicate events and it's breaking "
                        "our downstream pipeline."
                    ),
                    "subject": "Webhook idempotency",
                    "tone": "direct",
                },
            ),
            SampleRecord(
                record_id="intercom_001",
                source="intercom",
                timestamp="2026-03-05T09:10:00Z",
                payload={
                    "message": (
                        "I don't want to click anything. Just give me a "
                        "well-documented REST + GraphQL API with stable "
                        "versioning and I'll terraform the rest."
                    ),
                    "subject": "API wish list",
                    "tone": "pragmatic",
                },
            ),
            SampleRecord(
                record_id="ga4_000",
                source="ga4",
                timestamp="2026-03-06T22:45:00Z",
                payload={
                    "event": "api_reference_page_view",
                    "session_duration_s": 2280,
                    "scroll_depth": 0.92,
                    "from_referrer": "news.ycombinator.com",
                },
            ),
            SampleRecord(
                record_id="intercom_002",
                source="intercom",
                timestamp="2026-03-08T11:00:00Z",
                payload={
                    "message": (
                        "Terraform provider is community-maintained and the "
                        "last release was 6 months ago. We cannot depend on "
                        "that in production."
                    ),
                    "subject": "Terraform support",
                    "tone": "frustrated",
                },
            ),
        ],
        enrichment=EnrichmentPayload(
            firmographic={
                "company_size": "50-200 employees",
                "industry": "Fintech",
            },
            intent_signals=["high API doc engagement", "webhook config focus"],
            technographic={
                "tech_stack": "GitHub, Terraform, Slack, REST/GraphQL APIs",
            },
        ),
    )


def _format_list(label: str, items: list[str], width: int = 38) -> str:
    """Format a list of strings left/right-alignable in the A/B table."""
    if not items:
        return label + ": (empty)"
    lines = [f"{label}:"]
    for it in items:
        s = str(it)
        if len(s) > width - 2:
            s = s[: width - 5] + "..."
        lines.append(f"  • {s}")
    return "\n".join(lines)


async def main() -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1
    model = os.environ.get("default_model", "claude-haiku-4-5-20251001")

    cluster = build_fixture_cluster()
    client = AsyncAnthropic(api_key=api_key)
    backend = AnthropicBackend(client=client, model=model)

    print("=" * 78)
    print("exp-4.21 SYNTHESIS A/B — same cluster, two synthesize() calls")
    print("=" * 78)
    print(f"Cluster: {cluster.cluster_id} ({cluster.summary.cluster_size} records)")
    print(f"Model:   {model}")
    print()

    print("[1/2] Running humanize=False (pre-4.21 baseline) ...")
    result_off = await synthesize(cluster, backend, humanize=False)
    p_off = result_off.persona.model_dump(mode="json")

    print("[2/2] Running humanize=True (4.21 addendum path) ...")
    result_on = await synthesize(cluster, backend, humanize=True)
    p_on = result_on.persona.model_dump(mode="json")

    print()
    print("-" * 78)
    print(f"{'humanize=False':<38}  |  {'humanize=True':<38}")
    print("-" * 78)

    # Name
    print(f"NAME: {p_off.get('name', '?'):<31}  |  NAME: {p_on.get('name', '?')}")
    print()

    # Vocabulary — the voice-carrying list
    vocab_off = p_off.get("vocabulary") or []
    vocab_on = p_on.get("vocabulary") or []
    print("VOCABULARY:")
    max_v = max(len(vocab_off), len(vocab_on))
    for i in range(max_v):
        a = vocab_off[i] if i < len(vocab_off) else ""
        b = vocab_on[i] if i < len(vocab_on) else ""
        print(f"  {a:<36}  |  {b}")
    print()

    # Sample quotes — the voice carriers
    quotes_off = p_off.get("sample_quotes") or []
    quotes_on = p_on.get("sample_quotes") or []
    print("SAMPLE_QUOTES:")
    for i in range(max(len(quotes_off), len(quotes_on))):
        a = quotes_off[i] if i < len(quotes_off) else ""
        b = quotes_on[i] if i < len(quotes_on) else ""
        print(f"  [{i}] OFF: {a}")
        print(f"      ON : {b}")
        print()

    # Groundedness — regression check
    print("-" * 78)
    print("GROUNDEDNESS (regression check):")
    print(f"  humanize=False: {result_off.groundedness.score:.3f}  "
          f"({len(result_off.groundedness.violations)} violations)")
    print(f"  humanize=True : {result_on.groundedness.score:.3f}  "
          f"({len(result_on.groundedness.violations)} violations)")
    delta = result_on.groundedness.score - result_off.groundedness.score
    print(f"  delta         : {delta:+.3f}")

    if delta < -0.05:
        print("  WARNING: humanize=True regressed groundedness by >0.05 — "
              "do NOT flip default, revisit the addendum.")
    else:
        print("  OK: groundedness held within tolerance.")
    print()

    print("COST:")
    print(f"  humanize=False: ${result_off.total_cost_usd:.4f}  "
          f"({result_off.attempts} attempt(s))")
    print(f"  humanize=True : ${result_on.total_cost_usd:.4f}  "
          f"({result_on.attempts} attempt(s))")
    total = result_off.total_cost_usd + result_on.total_cost_usd
    print(f"  total         : ${total:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
