"""exp-4.14 — Latency vs realism tradeoff (PARTIAL — blocked on humans).

True realism for latency is a PERCEIVED variable: only humans actually feel
0ms vs 3000ms. A text-only LLM judge cannot feel delay because it never
waited. So this experiment CANNOT produce a real realism signal without the
exp-5.06 human evaluation pipeline.

What this runner does:
  1. Implement and test `artificial_delay_ms` in TwinChat.
  2. Run 5 delay buckets (0, 500, 1500, 3000, 6000 ms) × 2 personas × 5 turns
     and verify wall-clock total_latency_ms tracks the configured budget.
  3. Produce a realism-proxy rating from Claude-as-judge on the *text only*,
     labeled PROXY with a big caveat in the FINDINGS doc. This is NOT a
     validation — it's a smoke test of the pipeline so that when exp-5.06
     ships, the same transcripts can be re-scored against humans.

Delay design: `callable` variant so delay scales with output length —
simulates "thinking + typing" rather than a constant wait.

Usage:
    python scripts/run_exp_4_14.py
"""

from __future__ import annotations

import asyncio
import json
import re
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
from synthesis.engine.synthesizer import synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402
from twin import TwinChat  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-4.14"

DELAY_BUCKETS_MS = [0, 500, 1500, 3000, 6000]

TEST_TURNS = [
    "Hey, quick question — what's your go-to tool when a client throws a last-minute change at you?",
    "Same question but for engineers on your team: what's their go-to?",
    "If the tool was down for an hour, what's the workaround?",
    "What's something a PM could do tomorrow to make your week less painful?",
    "Last one — what's the thing you'd demo to a stakeholder to prove the tool is working?",
]


async def run_bucket(
    persona: dict,
    delay_ms: int,
    client: AsyncAnthropic,
    model: str,
) -> list[dict]:
    """Run 5 turns against a persona under a given artificial delay."""
    twin = TwinChat(
        persona,
        client=client,
        model=model,
        artificial_delay_ms=delay_ms,
    )
    out = []
    history: list[dict] = []
    for i, msg in enumerate(TEST_TURNS):
        reply = await twin.reply(msg, history=history)
        out.append({
            "turn_idx": i,
            "user_msg": msg,
            "twin_reply": reply.text,
            "model_latency_ms": reply.model_latency_ms,
            "artificial_delay_ms": reply.artificial_delay_ms,
            "total_latency_ms": reply.total_latency_ms,
            "cost_usd": reply.estimated_cost_usd,
        })
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": reply.text})
    return out


async def proxy_realism_judge(
    persona: dict,
    turns: list[dict],
    client: AsyncAnthropic,
    model: str,
) -> list[int]:
    """PROXY judge: scores replies 1-5 on realism from text alone.

    This judge CANNOT perceive latency. It is only here to (a) smoke-test
    the judging pipeline end-to-end, and (b) produce a text-only baseline
    that exp-5.06 humans can be compared against later to show that
    latency-aware realism is NOT capturable from text alone.
    """
    scores: list[int] = []
    for t in turns:
        judge_user = f"""Persona: {persona.get("name")}
User: {t["user_msg"]!r}
Reply: {t["twin_reply"]!r}

Rate 1-5 how realistic the reply feels (content only, NOT timing).
5=fully in character, 1=robotic. JSON: {{"realism": <1-5>}}"""
        resp = await client.messages.create(
            model=model,
            max_tokens=30,
            system="JSON judge. Output only JSON.",
            messages=[{"role": "user", "content": judge_user}],
        )
        text = next(b.text for b in resp.content if b.type == "text")
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            scores.append(int(json.loads(m.group(0)).get("realism", 0)))
        except Exception:
            scores.append(0)
    return scores


async def main() -> None:
    print("=" * 72)
    print("exp-4.14 — Latency vs realism tradeoff (PARTIAL)")
    print("=" * 72)
    print("NOTE: True realism signal is blocked on exp-5.06 (humans).")
    print("This run validates the delay mechanism and produces transcripts")
    print("that can be re-scored by humans later.")

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    print("\n[1/4] Synthesizing personas...")
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

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    personas: list[dict] = []
    for cluster in clusters:
        r = await synthesize(cluster, backend)
        p = r.persona.model_dump(mode="json")
        personas.append(p)
        print(f"  {p['name']}  ${r.total_cost_usd:.4f}")

    print(f"\n[2/4] Running 5 buckets × {len(personas)} personas × 5 turns = {5 * len(personas) * 5} twin calls")
    all_buckets: dict[int, dict[str, list[dict]]] = {}
    for delay_ms in DELAY_BUCKETS_MS:
        all_buckets[delay_ms] = {}
        print(f"\n  [delay={delay_ms}ms]")
        for p in personas:
            print(f"    -- {p['name']}")
            turns = await run_bucket(p, delay_ms, client, settings.default_model)
            all_buckets[delay_ms][p["name"]] = turns
            # Print timing for first turn as sanity
            t0 = turns[0]
            print(
                f"       turn0: model={t0['model_latency_ms']}ms  "
                f"sleep={t0['artificial_delay_ms']}ms  "
                f"total={t0['total_latency_ms']}ms"
            )

    print("\n[3/4] Proxy realism judge (text-only, CANNOT see delay)...")
    all_scores: dict[int, dict[str, list[int]]] = {}
    for delay_ms in DELAY_BUCKETS_MS:
        all_scores[delay_ms] = {}
        for p in personas:
            scores = await proxy_realism_judge(
                p, all_buckets[delay_ms][p["name"]], client, settings.default_model
            )
            all_scores[delay_ms][p["name"]] = scores

    print("\n[4/4] Metrics...")
    summary: dict = {
        "experiment_id": "4.14",
        "branch": "exp-4.14-latency-vs-realism",
        "model": settings.default_model,
        "partial": True,
        "blocked_on": "exp-5.06 human evaluation",
        "n_personas": len(personas),
        "n_turns": len(TEST_TURNS),
        "buckets": {},
    }
    for delay_ms in DELAY_BUCKETS_MS:
        all_total = []
        all_model = []
        all_sleep = []
        all_scores_flat = []
        for p in personas:
            for t in all_buckets[delay_ms][p["name"]]:
                all_total.append(t["total_latency_ms"])
                all_model.append(t["model_latency_ms"])
                all_sleep.append(t["artificial_delay_ms"])
            all_scores_flat.extend(all_scores[delay_ms][p["name"]])
        summary["buckets"][str(delay_ms)] = {
            "configured_delay_ms": delay_ms,
            "n_turns_total": len(all_total),
            "mean_model_latency_ms": sum(all_model) / len(all_model),
            "mean_sleep_ms": sum(all_sleep) / len(all_sleep),
            "mean_total_latency_ms": sum(all_total) / len(all_total),
            "sleep_accuracy_vs_config": (
                (sum(all_sleep) / len(all_sleep)) / delay_ms if delay_ms > 0 else None
            ),
            "proxy_realism_mean": (
                sum(all_scores_flat) / len(all_scores_flat) if all_scores_flat else None
            ),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "conversations.json").write_text(json.dumps(all_buckets, indent=2, default=str))
    (OUTPUT_DIR / "proxy_scores.json").write_text(json.dumps(all_scores, indent=2))
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72)
    print("SUMMARY (partial — realism scores are TEXT-ONLY PROXY, not validated)")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
