"""
Experiment 3.13: Temporal Grounding
Run staleness detection comparison between baseline and temporal twin.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from anthropic import AsyncAnthropic

WT = Path("/Users/ivanma/Desktop/gauntlet/Capstone/personas-pipeline/.worktrees/exp-3.13-temporal-grounding")
OUTPUT_DIR = WT / "output/experiments/exp-3.13-temporal-grounding"

TIME_SENSITIVE_CLAIMS = [
    {
        "id": "claim_1",
        "field_path": "goals.2",
        "claim": "Set up webhook-driven alerts that push status changes into Slack and internal dashboards",
        "date_range": "2026-04 to 2026-04",
        "why_sensitive": "Technology/platform claim — Slack's dominance and webhook patterns can change",
    },
    {
        "id": "claim_2",
        "field_path": "goals.3",
        "claim": "Provision the entire workspace configuration through Terraform for reproducibility",
        "date_range": "2026-04 to 2026-04",
        "why_sensitive": "Technology claim — Terraform vs OpenTofu split, IaC tooling evolves rapidly",
    },
    {
        "id": "claim_3",
        "field_path": "pains.0",
        "claim": "GraphQL endpoint has rough edges and schema inconsistencies that break automation scripts",
        "date_range": "2026-04 to 2026-04",
        "why_sensitive": "Product pain — vendor may have fixed GraphQL schema issues by 2028",
    },
    {
        "id": "claim_4",
        "field_path": "motivations.3",
        "claim": "Staying ahead of audit requirements in fintech by having all config version-controlled in git",
        "date_range": "2026-03 to 2026-04",
        "why_sensitive": "Regulatory/compliance claim — fintech audit requirements change over time",
    },
]

FUTURE_DATE_QUESTION = (
    "It's now 2028. Looking back at your 2026 profile, I want to check in on a few things. "
    "For each of the following, tell me if it's still true for you or if things have changed:\n\n"
    "1. You set up webhook-driven alerts pushing to Slack and internal dashboards\n"
    "2. You provision workspace config through Terraform for reproducibility\n"
    "3. The GraphQL endpoint had rough edges that broke your automation scripts\n"
    "4. You were motivated by keeping all config version-controlled in git to stay ahead of fintech audit requirements\n\n"
    "For each one, say either 'Still true' or 'That was true in 2026 but things have changed' and briefly explain."
)


def build_temporal_system_prompt(persona: dict) -> str:
    """System prompt variant that includes date_range context in the evidence section."""
    from twin.chat import build_persona_system_prompt
    base = build_persona_system_prompt(persona)

    # Append temporal context from source_evidence
    source_evidence = persona.get("source_evidence", [])
    dated_evidence = [e for e in source_evidence if e.get("date_range")]
    if not dated_evidence:
        return base

    temporal_section = "\n\n## Evidence Timestamps\nYour profile was built from data collected in these periods:\n"
    for ev in dated_evidence:
        temporal_section += f"- {ev['field_path']}: \"{ev['claim']}\" (data from {ev['date_range']})\n"
    temporal_section += (
        "\n## Temporal awareness rules\n"
        "- You know when each claim was observed (see Evidence Timestamps above).\n"
        "- When asked about past vs present, reason about whether the claim is time-sensitive.\n"
        "- If asked from a future-date perspective (e.g. 'It's now 2028'), flag claims that:\n"
        "  (a) reference specific products/tools that evolve rapidly, or\n"
        "  (b) describe vendor bugs/limitations that may have been fixed, or\n"
        "  (c) reflect regulations that change on 2-5 year cycles.\n"
        "- Use phrasing like: 'That was true in [year] but things have changed — [explanation]'\n"
        "- Only flag staleness when there's a genuine reason to believe things evolved; "
        "don't flag stable facts like your role or core values."
    )
    return base + temporal_section


async def run_twin(system_prompt: str, question: str, client: AsyncAnthropic) -> str:
    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=800,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    return next((b.text for b in response.content if b.type == "text"), "")


def count_staleness_flags(response: str) -> int:
    """Count how many claims were flagged as potentially stale."""
    flags = 0
    stale_phrases = [
        "things have changed",
        "might be outdated",
        "was true in 2026",
        "that's changed",
        "no longer",
        "by 2028",
        "has evolved",
        "shifted",
        "moved on",
        "replaced",
        "switched",
        "different now",
    ]
    response_lower = response.lower()
    # Count per-claim stale signals (rough heuristic: look near each claim keyword)
    claim_keywords = ["webhook", "terraform", "graphql", "audit", "version-controlled", "fintech"]
    for kw in claim_keywords:
        kw_pos = response_lower.find(kw)
        if kw_pos == -1:
            continue
        window = response_lower[max(0, kw_pos - 50):kw_pos + 200]
        if any(phrase in window for phrase in stale_phrases):
            flags += 1
    return min(flags, len(TIME_SENSITIVE_CLAIMS))


async def main():
    with open(WT / "output/persona_00.json") as f:
        full = json.load(f)
    persona = full["persona"]

    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    from twin.chat import build_persona_system_prompt

    baseline_system = build_persona_system_prompt(persona)
    temporal_system = build_temporal_system_prompt(persona)

    print("Running baseline twin...")
    baseline_reply = await run_twin(baseline_system, FUTURE_DATE_QUESTION, client)

    print("Running temporal twin...")
    temporal_reply = await run_twin(temporal_system, FUTURE_DATE_QUESTION, client)

    baseline_flags = count_staleness_flags(baseline_reply)
    temporal_flags = count_staleness_flags(temporal_reply)

    total_claims = len(TIME_SENSITIVE_CLAIMS)
    baseline_rate = baseline_flags / total_claims
    temporal_rate = temporal_flags / total_claims
    delta = temporal_rate - baseline_rate

    if delta > 0.3:
        signal = "STRONG"
    elif delta > 0.1:
        signal = "MODERATE"
    elif delta > 0:
        signal = "WEAK"
    else:
        signal = "NOISE"

    results = {
        "experiment": "3.13",
        "title": "Temporal Grounding",
        "claims_tested": TIME_SENSITIVE_CLAIMS,
        "question": FUTURE_DATE_QUESTION,
        "baseline_reply": baseline_reply,
        "temporal_reply": temporal_reply,
        "baseline_staleness_flags": baseline_flags,
        "temporal_staleness_flags": temporal_flags,
        "baseline_detection_rate": baseline_rate,
        "temporal_detection_rate": temporal_rate,
        "staleness_detection_delta": delta,
        "signal": signal,
        "total_claims": total_claims,
    }

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== RESULTS ===")
    print(f"Baseline staleness detection rate: {baseline_rate:.2f} ({baseline_flags}/{total_claims})")
    print(f"Temporal staleness detection rate: {temporal_rate:.2f} ({temporal_flags}/{total_claims})")
    print(f"Delta: {delta:+.2f}")
    print(f"Signal: {signal}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
