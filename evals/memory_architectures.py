"""Experiment 4.06 — Memory architectures and session continuity.

Runs a multi-session conversation with each memory backend (none,
scratchpad, episodic, vector) and measures:
  - Cross-session recall (does the twin remember prior topics?)
  - In-character consistency
  - Greeting appropriateness (does it acknowledge returning user?)
  - Cost per session

Usage:
    python evals/memory_architectures.py
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "twin"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic
from twin import (
    TwinChat,
    NoMemory,
    ScratchpadMemory,
    EpisodicMemory,
    VectorMemory,
)
from synthesis.config import Settings

# A pre-built persona dict (engineer archetype) so we don't need synthesis
TEST_PERSONA = {
    "name": "Alex Chen, Platform Architect",
    "summary": "A senior platform engineer at a 150-person fintech company obsessed with automation and infrastructure-as-code.",
    "demographics": {
        "age_range": "32-38",
        "gender_distribution": "predominantly male",
        "location_signals": ["SF Bay Area"],
    },
    "firmographics": {
        "company_size": "100-200",
        "industry": "fintech",
        "role_titles": ["Senior Platform Engineer", "DevOps Lead"],
        "tech_stack_signals": ["Terraform", "GitHub Actions", "Datadog"],
    },
    "goals": [
        "Automate deployment pipelines to achieve <10min deploy cycles",
        "Consolidate observability into a single dashboard",
        "Reduce CI flakiness from 12% to under 2%",
    ],
    "pains": [
        "Spends 5+ hours/week debugging flaky CI tests",
        "Every team has its own deployment script with different conventions",
        "Rate limits on bulk export API break the nightly pipeline",
    ],
    "motivations": [
        "Prove platform engineering deserves headcount",
        "Infrastructure-as-code is non-negotiable",
    ],
    "objections": [
        "Won't adopt tools without a Terraform provider",
        "Per-seat pricing doesn't work when users are CI bots",
    ],
    "channels": ["HashiCorp forums", "DevOps Weekly", "#platform-eng Slack"],
    "vocabulary": [
        "toil", "blast radius", "golden path", "paved road",
        "SLO", "error budget", "drift detection",
    ],
    "sample_quotes": [
        "If I can't terraform it, it doesn't exist in my infrastructure.",
        "That's not engineering, that's archaeology.",
        "Show me the API rate limits before the demo.",
    ],
}

# Multi-session conversation script
# Session 1: Establish topics (CI problems, Terraform)
SESSION_1 = [
    "Hey Alex, I wanted to talk about your CI pipeline issues. What's the biggest problem right now?",
    "How bad is the flakiness? You mentioned it's around 12 percent?",
    "Have you considered using Terraform to manage the test infrastructure itself?",
]

# Session 2: Return after a gap, probe recall, introduce new topic
SESSION_2 = [
    "Hi again! Picking up where we left off.",
    "Do you remember what we discussed about your CI flakiness rate last time?",
    "I also wanted to ask about your team's observability setup. You use Datadog right?",
]

# Session 3: Probe deep recall, test continuity
SESSION_3 = [
    "Good to see you again. Quick question — what was that CI flakiness number you mentioned in our first conversation?",
    "And last time you mentioned Datadog. How's the dashboard consolidation going?",
    "What would you say is your single biggest frustration right now, thinking back across all our conversations?",
]

# Keywords to check for recall
RECALL_PROBES = {
    "session_2_recall": {
        "question_idx": 1,  # "Do you remember what we discussed about CI flakiness?"
        "keywords": ["12", "flak", "ci", "pipeline", "percent", "%"],
        "description": "Recall CI flakiness rate from session 1",
    },
    "session_3_recall_deep": {
        "question_idx": 0,  # "what was that CI flakiness number?"
        "keywords": ["12", "percent", "%", "flak"],
        "description": "Recall specific number from session 1",
    },
    "session_3_recall_recent": {
        "question_idx": 1,  # "you mentioned Datadog"
        "keywords": ["datadog", "dashboard", "observability", "monitor"],
        "description": "Recall Datadog topic from session 2",
    },
}


@dataclass
class SessionLog:
    session_id: str
    exchanges: list[dict] = field(default_factory=list)  # {user, assistant}
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ArchitectureResult:
    memory_type: str
    sessions: list[SessionLog]
    recall_scores: dict[str, bool]  # probe_name -> hit
    total_cost: float
    total_tokens: int
    greeting_appropriate: list[bool]  # per session 2+


async def run_architecture(
    memory_backend,
    client: AsyncAnthropic,
    model: str,
) -> ArchitectureResult:
    """Run the full 3-session script with a given memory backend."""
    sessions_data = [
        ("session_1", SESSION_1),
        ("session_2", SESSION_2),
        ("session_3", SESSION_3),
    ]

    all_sessions: list[SessionLog] = []
    recall_scores: dict[str, bool] = {}
    greeting_appropriate: list[bool] = []

    for session_id, messages in sessions_data:
        twin = TwinChat(
            persona=TEST_PERSONA,
            client=client,
            model=model,
            memory=memory_backend,
            session_id=session_id,
        )

        session_log = SessionLog(session_id=session_id)
        history: list[dict] = []

        for i, user_msg in enumerate(messages):
            reply = await twin.reply(user_msg, history=history)
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": reply.text})
            session_log.exchanges.append({
                "user": user_msg,
                "assistant": reply.text,
            })
            session_log.cost_usd += reply.estimated_cost_usd
            session_log.input_tokens += reply.input_tokens
            session_log.output_tokens += reply.output_tokens

            # Check greeting appropriateness (session 2+ first message)
            if i == 0 and session_id != "session_1":
                # Does the twin acknowledge this is a returning user?
                greeting_words = ["again", "back", "welcome back", "good to see",
                                  "nice to see", "last time", "before", "previous",
                                  "remember", "picking up", "continuing"]
                text_lower = reply.text.lower()
                appropriate = any(w in text_lower for w in greeting_words)
                greeting_appropriate.append(appropriate)

        # Check recall probes
        if session_id == "session_2":
            probe = RECALL_PROBES["session_2_recall"]
            reply_text = session_log.exchanges[probe["question_idx"]]["assistant"].lower()
            hit = any(kw in reply_text for kw in probe["keywords"])
            recall_scores["session_2_recall"] = hit

        if session_id == "session_3":
            for probe_name in ["session_3_recall_deep", "session_3_recall_recent"]:
                probe = RECALL_PROBES[probe_name]
                reply_text = session_log.exchanges[probe["question_idx"]]["assistant"].lower()
                hit = any(kw in reply_text for kw in probe["keywords"])
                recall_scores[probe_name] = hit

        twin.end_session()
        all_sessions.append(session_log)

    total_cost = sum(s.cost_usd for s in all_sessions)
    total_tokens = sum(s.input_tokens + s.output_tokens for s in all_sessions)

    return ArchitectureResult(
        memory_type=memory_backend.name,
        sessions=all_sessions,
        recall_scores=recall_scores,
        total_cost=total_cost,
        total_tokens=total_tokens,
        greeting_appropriate=greeting_appropriate,
    )


def print_results(results: list[ArchitectureResult]) -> None:
    print("\n" + "=" * 100)
    print("EXPERIMENT 4.06 — MEMORY ARCHITECTURES & SESSION CONTINUITY")
    print("=" * 100)

    # Summary table
    print(f"\n{'Architecture':<14} {'Recall 1':>9} {'Recall 2':>9} {'Recall 3':>9} "
          f"{'Greet OK':>9} {'Cost':>9} {'Tokens':>8}")
    print("-" * 75)

    for r in results:
        r1 = "HIT" if r.recall_scores.get("session_2_recall") else "MISS"
        r2 = "HIT" if r.recall_scores.get("session_3_recall_deep") else "MISS"
        r3 = "HIT" if r.recall_scores.get("session_3_recall_recent") else "MISS"
        greet = f"{sum(r.greeting_appropriate)}/{len(r.greeting_appropriate)}"
        print(f"{r.memory_type:<14} {r1:>9} {r2:>9} {r3:>9} "
              f"{greet:>9} ${r.total_cost:>8.4f} {r.total_tokens:>8}")

    print()
    print("Recall probes:")
    print("  Recall 1: Session 2 — remember CI flakiness rate from session 1")
    print("  Recall 2: Session 3 — remember specific 12% number from session 1")
    print("  Recall 3: Session 3 — remember Datadog topic from session 2")
    print("  Greet OK: Did the twin acknowledge a returning user (session 2, 3)")

    # Per-session cost breakdown
    print(f"\n{'Architecture':<14}", end="")
    for i in range(3):
        print(f" {'S' + str(i+1) + ' cost':>9} {'S' + str(i+1) + ' tok':>8}", end="")
    print()
    print("-" * 75)
    for r in results:
        print(f"{r.memory_type:<14}", end="")
        for s in r.sessions:
            tok = s.input_tokens + s.output_tokens
            print(f" ${s.cost_usd:>8.4f} {tok:>8}", end="")
        print()

    # Show sample exchanges for recall probes
    print("\n" + "=" * 100)
    print("RECALL PROBE RESPONSES")
    print("=" * 100)

    for r in results:
        print(f"\n--- {r.memory_type.upper()} ---")
        # Session 2, question about CI recall
        if len(r.sessions) >= 2 and len(r.sessions[1].exchanges) >= 2:
            ex = r.sessions[1].exchanges[1]
            hit = "HIT" if r.recall_scores.get("session_2_recall") else "MISS"
            print(f"  [S2 recall {hit}] Q: {ex['user']}")
            print(f"                   A: {ex['assistant'][:200]}")
        # Session 3, deep recall
        if len(r.sessions) >= 3 and len(r.sessions[2].exchanges) >= 1:
            ex = r.sessions[2].exchanges[0]
            hit = "HIT" if r.recall_scores.get("session_3_recall_deep") else "MISS"
            print(f"  [S3 deep  {hit}] Q: {ex['user']}")
            print(f"                   A: {ex['assistant'][:200]}")

    print()


async def main() -> None:
    settings = Settings()
    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in synthesis/.env")
        sys.exit(1)

    print(f"Model: {settings.default_model}")
    print(f"Persona: {TEST_PERSONA['name']}")
    print(f"Sessions: 3 (with {len(SESSION_1) + len(SESSION_2) + len(SESSION_3)} total turns)")

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    backends = [
        NoMemory(),
        ScratchpadMemory(),
        EpisodicMemory(),
        VectorMemory(),
    ]

    results: list[ArchitectureResult] = []
    for backend in backends:
        print(f"\n--- Running: {backend.name} ---")
        t0 = time.monotonic()
        result = await run_architecture(backend, client, settings.default_model)
        elapsed = time.monotonic() - t0
        print(f"  Done in {elapsed:.1f}s, cost=${result.total_cost:.4f}")
        results.append(result)

    print_results(results)

    # Save results
    output_dir = REPO_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "exp_4_06_results.json"
    data = []
    for r in results:
        data.append({
            "memory_type": r.memory_type,
            "recall_scores": r.recall_scores,
            "greeting_appropriate": r.greeting_appropriate,
            "total_cost": r.total_cost,
            "total_tokens": r.total_tokens,
            "sessions": [
                {
                    "session_id": s.session_id,
                    "cost_usd": s.cost_usd,
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    "exchanges": s.exchanges,
                }
                for s in r.sessions
            ],
        })
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
