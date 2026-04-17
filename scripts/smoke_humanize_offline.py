"""Offline smoke test for exp-4.21 humanization layer.

Runs the humanize() post-processor against three personas with different
communication_style profiles and prints before/after + timing breakdown.

No API key required — uses canned reply text rather than calling Claude.
The point is to validate the TEXT TRANSFORMATION + TIMING model, not the
LLM behavior. LLM-in-the-loop comes next via run_exp_4_21_humanization.py.

Run from repo root:
    python3 scripts/smoke_humanize_offline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "twin"))

from twin.humanize import HumanizeConfig, humanize  # noqa: E402


CANNED_REPLY = (
    "Great question! I think the webhook retry logic is absolutely the "
    "right place to start. It is worth noting that we need to leverage "
    "idempotent handlers here — otherwise a lot of things could go wrong. "
    "Furthermore, you are going to want observability on every retry. "
    "I hope this helps!"
)


def load_fixture(name: str) -> dict:
    path = REPO / "tests" / "fixtures" / f"{name}.json"
    return json.loads(path.read_text())["persona"]


def synthetic_casual_basic() -> dict:
    """A contrast persona — casual + basic + enthusiastic.

    Real pipeline output skews professional + advanced, so synthesize one
    here to exercise the full range of the humanizer.
    """
    return {
        "name": "Riley the Community Streamer",
        "communication_style": {
            "tone": "warm",
            "formality": "casual",
            "vocabulary_level": "basic",
            "preferred_channels": ["Discord", "Twitch chat"],
        },
        "emotional_profile": {
            "baseline_mood": "enthusiastic",
            "stress_triggers": ["stream going offline"],
            "coping_mechanisms": ["memes"],
        },
        "vocabulary": ["vibe", "based", "goated", "wild", "fr"],
    }


def synthetic_anxious_medium() -> dict:
    """Another contrast — professional + intermediate + anxious."""
    return {
        "name": "Jordan the New PM",
        "communication_style": {
            "tone": "analytical",
            "formality": "professional",
            "vocabulary_level": "intermediate",
            "preferred_channels": ["Slack"],
        },
        "emotional_profile": {
            "baseline_mood": "anxious",
            "stress_triggers": ["missed deadlines"],
            "coping_mechanisms": ["over-documents"],
        },
        "vocabulary": ["blocker", "scope", "risk", "mitigation"],
    }


def print_run(persona: dict, cfg: HumanizeConfig, label: str) -> None:
    cs = persona["communication_style"]
    ep = persona["emotional_profile"]
    header = (
        f"{label}: {persona['name']}\n"
        f"  formality={cs['formality']}  vocab={cs['vocabulary_level']}  "
        f"mood={ep['baseline_mood']}"
    )
    print("=" * 78)
    print(header)
    print("=" * 78)
    print("INPUT (raw LLM reply):")
    print(f"  {CANNED_REPLY}")
    print()

    chunks = humanize(CANNED_REPLY, persona, cfg)
    total_ms = sum(c.pre_delay_ms for c in chunks)
    print(f"OUTPUT: {len(chunks)} chunk(s), total wall time {total_ms/1000:.1f}s")
    for i, c in enumerate(chunks, 1):
        print(
            f"  [{i}] pre_delay={c.pre_delay_ms/1000:5.2f}s  "
            f"typing={c.typing_ms/1000:5.2f}s  | {c.text!r}"
        )
    print()


def main() -> int:
    personas: list[tuple[str, dict]] = [
        ("Alex (real fixture, advanced -> high register)", load_fixture("persona_00")),
        ("Maya (real fixture, intermediate -> medium register)", load_fixture("persona_01")),
        ("Riley (synthetic, casual+basic -> low register)", synthetic_casual_basic()),
        ("Jordan (synthetic, anxious mood -> fast+bursty timing)", synthetic_anxious_medium()),
    ]

    # Fixed seed so the output is stable run-to-run — change if you want
    # to see fresh samples of the random choices.
    cfg = HumanizeConfig(seed=42, enable_grammar_slips=True, enable_emoji_chunks=True)

    for label, persona in personas:
        print_run(persona, cfg, label)

    # Bonus: same Riley with 3 different seeds to show stochastic variation.
    print("=" * 78)
    print("STOCHASTIC VARIATION — Riley across 3 seeds (shows slip/emoji rolls)")
    print("=" * 78)
    riley = synthetic_casual_basic()
    for seed in (1, 2, 3):
        chunks = humanize(CANNED_REPLY, riley, HumanizeConfig(seed=seed))
        print(f"\n  seed={seed}: {len(chunks)} chunks")
        for i, c in enumerate(chunks, 1):
            print(f"    [{i}] {c.text!r}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
