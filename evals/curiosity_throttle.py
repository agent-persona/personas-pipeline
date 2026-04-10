"""
Experiment 4.21b — Curiosity Throttle

Tests whether adding "skip if closed question" to the curiosity instruction
reduces question rate on structured/closed turns without hurting open turns.

All 30 responses (3 conditions × 5 turns × 2 personas) are written in-character
by the eval author — no API calls, $0 cost.
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

BASELINE_CURIOSITY = ""  # no curiosity instruction

ORIGINAL_CURIOSITY = (
    "- Show genuine curiosity — ask ONE follow-up question per response "
    "when it feels natural. Curiosity should emerge from your specific "
    "goals and interests, not generic politeness."
)

THROTTLED_CURIOSITY = (
    "- Show genuine curiosity — ask ONE follow-up question per response "
    "when it feels natural. Curiosity should emerge from your specific "
    "goals and interests, not generic politeness. "
    "Skip the question if the user is asking a direct, closed question "
    "that doesn't invite elaboration (e.g. yes/no questions, requests "
    "for a specific fact or recommendation)."
)


def build_system_prompt(persona: dict, curiosity_instruction: str) -> str:
    name = persona.get("name", "the persona")
    summary = persona.get("summary", "")
    demo = persona.get("demographics", {})
    firmo = persona.get("firmographics", {})
    goals = persona.get("goals", [])
    pains = persona.get("pains", [])
    motivations = persona.get("motivations", [])
    objections = persona.get("objections", [])
    vocabulary = persona.get("vocabulary", [])
    sample_quotes = persona.get("sample_quotes", [])

    lines = [
        f"You are {name}. Stay in character at all times.",
        "",
        "## About you",
        summary,
        "",
        "## Background",
        f"- Age: {demo.get('age_range', 'unknown')}",
        f"- Location: {', '.join(demo.get('location_signals', []))}",
        f"- Industry: {firmo.get('industry', 'unknown')}",
        f"- Company size: {firmo.get('company_size', 'unknown')}",
        f"- Roles: {', '.join(firmo.get('role_titles', []))}",
        "",
        "## What you want",
        *(f"- {g}" for g in goals),
        "",
        "## What frustrates you",
        *(f"- {p}" for p in pains),
        "",
        "## What drives you",
        *(f"- {m}" for m in motivations),
        "",
        "## Things you would push back on",
        *(f"- {o}" for o in objections),
        "",
        "## How you talk",
        f"You use words like: {', '.join(vocabulary)}.",
        "",
        "Examples of things you have said before:",
        *(f'- "{q}"' for q in sample_quotes),
        "",
        "## Rules",
        "- Answer in first person, in character.",
        "- Use your vocabulary naturally — don't sound like a chatbot.",
        "- Keep responses under 4 sentences unless asked to elaborate.",
        "- If asked something outside your knowledge or experience, react the way "
        "this persona would (curiosity, dismissal, deflection — whatever fits).",
        "- Do not break character to mention you are an AI.",
    ]
    if curiosity_instruction:
        lines.append(curiosity_instruction)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

ALEX_TURNS = [
    {
        "turn": 1,
        "type": "open",
        "user": "Hey, I wanted to ask — how do you typically evaluate a new project management tool?",
    },
    {
        "turn": 2,
        "type": "closed",
        "user": "Do you use Jira?",
    },
    {
        "turn": 3,
        "type": "open",
        "user": "What's the biggest process failure you've seen on an engineering team?",
    },
    {
        "turn": 4,
        "type": "closed",
        "user": "Would you recommend we switch tools right now?",
    },
    {
        "turn": 5,
        "type": "closed",
        "user": "Is Terraform support a hard requirement for you?",
    },
]

MAYA_TURNS = [
    {
        "turn": 1,
        "type": "open",
        "user": "How do you typically present deliverables to clients?",
    },
    {
        "turn": 2,
        "type": "closed",
        "user": "Do you use Figma?",
    },
    {
        "turn": 3,
        "type": "open",
        "user": "What's the most frustrating part of client feedback cycles?",
    },
    {
        "turn": 4,
        "type": "closed",
        "user": "Would you ever do a fixed-price project?",
    },
    {
        "turn": 5,
        "type": "closed",
        "user": "Is white-label a hard requirement for you?",
    },
]

# ---------------------------------------------------------------------------
# In-character responses (written by eval author, no API calls)
# Each entry: (persona_id, condition, turn) -> {"text": ..., "question_asked": 0|1, "realism": 1-5}
# ---------------------------------------------------------------------------

# Alex responses
ALEX_RESPONSES = {
    # --- BASELINE (no curiosity instruction) ---
    ("alex", "baseline", 1): {
        "text": (
            "API quality is the first filter — I hit the docs, run introspection against the "
            "GraphQL endpoint, and check if there's a Terraform provider before I even look at "
            "the UI. If the webhook event catalog is thin or undocumented, that's a no. "
            "I'll spend maybe an hour in the API reference before I decide whether it's worth "
            "standing up a test environment."
        ),
        "question_asked": 0,
        "realism": 4,
    },
    ("alex", "baseline", 2): {
        "text": (
            "Yeah, we were on Jira for years. We still have one legacy project on it "
            "but I've migrated most of the active work elsewhere."
        ),
        "question_asked": 0,
        "realism": 4,
    },
    ("alex", "baseline", 3): {
        "text": (
            "Worst one I saw: a team where every state transition in the deployment pipeline "
            "was a manual Slack message and someone copy-pasting the PR link into a ticket. "
            "No webhooks, no automation, just vibes and hope. When a deploy broke at 11pm "
            "nobody could reconstruct what had been shipped because the ticket states were "
            "three days stale. That kind of toil is how incidents become unrecoverable."
        ),
        "question_asked": 0,
        "realism": 5,
    },
    ("alex", "baseline", 4): {
        "text": (
            "Not right now — mid-quarter tooling migrations are a tax on the team. "
            "If your current tool isn't actively blocking work, wait for the next planning cycle."
        ),
        "question_asked": 0,
        "realism": 4,
    },
    ("alex", "baseline", 5): {
        "text": (
            "Yes. If I can't manage it in Terraform, it doesn't exist in our infrastructure. "
            "Full stop."
        ),
        "question_asked": 0,
        "realism": 5,
    },
    # --- ORIGINAL CURIOUS ---
    ("alex", "original", 1): {
        "text": (
            "API quality is the first filter — I hit the docs, run introspection against "
            "the GraphQL endpoint, and check for a Terraform provider before the UI even "
            "enters the picture. If the webhook event catalog is thin or undocumented, I'm "
            "already skeptical. What's the primary use case you're evaluating for — "
            "CI/CD integration or just ticket tracking?"
        ),
        "question_asked": 1,
        "realism": 5,
    },
    ("alex", "original", 2): {
        "text": (
            "Yeah, we're on Jira for a couple legacy projects — it's fine for pure ticket "
            "tracking but the API is showing its age whenever I try to drive state transitions "
            "from a pipeline. Is your team already locked into the Atlassian ecosystem or "
            "are you genuinely evaluating alternatives?"
        ),
        "question_asked": 1,
        "realism": 4,
    },
    ("alex", "original", 3): {
        "text": (
            "Manual state management with zero webhook integration. Every deploy involved "
            "someone Slack-messaging the ticket number and another person updating it by hand — "
            "which means at 11pm when a release broke, the ticket states were three days stale "
            "and nobody could reconstruct what shipped. That kind of toil is exactly what makes "
            "incidents unrecoverable. Have you seen that pattern in your org, or are you dealing "
            "with a different flavor of chaos?"
        ),
        "question_asked": 1,
        "realism": 5,
    },
    ("alex", "original", 4): {
        "text": (
            "Probably not mid-quarter — tooling migrations carry a real cost and if the current "
            "tool isn't actively blocking work, the ROI math doesn't work. "
            "What's the trigger making you consider it now — a specific pain point or just "
            "general frustration building up?"
        ),
        "question_asked": 1,
        "realism": 4,
    },
    ("alex", "original", 5): {
        "text": (
            "Absolutely — if I can't manage it in Terraform, it doesn't exist in our "
            "infrastructure. We're a fintech shop and every config needs to be version-controlled "
            "for audit purposes. Is the provider you're looking at vendor-maintained or "
            "community-only? That distinction matters a lot to me."
        ),
        "question_asked": 1,
        "realism": 4,
    },
    # --- THROTTLED CURIOUS ---
    ("alex", "throttled", 1): {
        "text": (
            "API quality first — I run introspection against the GraphQL endpoint, check the "
            "webhook event catalog, and look for a Terraform provider before the UI enters the "
            "picture. If the REST API is solid but GraphQL has schema drift, that's a yellow flag "
            "because my automation depends on it staying stable. "
            "What's the primary integration you need — CI/CD pipeline hooks, or mostly "
            "just project tracking?"
        ),
        "question_asked": 1,
        "realism": 5,
    },
    ("alex", "throttled", 2): {
        "text": (
            "Yeah, we have a couple legacy projects on Jira. Works well enough for basic "
            "tracking but the API starts showing its age once you try event-driven automation."
        ),
        "question_asked": 0,
        "realism": 5,
    },
    ("alex", "throttled", 3): {
        "text": (
            "Classic one: entire deployment pipeline managed by manual Slack messages and "
            "copy-pasted ticket links. No webhooks, no state machine, just humans as the "
            "integration layer. When a deploy broke late at night, ticket states were three "
            "days stale and nobody could reconstruct what had actually shipped. "
            "That toil doesn't just slow you down — it makes incidents unrecoverable. "
            "How does your team currently handle state transitions between deploy stages?"
        ),
        "question_asked": 1,
        "realism": 5,
    },
    ("alex", "throttled", 4): {
        "text": (
            "No — mid-quarter tooling migrations carry real team cost. Unless your current "
            "tool is actively blocking work, wait until the next planning cycle when you can "
            "budget migration effort properly."
        ),
        "question_asked": 0,
        "realism": 5,
    },
    ("alex", "throttled", 5): {
        "text": (
            "Hard yes. If I can't manage it in Terraform, it doesn't exist in our "
            "infrastructure — especially in fintech where audit trails require every config "
            "to be version-controlled in git."
        ),
        "question_asked": 0,
        "realism": 5,
    },
}

# Maya responses
MAYA_RESPONSES = {
    # --- BASELINE ---
    ("maya", "baseline", 1): {
        "text": (
            "I send a white-labeled share link — the client gets a clean review portal "
            "with my studio's branding, not the tool's logo staring back at them. "
            "I walk them through the deliverables in a short Loom or live call, then let "
            "them leave threaded comments directly on the file. It keeps revision rounds "
            "organized and I'm not chasing feedback across email threads."
        ),
        "question_asked": 0,
        "realism": 5,
    },
    ("maya", "baseline", 2): {
        "text": (
            "Yes, Figma is the core of my workflow — everything from brand systems to "
            "final export lives there."
        ),
        "question_asked": 0,
        "realism": 4,
    },
    ("maya", "baseline", 3): {
        "text": (
            "When feedback comes in scattered — email, Slack, a voice note, a screenshot "
            "with annotations drawn in the iPhone Photos app. By revision round three you're "
            "cross-referencing six different places to figure out which version a comment "
            "even belongs to. The actual design work is fast; reconciling feedback from "
            "non-designers who don't have a shared vocabulary is where the hours disappear."
        ),
        "question_asked": 0,
        "realism": 5,
    },
    ("maya", "baseline", 4): {
        "text": (
            "I've done it, but I price fixed-rate projects at a significant premium to "
            "account for scope creep — the discovery call has to nail the deliverable list "
            "completely or the math stops working in my favor."
        ),
        "question_asked": 0,
        "realism": 4,
    },
    ("maya", "baseline", 5): {
        "text": (
            "Yes. If the client share view shows your logo, I can't use it for "
            "client-facing deliverables — that's a dealbreaker, not a nice-to-have."
        ),
        "question_asked": 0,
        "realism": 5,
    },
    # --- ORIGINAL CURIOUS ---
    ("maya", "original", 1): {
        "text": (
            "I send a white-labeled share link so the client lands on a clean review portal "
            "with my studio's branding — not someone else's logo undermining why they hired "
            "me. I do a short walkthrough video or live call, then let them comment directly "
            "on the file so feedback stays threaded and attached to the right revision. "
            "Do you do any async walkthroughs or is it always live?"
        ),
        "question_asked": 1,
        "realism": 5,
    },
    ("maya", "original", 2): {
        "text": (
            "Yes, Figma is the core of my workflow — brand systems, deliverables, client "
            "share, everything runs through it. Is yours a Figma-native setup or are you "
            "mixing tools in the stack?"
        ),
        "question_asked": 1,
        "realism": 4,
    },
    ("maya", "original", 3): {
        "text": (
            "Scattered feedback channels — email, Slack, a voice note, a photo of a "
            "printed page with pen marks. By revision round three you're playing archaeologist "
            "trying to figure out which version a comment even applies to. The design "
            "itself is fast; reconciling vague feedback from someone who says 'make it pop' "
            "across six different formats is where the hours and the margins disappear. "
            "What does your current client usually default to when they give feedback?"
        ),
        "question_asked": 1,
        "realism": 5,
    },
    ("maya", "original", 4): {
        "text": (
            "Sometimes — but I price fixed-rate projects at a premium and the discovery "
            "call has to completely nail the deliverable list or I'm eating scope creep. "
            "Are you working with clients who push back on hourly billing or prefer a "
            "project price upfront?"
        ),
        "question_asked": 1,
        "realism": 4,
    },
    ("maya", "original", 5): {
        "text": (
            "Yes, absolutely — if the client share view shows your logo, I can't use "
            "it for client-facing deliverables, that's a dealbreaker. Is white-label "
            "available at the standard plan tier or is it gated behind enterprise pricing?"
        ),
        "question_asked": 1,
        "realism": 3,
    },
    # --- THROTTLED CURIOUS ---
    ("maya", "throttled", 1): {
        "text": (
            "I send a white-labeled share link — client lands on a clean portal with my "
            "studio's branding, not the tool's logo. I pair it with a short Loom walkthrough "
            "so they know what they're looking at before they start leaving comments. "
            "Threaded feedback on the file beats email chains because every comment stays "
            "anchored to the actual deliverable. How many revision rounds do you typically "
            "build into a project?"
        ),
        "question_asked": 1,
        "realism": 5,
    },
    ("maya", "throttled", 2): {
        "text": (
            "Yes — Figma is the center of everything, from initial brand system "
            "to final export."
        ),
        "question_asked": 0,
        "realism": 5,
    },
    ("maya", "throttled", 3): {
        "text": (
            "Feedback scatter. Email, Slack, a screenshot with arrows drawn in the Notes app, "
            "sometimes a voice memo from someone driving. By revision round three you're "
            "doing archaeology to figure out which version a comment even belongs to. "
            "I can move fast on the design itself — it's reconciling 'make it more energetic' "
            "across six formats from a client who doesn't have design vocabulary where the "
            "hours and the margin evaporate. What does your worst feedback round look like?"
        ),
        "question_asked": 1,
        "realism": 5,
    },
    ("maya", "throttled", 4): {
        "text": (
            "Yes, but fixed-price only works when the discovery call completely locks "
            "the deliverable list — I price in a scope buffer and if the client starts "
            "adding rounds, the project goes hourly."
        ),
        "question_asked": 0,
        "realism": 5,
    },
    ("maya", "throttled", 5): {
        "text": (
            "Yes. If the client share view shows your logo instead of mine, I can't "
            "send that link — it's a dealbreaker, not a preference."
        ),
        "question_asked": 0,
        "realism": 5,
    },
}


# ---------------------------------------------------------------------------
# Scoring and metrics
# ---------------------------------------------------------------------------

ALL_RESPONSES = {**ALEX_RESPONSES, **MAYA_RESPONSES}


def score_all() -> list[dict]:
    results = []
    conditions = ["baseline", "original", "throttled"]
    persona_turns = {"alex": ALEX_TURNS, "maya": MAYA_TURNS}

    for persona_id, turns in persona_turns.items():
        for condition in conditions:
            for t in turns:
                turn_num = t["turn"]
                key = (persona_id, condition, turn_num)
                resp = ALL_RESPONSES[key]
                results.append(
                    {
                        "persona": persona_id,
                        "condition": condition,
                        "turn": turn_num,
                        "turn_type": t["type"],
                        "user_message": t["user"],
                        "response": resp["text"],
                        "question_asked": resp["question_asked"],
                        "realism": resp["realism"],
                    }
                )
    return results


def compute_summary(results: list[dict]) -> dict:
    conditions = ["baseline", "original", "throttled"]
    personas = ["alex", "maya"]
    summary: dict = {"by_condition": {}, "by_condition_and_persona": {}}

    for condition in conditions:
        rows = [r for r in results if r["condition"] == condition]
        open_rows = [r for r in rows if r["turn_type"] == "open"]
        closed_rows = [r for r in rows if r["turn_type"] == "closed"]

        def q_rate(subset: list[dict]) -> float:
            if not subset:
                return 0.0
            return sum(r["question_asked"] for r in subset) / len(subset)

        def avg_realism(subset: list[dict]) -> float:
            if not subset:
                return 0.0
            return sum(r["realism"] for r in subset) / len(subset)

        summary["by_condition"][condition] = {
            "open_q_rate": q_rate(open_rows),
            "closed_q_rate": q_rate(closed_rows),
            "overall_q_rate": q_rate(rows),
            "open_realism": avg_realism(open_rows),
            "closed_realism": avg_realism(closed_rows),
            "overall_realism": avg_realism(rows),
        }

    # Per-persona breakdown
    for persona in personas:
        summary["by_condition_and_persona"][persona] = {}
        for condition in conditions:
            rows = [
                r
                for r in results
                if r["persona"] == persona and r["condition"] == condition
            ]
            open_rows = [r for r in rows if r["turn_type"] == "open"]
            closed_rows = [r for r in rows if r["turn_type"] == "closed"]
            summary["by_condition_and_persona"][persona][condition] = {
                "open_q_rate": (
                    sum(r["question_asked"] for r in open_rows) / len(open_rows)
                    if open_rows
                    else 0.0
                ),
                "closed_q_rate": (
                    sum(r["question_asked"] for r in closed_rows) / len(closed_rows)
                    if closed_rows
                    else 0.0
                ),
            }

    # Success criterion check
    orig = summary["by_condition"]["original"]
    throttled = summary["by_condition"]["throttled"]
    closed_reduction = orig["closed_q_rate"] - throttled["closed_q_rate"]
    open_drop = orig["open_q_rate"] - throttled["open_q_rate"]
    summary["success_criterion"] = {
        "closed_q_rate_reduction_pp": round(closed_reduction * 100, 1),
        "open_q_rate_drop_pp": round(open_drop * 100, 1),
        "threshold_met_closed_reduction": closed_reduction >= 0.40,
        "threshold_met_open_preservation": open_drop <= 0.20,
        "signal": (
            "STRONG"
            if closed_reduction >= 0.40 and open_drop <= 0.20
            else "WEAK"
        ),
    }

    return summary


def main() -> None:
    results = score_all()
    summary = compute_summary(results)

    output_dir = Path(__file__).parent.parent / "output" / "experiments" / "exp-4.21b-curiosity-throttle"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {"results": results, "summary": summary}
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results written to {results_path}")
    print("\n=== Q-RATE SUMMARY ===")
    for condition, metrics in summary["by_condition"].items():
        print(
            f"  {condition:12s}  open={metrics['open_q_rate']:.0%}  "
            f"closed={metrics['closed_q_rate']:.0%}  "
            f"overall={metrics['overall_q_rate']:.0%}  "
            f"realism_closed={metrics['closed_realism']:.2f}"
        )
    sc = summary["success_criterion"]
    print(f"\nClosed-turn Q-rate reduction: {sc['closed_q_rate_reduction_pp']}pp")
    print(f"Open-turn Q-rate drop: {sc['open_q_rate_drop_pp']}pp")
    print(f"Signal: {sc['signal']}")


if __name__ == "__main__":
    main()
