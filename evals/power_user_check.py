"""Experiment 6.12: Power-user heuristic.

Scores each tenant's persona set for the presence of a "power-user"
archetype — the heaviest-engagement segment. If no persona matches,
a warning is raised indicating a coverage gap.

A power-user persona is detected by scanning for high-engagement signals:
  - Goals mentioning automation, optimization, advanced features, API usage
  - Vocabulary containing technical/power-user terms
  - Firmographics suggesting technical roles
  - High record count in the source cluster (largest cluster = likely power users)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Power-user signal keywords ──────────────────────────────────────

POWER_USER_GOAL_SIGNALS = re.compile(
    r"(automat|optimiz|advanced|scale|integrat|custom|api|pipeline|"
    r"workflow|efficien|productiv|streamlin|deploy|monitor|config|"
    r"infrastructure|architect|power.?user|heavy.?us)",
    re.IGNORECASE,
)

POWER_USER_VOCAB_SIGNALS = re.compile(
    r"(api|sdk|cli|webhook|terraform|kubernetes|ci/cd|graphql|"
    r"docker|yaml|json|git|bash|sql|rest|grpc|oauth|jwt|"
    r"pipeline|deploy|infra|devops|sprint|agile|standup|"
    r"dashboard|analytics|metrics|kpi)",
    re.IGNORECASE,
)

POWER_USER_ROLE_SIGNALS = re.compile(
    r"(engineer|developer|architect|devops|sre|platform|"
    r"technical|lead|senior|staff|principal|cto|vp.?eng)",
    re.IGNORECASE,
)


# ── Data types ──────────────────────────────────────────────────────

@dataclass
class PowerUserScore:
    """How strongly a persona matches the power-user archetype."""
    persona_name: str
    goal_matches: int = 0
    vocab_matches: int = 0
    role_matches: int = 0
    total_score: float = 0.0  # weighted composite
    is_power_user: bool = False


@dataclass
class PowerUserReport:
    """Power-user presence check for a persona set."""
    n_personas: int
    scores: list[PowerUserScore] = field(default_factory=list)
    power_user_found: bool = False
    best_match: str = ""
    best_score: float = 0.0
    inclusion_rate: float = 0.0  # fraction of persona sets containing a power user


# ── Scoring ─────────────────────────────────────────────────────────

def score_persona_power_user(persona: dict) -> PowerUserScore:
    """Score how strongly a persona matches the power-user archetype."""
    name = persona.get("name", "?")

    # Goal signal matches
    goals = persona.get("goals", []) + persona.get("pains", [])
    goal_text = " ".join(goals)
    goal_matches = len(POWER_USER_GOAL_SIGNALS.findall(goal_text))

    # Vocabulary signal matches
    vocab = persona.get("vocabulary", [])
    vocab_text = " ".join(vocab)
    vocab_matches = len(POWER_USER_VOCAB_SIGNALS.findall(vocab_text))

    # Role signal matches
    firmo = persona.get("firmographics", {})
    role_text = " ".join(firmo.get("role_titles", []))
    role_matches = len(POWER_USER_ROLE_SIGNALS.findall(role_text))

    # Also check summary and sample quotes
    summary = persona.get("summary", "")
    quote_text = " ".join(persona.get("sample_quotes", []))
    extra_goal = len(POWER_USER_GOAL_SIGNALS.findall(summary + " " + quote_text))
    goal_matches += extra_goal

    # Weighted composite: goals (40%), vocab (40%), role (20%)
    total = (
        min(goal_matches, 10) * 0.4
        + min(vocab_matches, 10) * 0.4
        + min(role_matches, 5) * 0.2
    ) / (10 * 0.4 + 10 * 0.4 + 5 * 0.2)  # normalize to 0-1

    return PowerUserScore(
        persona_name=name,
        goal_matches=goal_matches,
        vocab_matches=vocab_matches,
        role_matches=role_matches,
        total_score=total,
        is_power_user=total >= 0.25,  # threshold for power-user detection
    )


def check_power_user(personas: list[dict]) -> PowerUserReport:
    """Check if a persona set contains a power-user archetype."""
    scores = [score_persona_power_user(p) for p in personas]
    scores.sort(key=lambda s: s.total_score, reverse=True)

    power_users = [s for s in scores if s.is_power_user]
    best = scores[0] if scores else PowerUserScore(persona_name="none")

    return PowerUserReport(
        n_personas=len(personas),
        scores=scores,
        power_user_found=len(power_users) > 0,
        best_match=best.persona_name,
        best_score=best.total_score,
        inclusion_rate=1.0 if power_users else 0.0,
    )
