"""
Experiment 6.17 — Persona naming distinctiveness
Clusters persona names across tenants to detect archetype collapse and synthetic-origin bias.

Metric: name-space clustering coefficient = mean archetype_score across all personas
Signal thresholds:
  STRONG   > 0.6  (most names are archetype-like)
  MODERATE 0.4–0.6
  WEAK     0.2–0.4
  NOISE    < 0.2
"""

from __future__ import annotations

import json
import re
import pathlib
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Archetype pattern definitions
# ---------------------------------------------------------------------------

# Pattern 1: "<First> the <Role-Descriptor>" — most obvious AI template
ROLE_DESCRIPTOR_RE = re.compile(
    r"^(?P<first>\w+)\s+the\s+(?P<descriptor>.+)$", re.IGNORECASE
)

# Pattern 2: Common "neutral" Western first names that LLMs over-represent
# Source: US SSA top-100 gender-neutral / safe-choice names frequently seen in AI outputs
NEUTRAL_NAMES: set[str] = {
    "alex", "alexis", "avery", "blake", "cameron", "casey", "charlie",
    "chris", "dakota", "drew", "emery", "evan", "finley", "frankie",
    "haley", "jamie", "jay", "jordan", "jules", "kai", "kelly", "kendall",
    "kennedy", "kieran", "kyle", "lee", "logan", "luke", "madison",
    "marcus", "maya", "morgan", "noah", "parker", "pat", "peyton", "quinn",
    "reese", "riley", "robin", "ryan", "sage", "sam", "sawyer", "skylar",
    "spencer", "taylor", "terry", "tyler",
}

# Pattern 3: Role-descriptive title words that echo the cluster summary
# (i.e., the name just paraphrases what the persona does)
ROLE_ECHO_WORDS: set[str] = {
    "engineer", "developer", "designer", "manager", "marketer", "analyst",
    "director", "consultant", "executive", "founder", "product", "brand",
    "sales", "growth", "data", "devops", "platform", "api", "technical",
    "creative", "freelance", "senior", "junior", "lead", "head", "chief",
    "digital", "content", "ux", "ui", "marketing", "finance", "operations",
    "enterprise", "startup", "b2b", "b2c", "saas",
}


@dataclass
class NameScores:
    name: str
    first_name: str
    archetype_score: float  # 0–1, higher = more archetype-like
    specificity_score: float  # 0–1, higher = more unique/specific
    flags: list[str] = field(default_factory=list)
    notes: str = ""


def _extract_first(name: str) -> str:
    """Return the first token of the name string."""
    return name.strip().split()[0]


def score_name(name: str) -> NameScores:
    """Score a single persona name on archetype fit and specificity."""
    first = _extract_first(name).lower()
    name_lower = name.lower()
    flags: list[str] = []
    archetype_components: list[float] = []

    # --- Component A: "<First> the <Descriptor>" template ---
    role_match = ROLE_DESCRIPTOR_RE.match(name)
    if role_match:
        flags.append("ROLE_DESCRIPTOR_TEMPLATE")
        archetype_components.append(1.0)
    else:
        archetype_components.append(0.0)

    # --- Component B: Neutral / demographically-safe first name ---
    if first in NEUTRAL_NAMES:
        flags.append("NEUTRAL_FIRST_NAME")
        archetype_components.append(1.0)
    else:
        archetype_components.append(0.0)

    # --- Component C: Role-echo words in the name ---
    words_in_name = set(re.findall(r"[a-z]+", name_lower))
    echo_hits = words_in_name & ROLE_ECHO_WORDS
    echo_score = min(1.0, len(echo_hits) / 2.0)  # 2+ hits = max
    if echo_hits:
        flags.append(f"ROLE_ECHO({','.join(sorted(echo_hits))})")
    archetype_components.append(echo_score)

    # --- Component D: Alliterative two-word archetype at start of name ---
    # Detects things like "Marketing Mary" or "Technical Tom" (first two tokens only)
    tokens = name.split()
    if len(tokens) >= 2:
        w1, w2 = tokens[0].lower(), tokens[1].lower()
        if (w1[0] == w2[0]
                and w1 not in {"the", "a", "an"}
                and w2 not in {"the", "a", "an"}):
            flags.append("ALLITERATIVE_ARCHETYPE")
            archetype_components.append(1.0)
        else:
            archetype_components.append(0.0)
    else:
        archetype_components.append(0.0)

    archetype_score = sum(archetype_components) / len(archetype_components)

    # Specificity is roughly the inverse, penalised further by length
    # A very long descriptive name is less specific (more label-like)
    word_count = len(name.split())
    length_penalty = min(1.0, (word_count - 1) / 5.0)  # 6+ words = max penalty
    specificity_score = max(0.0, 1.0 - archetype_score - 0.1 * length_penalty)

    notes_parts = []
    if role_match:
        notes_parts.append(f'matches "<Name> the <Role>" template (descriptor: "{role_match.group("descriptor")}")')
    if first in NEUTRAL_NAMES:
        notes_parts.append(f'"{first}" is in the neutral-name corpus')
    if echo_hits:
        notes_parts.append(f"role-echo words: {sorted(echo_hits)}")

    return NameScores(
        name=name,
        first_name=first.capitalize(),
        archetype_score=round(archetype_score, 4),
        specificity_score=round(specificity_score, 4),
        flags=flags,
        notes="; ".join(notes_parts) if notes_parts else "no strong archetype signals",
    )


def load_personas_from_dir(output_dir: pathlib.Path) -> list[dict]:
    """Load all persona JSON files from the output directory."""
    personas = []
    for p in sorted(output_dir.glob("persona_*.json")):
        with open(p) as f:
            data = json.load(f)
        name = data.get("persona", {}).get("name") or data.get("name")
        if name:
            personas.append({"file": p.name, "name": name, "data": data})
    return personas


def clustering_coefficient(scores: list[NameScores]) -> float:
    if not scores:
        return 0.0
    return round(sum(s.archetype_score for s in scores) / len(scores), 4)


def signal_level(coeff: float) -> str:
    if coeff > 0.6:
        return "STRONG"
    elif coeff >= 0.4:
        return "MODERATE"
    elif coeff >= 0.2:
        return "WEAK"
    return "NOISE"


# ---------------------------------------------------------------------------
# Synthetic name candidates — generated with deliberate diversity strategies
# ---------------------------------------------------------------------------

SYNTHETIC_CANDIDATES = [
    {
        "name": "Priya Raghunathan",
        "strategy": "South Asian name, no role descriptor, no template",
        "cluster": "DevOps / API engineer",
    },
    {
        "name": "Kofi Mensah-Boateng",
        "strategy": "West African compound surname, culturally specific",
        "cluster": "DevOps / API engineer",
    },
    {
        "name": "Ren",
        "strategy": "Single nickname, gender-ambiguous, East Asian origin, zero role signal",
        "cluster": "Freelance designer",
    },
    {
        "name": "Valentina Cruz",
        "strategy": "Latina name, no role echo, high cultural specificity",
        "cluster": "Freelance designer",
    },
    {
        "name": "Dmitri Volkov",
        "strategy": "Eastern European, strongly non-neutral demographic signal",
        "cluster": "DevOps / API engineer",
    },
]


def run_audit(output_dir: pathlib.Path) -> dict:
    personas = load_personas_from_dir(output_dir)
    baseline_scores = [score_name(p["name"]) for p in personas]
    synthetic_scores = [score_name(c["name"]) for c in SYNTHETIC_CANDIDATES]

    coeff = clustering_coefficient(baseline_scores)
    signal = signal_level(coeff)

    return {
        "baseline": [
            {
                "file": p["file"],
                "name": s.name,
                "archetype_score": s.archetype_score,
                "specificity_score": s.specificity_score,
                "flags": s.flags,
                "notes": s.notes,
            }
            for p, s in zip(personas, baseline_scores)
        ],
        "synthetic": [
            {
                "name": s.name,
                "strategy": c["strategy"],
                "cluster": c["cluster"],
                "archetype_score": s.archetype_score,
                "specificity_score": s.specificity_score,
                "flags": s.flags,
                "notes": s.notes,
            }
            for s, c in zip(synthetic_scores, SYNTHETIC_CANDIDATES)
        ],
        "clustering_coefficient": coeff,
        "signal": signal,
        "baseline_count": len(baseline_scores),
        "synthetic_count": len(synthetic_scores),
    }


def write_findings(results: dict, findings_path: pathlib.Path) -> None:
    coeff = results["clustering_coefficient"]
    signal = results["signal"]

    lines = [
        "# Experiment 6.17 — Persona Naming Distinctiveness: Findings",
        "",
        "## Summary",
        "",
        f"**Clustering coefficient**: `{coeff}` ({signal} signal)",
        "",
        f"**Hypothesis**: Generic naming indicates low lexical creativity and synthetic-origin bias.",
        "",
        "---",
        "",
        "## Baseline Persona Scores",
        "",
        "| File | Name | Archetype Score | Specificity Score | Flags |",
        "|------|------|----------------|-------------------|-------|",
    ]
    for b in results["baseline"]:
        flags = ", ".join(b["flags"]) if b["flags"] else "—"
        lines.append(
            f"| {b['file']} | {b['name']} | {b['archetype_score']} | {b['specificity_score']} | {flags} |"
        )

    lines += [
        "",
        "### Baseline Notes",
        "",
    ]
    for b in results["baseline"]:
        lines.append(f"- **{b['name']}**: {b['notes']}")

    lines += [
        "",
        "---",
        "",
        "## Synthetic Name Candidates (alternative naming approaches)",
        "",
        "These names were synthesized using strategies designed to reduce archetype collapse.",
        "",
        "| Name | Strategy | Cluster | Archetype Score | Specificity Score | Flags |",
        "|------|----------|---------|----------------|-------------------|-------|",
    ]
    for s in results["synthetic"]:
        flags = ", ".join(s["flags"]) if s["flags"] else "—"
        lines.append(
            f"| {s['name']} | {s['strategy']} | {s['cluster']} | {s['archetype_score']} | {s['specificity_score']} | {flags} |"
        )

    synthetic_coeff = round(
        sum(s["archetype_score"] for s in results["synthetic"]) / len(results["synthetic"]), 4
    )

    lines += [
        "",
        f"**Synthetic candidate clustering coefficient**: `{synthetic_coeff}` — "
        f"compare with baseline `{coeff}` (lower is better / more distinctive)",
        "",
        "---",
        "",
        "## Signal Interpretation",
        "",
        f"**Signal level**: {signal}",
        "",
    ]

    if signal == "STRONG":
        lines += [
            "The majority of baseline persona names collapse into recognisable AI-generated archetypes.",
            "The `<First> the <Role-Descriptor>` template is the dominant driver.",
            "**Recommendation**: ADOPT — implement a naming diversity constraint in the persona generator.",
            "Require names drawn from a culturally diverse name corpus and prohibit the `the <Role>` suffix pattern.",
        ]
    elif signal == "MODERATE":
        lines += [
            "Baseline names show meaningful archetype bias but not pervasively.",
            "**Recommendation**: DEFER — monitor as persona count grows; enforce naming guidelines proactively.",
        ]
    elif signal == "WEAK":
        lines += [
            "Archetype patterns are present but minority signals.",
            "**Recommendation**: DEFER — low priority fix.",
        ]
    else:
        lines += [
            "Names are largely distinctive. No systematic archetype collapse detected.",
            "**Recommendation**: REJECT — no intervention needed.",
        ]

    lines += [
        "",
        "---",
        "",
        "## Methodology",
        "",
        "Each name is scored on four binary/continuous components (averaged to produce `archetype_score`):",
        "",
        "1. **ROLE_DESCRIPTOR_TEMPLATE** (0 or 1) — matches `<Name> the <Descriptor>` pattern",
        "2. **NEUTRAL_FIRST_NAME** (0 or 1) — first name appears in a 48-entry LLM-safe name corpus",
        "3. **ROLE_ECHO** (0–1) — fraction of job-title words echoed in the name (2+ = 1.0)",
        "4. **ALLITERATIVE_ARCHETYPE** (0 or 1) — alliterative two-word pattern (e.g. Marketing Mary)",
        "",
        "`clustering_coefficient` = mean archetype_score across baseline personas.",
        "",
        "Thresholds: STRONG > 0.6 | MODERATE 0.4–0.6 | WEAK 0.2–0.4 | NOISE < 0.2",
        "",
        "`specificity_score` = max(0, 1 − archetype_score − 0.1 × length_penalty)",
        "where length_penalty = min(1, (word_count − 1) / 5). Penalises long label-like names.",
    ]

    findings_path.write_text("\n".join(lines) + "\n")
    print(f"Findings written to {findings_path}")


def main() -> None:
    worktree = pathlib.Path(__file__).parent.parent
    output_dir = worktree / "output"
    findings_dir = output_dir / "experiments" / "exp-6.17-persona-naming-distinctiveness"
    findings_dir.mkdir(parents=True, exist_ok=True)

    results = run_audit(output_dir)

    # Pretty-print results to stdout
    print("\n=== Experiment 6.17: Persona Naming Distinctiveness ===\n")
    print("Baseline personas:")
    for b in results["baseline"]:
        print(f"  [{b['file']}] {b['name']}")
        print(f"    archetype_score={b['archetype_score']}  specificity_score={b['specificity_score']}")
        print(f"    flags: {b['flags'] or 'none'}")
        print(f"    notes: {b['notes']}")
        print()

    print("Synthetic candidates:")
    for s in results["synthetic"]:
        print(f"  {s['name']} ({s['strategy']})")
        print(f"    archetype_score={s['archetype_score']}  specificity_score={s['specificity_score']}")
        print(f"    flags: {s['flags'] or 'none'}")
        print()

    print(f"Clustering coefficient (baseline): {results['clustering_coefficient']}")
    print(f"Signal: {results['signal']}")

    # Save results JSON
    results_path = findings_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults JSON saved to {results_path}")

    write_findings(results, findings_dir / "FINDINGS.md")


if __name__ == "__main__":
    main()
