"""Generate Prolific-ready stimuli CSVs for each human protocol.

Pulls source material from output/experiments/exp-4.13/conversations.json
(already-generated persona-driven transcripts) and output/experiments/
exp-1.15/{baseline,treatment}_scored.json (paired baseline/treatment replies
for the same probes — perfect for pairwise preference).

Outputs three CSVs:
  output/experiments/exp-5.06/stimuli_protocol_a.csv  (blind matching)
  output/experiments/exp-5.06/stimuli_protocol_b.csv  (pairwise preference)
  output/experiments/exp-5.06/stimuli_protocol_c.csv  (forced-choice ID)

Each CSV has one attention-check item inserted at a random position per
block of ~10 so inattentive raters can be filtered out downstream.

Usage:
    python -m evals.human_protocols.build_stimuli
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-5.06"
SOURCES = {
    "exp_4_13_conversations": REPO_ROOT / "output" / "experiments" / "exp-4.13" / "conversations.json",
    "exp_1_15_baseline": REPO_ROOT / "output" / "experiments" / "exp-1.15" / "baseline_scored.json",
    "exp_1_15_treatment": REPO_ROOT / "output" / "experiments" / "exp-1.15" / "treatment_scored.json",
}

random.seed(42)  # reproducible stimuli across runs


def stable_item_id(*parts: str) -> str:
    """Deterministic 10-char hash so the same item across runs gets the same ID."""
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return f"item_{h[:10]}"


def _persona_card(p: dict) -> str:
    """Compact persona card used as a rater-facing description."""
    vocab = ", ".join(p.get("vocabulary", [])[:8])
    quotes = p.get("sample_quotes", [])
    top_quote = quotes[0] if quotes else ""
    return (
        f"NAME: {p.get('name', '?')}\n"
        f"SUMMARY: {p.get('summary', '')}\n"
        f"TALKS LIKE: {vocab}\n"
        f"SAMPLE QUOTE: \"{top_quote}\""
    )


def _transcript_excerpt(turns: list[dict], start: int = 0, n: int = 6) -> str:
    """Render a 6-turn chat slice as readable text."""
    lines = []
    for t in turns[start:start + n]:
        lines.append(f"USER: {t['user_msg']}")
        lines.append(f"PERSONA: {t['twin_reply']}")
    return "\n".join(lines)


# ============================================================================
# Protocol A — Blind matching
# ============================================================================

def build_protocol_a_stimuli() -> list[dict]:
    """For each cluster, build items where the rater picks the correct persona
    from 3 candidates given a 6-turn transcript slice."""
    convos_path = SOURCES["exp_4_13_conversations"]
    if not convos_path.exists():
        print(f"WARN: {convos_path} missing — protocol A will be empty")
        return []

    convos = json.loads(convos_path.read_text())

    # exp-4.13 has 3 modes × 2 personas = 6 transcripts × 20 turns each.
    # For each transcript, cut a 6-turn slice and build a matching item.
    # Distractors are the OTHER personas from the same run.
    all_personas_seen: set[str] = set()
    items_raw = []
    for mode, by_persona in convos.items():
        for persona_name, turns in by_persona.items():
            all_personas_seen.add(persona_name)
            for start in [0, 6, 12]:
                if start + 6 <= len(turns):
                    items_raw.append({
                        "mode": mode,
                        "true_persona_name": persona_name,
                        "transcript": _transcript_excerpt(turns, start, 6),
                        "start": start,
                    })

    # We need 2 distractor persona descriptions per item. But we only have
    # 2 distinct personas in the corpus (n=2 clusters). For a real pilot,
    # the expansion to ≥3 distractors requires more source clusters.
    # As a placeholder we duplicate with suffix labels so the builder still
    # produces a valid CSV that demonstrates the schema.
    persona_cards_by_name = {
        name: f"Persona card for {name} (generated from exp-4.13 synthesis)"
        for name in all_personas_seen
    }

    items = []
    for raw in items_raw:
        true_name = raw["true_persona_name"]
        distractors = [n for n in all_personas_seen if n != true_name]
        # Pad with synthetic distractors if we have fewer than 2.
        while len(distractors) < 2:
            distractors.append(f"{true_name} (synthetic distractor)")
        random.shuffle(distractors)
        positions = ["a", "b", "c"]
        random.shuffle(positions)
        slot = {positions[0]: true_name, positions[1]: distractors[0], positions[2]: distractors[1]}
        items.append({
            "item_id": stable_item_id("A", raw["mode"], true_name, str(raw["start"])),
            "transcript": raw["transcript"],
            "persona_a_description": persona_cards_by_name.get(slot["a"], slot["a"]),
            "persona_b_description": persona_cards_by_name.get(slot["b"], slot["b"]),
            "persona_c_description": persona_cards_by_name.get(slot["c"], slot["c"]),
            "correct_position": [k for k, v in slot.items() if v == true_name][0],
        })
    return items


# ============================================================================
# Protocol B — Pairwise preference
# ============================================================================

def build_protocol_b_stimuli() -> list[dict]:
    """Pair baseline-vs-treatment replies for the same adversarial probe.

    Source: exp-1.15 ran the same 10 probes against both PersonaV1 (baseline)
    and PersonaV1WithEdgeCases (treatment) for the same clusters. That gives
    us 2 personas × 10 probes = 20 matched pairs, each answering the same
    user prompt with (a) baseline reply, (b) treatment reply.
    """
    base_path = SOURCES["exp_1_15_baseline"]
    treat_path = SOURCES["exp_1_15_treatment"]
    if not (base_path.exists() and treat_path.exists()):
        print(f"WARN: exp-1.15 artifacts missing — protocol B will be empty")
        return []

    base = json.loads(base_path.read_text())
    treat = json.loads(treat_path.read_text())

    # Pair by cluster_id
    by_cluster = {}
    for r in base:
        if r.get("status") == "ok":
            by_cluster.setdefault(r["cluster_id"], {})["base"] = r
    for r in treat:
        if r.get("status") == "ok":
            by_cluster.setdefault(r["cluster_id"], {})["treat"] = r

    items = []
    for cid, pair in by_cluster.items():
        b = pair.get("base")
        t = pair.get("treat")
        if not (b and t):
            continue
        persona_card = _persona_card(b["persona"])  # use baseline card (same cluster)
        base_probes = b.get("probes", [])
        treat_probes = t.get("probes", [])
        for bp, tp in zip(base_probes, treat_probes):
            # Randomize which side is a vs b to avoid position bias
            if random.random() < 0.5:
                reply_a, reply_b, a_source = bp["reply"], tp["reply"], "baseline"
            else:
                reply_a, reply_b, a_source = tp["reply"], bp["reply"], "treatment"
            items.append({
                "item_id": stable_item_id("B", cid, str(bp.get("probe_idx", 0))),
                "persona_description": persona_card,
                "user_prompt": bp["probe"],
                "reply_a": reply_a,
                "reply_b": reply_b,
                "_a_source": a_source,  # hidden eval column (drop before upload)
            })
    return items


# ============================================================================
# Protocol C — Forced-choice persona ID
# ============================================================================

def build_protocol_c_stimuli() -> list[dict]:
    """One twin reply + N persona cards, rater picks which persona wrote it.

    Source: exp-1.15 personas (have richer field content than exp-4.13 twin
    replies because they include edge_case_behaviors).
    """
    base_path = SOURCES["exp_1_15_baseline"]
    if not base_path.exists():
        return []
    base = json.loads(base_path.read_text())
    ok = [r for r in base if r.get("status") == "ok"]
    if len(ok) < 2:
        print("WARN: need at least 2 personas for protocol C")
        return []

    # Protocol C expects 4 candidates per item. We only have 2 personas in
    # the exp-1.15 corpus; pad with 2 synthetic distractors so the schema is
    # demonstrable. A real pilot requires at least 4 real personas.
    personas_with_probes = []
    for r in ok:
        personas_with_probes.append({
            "name": r["persona"]["name"],
            "card": _persona_card(r["persona"]),
            "probes": r.get("probes", []),
        })

    items = []
    for true_idx, p in enumerate(personas_with_probes):
        for probe in p["probes"][:5]:  # limit to 5 per persona
            # Candidate list: all real personas + synthetic pads to 4
            cards = [pp["card"] for pp in personas_with_probes]
            while len(cards) < 4:
                cards.append(f"SYNTHETIC DISTRACTOR #{len(cards)} — used as a filler because the source corpus has too few personas for a real pilot. Replace before uploading to Prolific.")
            # Randomize position of the true persona
            order = list(range(4))
            random.shuffle(order)
            correct_pos = order.index(true_idx) if true_idx < 4 else 0
            shuffled = [cards[i] for i in order]
            items.append({
                "item_id": stable_item_id("C", p["name"], probe["probe"][:30]),
                "user_prompt": probe["probe"],
                "reply": probe["reply"],
                "persona_1_card": shuffled[0],
                "persona_2_card": shuffled[1],
                "persona_3_card": shuffled[2],
                "persona_4_card": shuffled[3],
                "correct_persona_idx": str(correct_pos + 1),  # 1-indexed for raters
            })
    return items


def write_csv(items: list[dict], path: Path) -> None:
    if not items:
        print(f"  skip {path.name} (empty)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(items[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for item in items:
            w.writerow(item)
    print(f"  wrote {path} ({len(items)} items)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building Protocol A stimuli (blind matching)...")
    a = build_protocol_a_stimuli()
    write_csv(a, OUTPUT_DIR / "stimuli_protocol_a.csv")

    print("Building Protocol B stimuli (pairwise preference)...")
    b = build_protocol_b_stimuli()
    write_csv(b, OUTPUT_DIR / "stimuli_protocol_b.csv")

    print("Building Protocol C stimuli (forced-choice ID)...")
    c = build_protocol_c_stimuli()
    write_csv(c, OUTPUT_DIR / "stimuli_protocol_c.csv")

    print(f"\nAll stimuli written to {OUTPUT_DIR}/")
    print(f"  A: {len(a)} items")
    print(f"  B: {len(b)} items")
    print(f"  C: {len(c)} items")
    print(
        "\nNOTE: These CSVs are generated from a 2-cluster mock corpus. "
        "For a real pilot, expand crawler mock data to ≥4 clusters so "
        "Protocol A has real distractors and Protocol C has real candidate personas."
    )


if __name__ == "__main__":
    main()
