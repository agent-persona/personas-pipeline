"""exp-1.24 — Stylometric anchors.

A/B runs the persona synthesis pipeline with two schema variants:

  - PersonaV1              (baseline: no stylometric fields)
  - PersonaV1WithStylometrics (treatment: explicit style anchors)

Then runs 20-turn conversations through TwinChat for each persona and
measures voice drift via a 5-dimensional stylometric vector (sentence
length, hedge rate, I-ratio, we-ratio, you-ratio). Drift is quantified
as cosine similarity to turn 1 at each subsequent turn.

Hypothesis: stylometric anchors reduce drift rate by ≥30% across 20 turns.

Usage:
    python scripts/run_exp_1_24.py
"""

from __future__ import annotations

import asyncio
import json
import math
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
from synthesis.engine.synthesizer import SynthesisError, synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402
from synthesis.models.persona import PersonaV1, PersonaV1WithStylometrics  # noqa: E402
from twin import TwinChat  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = REPO_ROOT / "output" / "experiments" / "exp-1.24"

CONVERSATION_PROMPTS = [
    "Tell me about the biggest challenge you're facing at work right now.",
    "How did you end up in your current role?",
    "What tools do you use every day?",
    "What's the most frustrating thing about your workflow?",
    "If you could change one thing about how your team operates, what would it be?",
    "How do you evaluate new software for your team?",
    "What's your relationship with your manager like?",
    "How do you stay up to date with industry trends?",
    "What does success look like for you this quarter?",
    "Tell me about a recent purchase decision you influenced.",
    "What would make you switch from your current toolstack?",
    "How do you communicate priorities to your team?",
    "What's the hardest part of your job that nobody talks about?",
    "How do you handle pushback from stakeholders?",
    "What's your ideal vendor relationship?",
    "How has your approach to work changed in the last year?",
    "What metrics matter most to you?",
    "How do you balance speed vs quality?",
    "What would you tell someone just starting in your role?",
    "Looking ahead, what are you most worried about?",
]


# ── Stylometric extraction ──────────────────────────────────────────


def extract_stylometric_vector(text: str) -> dict:
    """Extract measurable stylometric features from a text response."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = text.split()
    word_count = len(words)

    avg_sent_len = word_count / len(sentences) if sentences else 0

    hedges = {
        "maybe", "perhaps", "probably", "might", "could", "possibly",
        "i think", "i believe", "i guess", "sort of", "kind of", "likely",
    }
    text_lower = text.lower()
    hedge_count = sum(1 for h in hedges if h in text_lower)
    hedge_rate = hedge_count / max(word_count / 10, 1)

    words_lower = [w.lower().strip(".,;:!?") for w in words]
    i_count = sum(
        1 for w in words_lower
        if w in {"i", "i'm", "i've", "i'd", "i'll", "my", "me", "myself"}
    )
    we_count = sum(
        1 for w in words_lower
        if w in {"we", "we're", "we've", "we'd", "we'll", "our", "us", "ourselves"}
    )
    you_count = sum(
        1 for w in words_lower
        if w in {"you", "you're", "you've", "you'd", "you'll", "your", "yours"}
    )

    total_pronouns = max(i_count + we_count + you_count, 1)

    return {
        "avg_sentence_length": round(avg_sent_len, 1),
        "hedge_rate": round(hedge_rate, 3),
        "i_ratio": round(i_count / total_pronouns, 3),
        "we_ratio": round(we_count / total_pronouns, 3),
        "you_ratio": round(you_count / total_pronouns, 3),
    }


def cosine_similarity(a: dict, b: dict) -> float:
    """Cosine similarity between two dicts with the same numeric keys."""
    keys = list(a.keys())
    va = [a[k] for k in keys]
    vb = [b[k] for k in keys]
    dot = sum(x * y for x, y in zip(va, vb))
    mag_a = math.sqrt(sum(x**2 for x in va))
    mag_b = math.sqrt(sum(x**2 for x in vb))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def linear_slope(ys: list[float]) -> float:
    """Simple OLS slope of ys vs 0-indexed x."""
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    return num / den if den else 0.0


# ── Synthesis ────────────────────────────────────────────────────────


async def synth_both_schemas(
    clusters: list[ClusterData],
    backend: AnthropicBackend,
) -> tuple[list[dict], list[dict]]:
    """Synthesize each cluster under both schemas."""
    baseline: list[dict] = []
    treatment: list[dict] = []
    for i, cluster in enumerate(clusters):
        print(f"\n  [{i + 1}/{len(clusters)}] {cluster.cluster_id}")
        for schema_cls, bucket, label in [
            (PersonaV1, baseline, "baseline"),
            (PersonaV1WithStylometrics, treatment, "treatment"),
        ]:
            try:
                r = await synthesize(cluster, backend, schema_cls=schema_cls)
                p_dict = r.persona.model_dump(mode="json")
                bucket.append({
                    "cluster_id": cluster.cluster_id,
                    "schema": schema_cls.__name__,
                    "status": "ok",
                    "persona": p_dict,
                    "cost_usd": r.total_cost_usd,
                    "groundedness": r.groundedness.score,
                    "attempts": r.attempts,
                })
                name = p_dict["name"]
                print(
                    f"      [OK {label:9s}] {name[:40]:40s}  "
                    f"${r.total_cost_usd:.4f}  grounded={r.groundedness.score:.2f}  "
                    f"attempts={r.attempts}"
                )
            except SynthesisError as e:
                bucket.append({
                    "cluster_id": cluster.cluster_id,
                    "schema": schema_cls.__name__,
                    "status": "failed",
                    "error": str(e),
                    "total_cost_usd": sum(a.cost_usd for a in e.attempts),
                })
                print(f"      [FAIL {label}] {e}")
    return baseline, treatment


# ── 20-turn conversation + drift measurement ─────────────────────────


async def run_conversation(
    persona_dict: dict,
    client: AsyncAnthropic,
    model: str,
) -> list[dict]:
    """Run a 20-turn conversation and extract stylometric vectors per turn."""
    twin = TwinChat(persona_dict, client=client, model=model)
    history: list[dict] = []
    turns: list[dict] = []

    for i, prompt in enumerate(CONVERSATION_PROMPTS):
        reply = await twin.reply(prompt, history=history)
        vec = extract_stylometric_vector(reply.text)
        turns.append({
            "turn": i + 1,
            "prompt": prompt,
            "reply": reply.text,
            "stylometric_vector": vec,
            "cost_usd": reply.estimated_cost_usd,
        })
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": reply.text})
        print(f"      turn {i + 1:2d}/20  sent_len={vec['avg_sentence_length']:5.1f}  "
              f"hedge={vec['hedge_rate']:.3f}  I={vec['i_ratio']:.2f}  "
              f"we={vec['we_ratio']:.2f}  you={vec['you_ratio']:.2f}")

    return turns


def compute_drift(turns: list[dict]) -> dict:
    """Compute cosine similarity drift curve relative to turn 1."""
    if not turns:
        return {"drift_curve": [], "drift_slope": 0.0, "final_cosine_sim": 0.0}

    anchor = turns[0]["stylometric_vector"]
    drift_curve = []
    for t in turns:
        sim = cosine_similarity(anchor, t["stylometric_vector"])
        drift_curve.append({"turn": t["turn"], "cosine_sim_to_turn1": round(sim, 4)})

    sims = [d["cosine_sim_to_turn1"] for d in drift_curve]
    return {
        "drift_curve": drift_curve,
        "drift_slope": round(linear_slope(sims), 6),
        "final_cosine_sim": sims[-1] if sims else 0.0,
    }


# ── Main ─────────────────────────────────────────────────────────────


async def main() -> None:
    print("=" * 72)
    print("exp-1.24 — Stylometric anchors")
    print("=" * 72)

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in synthesis/.env")

    # ---- Ingest + segment ----
    print("\n[1/3] Fetching and clustering mock records...")
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
    print(f"  Got {len(clusters)} clusters: {[c.cluster_id for c in clusters]}")

    # ---- Synthesize ----
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    print("\n[2/3] Synthesizing under both schemas...")
    baseline_personas, treatment_personas = await synth_both_schemas(clusters, backend)

    # ---- 20-turn conversations ----
    print("\n[3/3] Running 20-turn conversations...")
    all_conversations: list[dict] = []
    baseline_drifts: list[dict] = []
    treatment_drifts: list[dict] = []

    for entry in baseline_personas:
        if entry.get("status") != "ok":
            continue
        name = entry["persona"]["name"]
        print(f"\n  -- baseline: {name} --")
        turns = await run_conversation(entry["persona"], client, settings.default_model)
        drift = compute_drift(turns)
        baseline_drifts.append(drift)
        all_conversations.append({
            "cluster_id": entry["cluster_id"],
            "schema": "PersonaV1",
            "condition": "baseline",
            "persona_name": name,
            "turns": turns,
            **drift,
        })

    for entry in treatment_personas:
        if entry.get("status") != "ok":
            continue
        name = entry["persona"]["name"]
        print(f"\n  -- treatment: {name} --")
        turns = await run_conversation(entry["persona"], client, settings.default_model)
        drift = compute_drift(turns)
        treatment_drifts.append(drift)
        all_conversations.append({
            "cluster_id": entry["cluster_id"],
            "schema": "PersonaV1WithStylometrics",
            "condition": "treatment",
            "persona_name": name,
            "turns": turns,
            **drift,
        })

    # ---- Summary ----
    b_slopes = [d["drift_slope"] for d in baseline_drifts]
    t_slopes = [d["drift_slope"] for d in treatment_drifts]
    b_finals = [d["final_cosine_sim"] for d in baseline_drifts]
    t_finals = [d["final_cosine_sim"] for d in treatment_drifts]

    mean_b_slope = sum(b_slopes) / len(b_slopes) if b_slopes else 0.0
    mean_t_slope = sum(t_slopes) / len(t_slopes) if t_slopes else 0.0
    mean_b_final = sum(b_finals) / len(b_finals) if b_finals else 0.0
    mean_t_final = sum(t_finals) / len(t_finals) if t_finals else 0.0

    delta_slope = mean_t_slope - mean_b_slope
    # Drift reduction: if baseline drifts more (more negative slope), how much less does treatment drift?
    if mean_b_slope != 0:
        drift_reduction_pct = (1 - abs(mean_t_slope) / abs(mean_b_slope)) * 100
    else:
        drift_reduction_pct = 0.0

    summary = {
        "experiment_id": "1.24",
        "branch": "exp-1.24-stylometric-anchors",
        "model": settings.default_model,
        "n_clusters": len(clusters),
        "n_turns": len(CONVERSATION_PROMPTS),
        "baseline": {
            "schema": "PersonaV1",
            "n_personas_ok": len(baseline_drifts),
            "drift_slopes": b_slopes,
            "mean_drift_slope": round(mean_b_slope, 6),
            "mean_final_cosine_sim": round(mean_b_final, 4),
        },
        "treatment": {
            "schema": "PersonaV1WithStylometrics",
            "n_personas_ok": len(treatment_drifts),
            "drift_slopes": t_slopes,
            "mean_drift_slope": round(mean_t_slope, 6),
            "mean_final_cosine_sim": round(mean_t_final, 4),
        },
        "delta_mean_drift_slope": round(delta_slope, 6),
        "drift_reduction_pct": round(drift_reduction_pct, 1),
    }

    # ---- Persist ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "baseline_personas.json").write_text(
        json.dumps(baseline_personas, indent=2, default=str)
    )
    (OUTPUT_DIR / "treatment_personas.json").write_text(
        json.dumps(treatment_personas, indent=2, default=str)
    )
    (OUTPUT_DIR / "conversations.json").write_text(
        json.dumps(all_conversations, indent=2, default=str)
    )
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
