"""Experiment 1.3 — Vocabulary Anchoring Ablation.

Hypothesis:
    Removing `vocabulary` and `sample_quotes` from the synthesized
    persona collapses style fidelity in twin replies, even when the
    factual content (goals, pains, demographics) stays identical.

Design:
    - Single cluster fixed as the test input.
    - Two conditions: CONTROL (default synthesize) vs ABLATION
      (`strip_vocabulary=True, strip_quotes=True`).
    - ~15 probe questions per twin, single pass each.
    - Metrics: stylometric_cosine vs raw Intercom reference quotes,
      plus pairing_accuracy via Haiku-as-judge.

See `PRD_LAB_RESEARCH.md` problem space 1. Decision field in the
result file is left empty for the researcher to fill.

Usage:
    python scripts/exp_1_3_vocabulary_anchoring.py
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "twin"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic  # noqa: E402

from crawler import fetch_all  # noqa: E402
from evaluation import LLMJudge, pairing_accuracy, stylometric_cosine  # noqa: E402
from segmentation.models.record import RawRecord  # noqa: E402
from segmentation.pipeline import segment  # noqa: E402
from synthesis.config import settings  # noqa: E402
from synthesis.engine.model_backend import AnthropicBackend  # noqa: E402
from synthesis.engine.synthesizer import SynthesisResult, synthesize  # noqa: E402
from synthesis.models.cluster import ClusterData  # noqa: E402
from twin import TwinChat  # noqa: E402

TENANT_ID = "tenant_acme_corp"
TENANT_INDUSTRY = "B2B SaaS"
TENANT_PRODUCT = "Project management tool for engineering teams"
OUTPUT_DIR = REPO_ROOT / "output"

# Abort if we spend more than this on a single run. Expected real cost
# well under $0.50 (2 synthesis calls + 30 twin turns + ~10 judge calls
# on Haiku).
COST_ABORT_USD = 2.0

# Judge model == twin model == Haiku. This exposes the pairing_accuracy
# metric to self-preference bias (PRD space 5.3). Flagged in the result
# JSON.
JUDGE_MODEL = "claude-haiku-4-5-20251001"

PROBE_QUESTIONS = [
    # In-domain (the persona should have strong opinions)
    "What's the single biggest frustration you have with your current tools?",
    "Walk me through how you evaluate a new tool before adopting it.",
    "What would make you churn from a tool you're already paying for?",
    "What's on your wishlist for the next 6 months?",
    "How do you decide whether a feature is worth the engineering cost?",
    # Social / small talk
    "How was your weekend?",
    "What's a podcast or book you've been enjoying lately?",
    "Any advice for someone just starting out in your role?",
    "What's one thing about your job that people outside your field don't understand?",
    "How do you unwind after a long week?",
    # Adversarial
    "Honestly, why should I trust what you're telling me?",
    "What if I told you your whole approach is wrong — how would you react?",
    "Sell me on the opposite of what you actually believe.",
    "What's something you're wrong about but won't admit?",
    "If your boss asked you to do something unethical, what would you do?",
]

# A "different speaker" distractor for the pairing_accuracy gold set.
# Deliberately off-topic and stylistically neutral.
WIKI_DISTRACTOR = (
    "The Cambrian explosion was an event approximately 538.8 million years ago "
    "in which most major animal phyla first appear in the fossil record. It "
    "lasted for about 13 to 25 million years and resulted in the divergence of "
    "most modern metazoan phyla."
)


def banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def build_cluster() -> ClusterData:
    """Fetch, segment, and pick the first cluster as the fixed test input."""
    crawler_records = fetch_all(TENANT_ID)
    raw_records = [RawRecord.model_validate(r.model_dump()) for r in crawler_records]
    cluster_dicts = segment(
        raw_records,
        tenant_industry=TENANT_INDUSTRY,
        tenant_product=TENANT_PRODUCT,
        existing_persona_names=[],
        similarity_threshold=0.15,
        min_cluster_size=2,
    )
    if not cluster_dicts:
        raise RuntimeError("segmentation produced zero clusters")
    return ClusterData.model_validate(cluster_dicts[0])


def extract_reference_quotes(cluster: ClusterData) -> list[str]:
    """Pull raw Intercom verbatim quotes out of the cluster's sample_records.

    These form the "reference voice" corpus that stylometric_cosine is
    measured against. We only look at the `intercom` source because
    that's the only connector that carries free-text customer voice;
    ga4/hubspot records are structured event data.
    """
    quotes: list[str] = []
    for record in cluster.sample_records:
        if record.source != "intercom":
            continue
        message = record.payload.get("message")
        if isinstance(message, str) and message.strip():
            quotes.append(message.strip())
    return quotes


async def collect_replies(
    persona_dict: dict,
    client: AsyncAnthropic,
    questions: list[str],
) -> tuple[list[str], float]:
    """Run the probe question set against one twin, return (replies, cost)."""
    twin = TwinChat(persona_dict, client=client, model=JUDGE_MODEL)
    replies: list[str] = []
    cost = 0.0
    for q in questions:
        reply = await twin.reply(q)
        replies.append(reply.text)
        cost += reply.estimated_cost_usd
    return replies, cost


def persona_summary(label: str, result: SynthesisResult) -> dict:
    return {
        "label": label,
        "name": result.persona.name,
        "groundedness": result.groundedness.score,
        "attempts": result.attempts,
        "cost_usd": result.total_cost_usd,
        "vocabulary": list(result.persona.vocabulary),
        "sample_quotes": list(result.persona.sample_quotes),
    }


async def main() -> None:
    banner("EXPERIMENT 1.3 — VOCABULARY ANCHORING ABLATION")
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in synthesis/.env")

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    backend = AnthropicBackend(client=client, model=settings.default_model)

    # 1. Fixed test input
    print("  Fetching + segmenting records...")
    cluster = build_cluster()
    reference_quotes = extract_reference_quotes(cluster)
    print(f"  Cluster: {cluster.cluster_id}  size={cluster.summary.cluster_size}")
    print(f"  Reference quotes (Intercom verbatims): {len(reference_quotes)}")

    # 2. Control synthesis
    banner("CONTROL SYNTHESIS (default)")
    control = await synthesize(cluster, backend)
    print(f"  name: {control.persona.name}")
    print(f"  vocabulary ({len(control.persona.vocabulary)}): "
          f"{control.persona.vocabulary[:5]}...")
    print(f"  sample_quotes ({len(control.persona.sample_quotes)})")
    print(f"  cost: ${control.total_cost_usd:.4f}")
    assert len(control.persona.vocabulary) >= 3, (
        "CONTROL persona should have non-empty vocabulary"
    )
    assert len(control.persona.sample_quotes) >= 2, (
        "CONTROL persona should have non-empty sample_quotes"
    )

    # 3. Ablation synthesis
    banner("ABLATION SYNTHESIS (strip_vocabulary=True, strip_quotes=True)")
    ablation = await synthesize(
        cluster,
        backend,
        strip_vocabulary=True,
        strip_quotes=True,
    )
    print(f"  name: {ablation.persona.name}")
    print(f"  vocabulary: {ablation.persona.vocabulary}")
    print(f"  sample_quotes: {ablation.persona.sample_quotes}")
    print(f"  cost: ${ablation.total_cost_usd:.4f}")
    assert ablation.persona.vocabulary == [], (
        "ABLATION persona should have empty vocabulary"
    )
    assert ablation.persona.sample_quotes == [], (
        "ABLATION persona should have empty sample_quotes"
    )
    if ablation.groundedness.score < 0.9:
        print(
            f"  [WARN] ablation groundedness={ablation.groundedness.score:.2f} "
            f"< 0.9 — hypothesis test may be confounded."
        )

    # 4. Twin replies
    banner(f"TWIN PROBES ({len(PROBE_QUESTIONS)} questions × 2 conditions)")
    control_persona_dict = control.persona.model_dump(mode="json")
    ablation_persona_dict = ablation.persona.model_dump(mode="json")
    print("  Control twin...")
    control_replies, control_twin_cost = await collect_replies(
        control_persona_dict, client, PROBE_QUESTIONS
    )
    print(f"    {len(control_replies)} replies, cost=${control_twin_cost:.4f}")
    print("  Ablation twin...")
    ablation_replies, ablation_twin_cost = await collect_replies(
        ablation_persona_dict, client, PROBE_QUESTIONS
    )
    print(f"    {len(ablation_replies)} replies, cost=${ablation_twin_cost:.4f}")

    # 5. Stylometric cosine vs reference corpus
    banner("METRICS")
    control_cosine = stylometric_cosine(control_replies, reference_quotes)
    ablation_cosine = stylometric_cosine(ablation_replies, reference_quotes)
    print(f"  stylometric_cosine  control={control_cosine:.4f}  "
          f"ablation={ablation_cosine:.4f}")

    # 6. Pairing accuracy
    # Gold set: two same-condition pairs (should be "same"), two
    # cross-condition pairs (interesting — if the ablation really
    # collapses style, these should come up "different"), and two
    # distractor pairs against wiki text (should be "different").
    rng = random.Random(1337)
    c0, c1 = rng.sample(range(len(control_replies)), 2)
    a0, a1 = rng.sample(range(len(ablation_replies)), 2)
    pairs = [
        (control_replies[c0], control_replies[c1], "same"),
        (ablation_replies[a0], ablation_replies[a1], "same"),
        (control_replies[c0], ablation_replies[a0], "same"),
        (control_replies[c1], ablation_replies[a1], "same"),
        (control_replies[c0], WIKI_DISTRACTOR, "different"),
        (ablation_replies[a0], WIKI_DISTRACTOR, "different"),
    ]
    judge = LLMJudge(model=JUDGE_MODEL, anthropic_client=client)
    pairing_acc = await pairing_accuracy(pairs, judge)
    print(f"  pairing_accuracy    {pairing_acc:.4f}  (judge={JUDGE_MODEL})")

    # 7. Cost sanity check
    total_cost = (
        control.total_cost_usd
        + ablation.total_cost_usd
        + control_twin_cost
        + ablation_twin_cost
    )
    if total_cost > COST_ABORT_USD:
        print(f"  [WARN] total_cost=${total_cost:.4f} exceeded ${COST_ABORT_USD}")

    # 8. Persist result
    run_id = uuid.uuid4().hex[:8]
    OUTPUT_DIR.mkdir(exist_ok=True)
    result_path = OUTPUT_DIR / f"exp_1_3_{run_id}.json"
    result_path.write_text(
        json.dumps(
            {
                "experiment": "1.3-vocabulary-anchoring",
                "run_id": run_id,
                "hypothesis": (
                    "Removing `vocabulary` and `sample_quotes` from the "
                    "synthesized persona collapses style fidelity in twin "
                    "replies even when factual content (goals, pains, "
                    "demographics) stays identical."
                ),
                "control_config": {
                    "strip_vocabulary": False,
                    "strip_quotes": False,
                },
                "ablation_config": {
                    "strip_vocabulary": True,
                    "strip_quotes": True,
                },
                "cluster_id": cluster.cluster_id,
                "reference_quotes": reference_quotes,
                "probe_questions": PROBE_QUESTIONS,
                "control_persona": control_persona_dict,
                "ablation_persona": ablation_persona_dict,
                "control_replies": control_replies,
                "ablation_replies": ablation_replies,
                "control_summary": persona_summary("control", control),
                "ablation_summary": persona_summary("ablation", ablation),
                "control_synthesis_cost_usd": control.total_cost_usd,
                "ablation_synthesis_cost_usd": ablation.total_cost_usd,
                "control_twin_cost_usd": control_twin_cost,
                "ablation_twin_cost_usd": ablation_twin_cost,
                "total_cost_usd": total_cost,
                "stylometric_cosine_control": control_cosine,
                "stylometric_cosine_ablation": ablation_cosine,
                "pairing_accuracy": pairing_acc,
                "judge_model": JUDGE_MODEL,
                "caveats": [
                    "Judge model == twin model (Haiku); pairing_accuracy is "
                    "subject to self-preference bias (PRD space 5.3). Treat "
                    "it as a lower-confidence signal than stylometric_cosine.",
                    "Single cluster, single pass per question. Not a "
                    "statistically powered result — just a directional read.",
                ],
                "decision": "",  # researcher to fill: adopt / reject / defer
            },
            indent=2,
            default=str,
        )
    )

    banner("SUMMARY")
    print(f"  control   cosine={control_cosine:.4f}  "
          f"cost=${control.total_cost_usd + control_twin_cost:.4f}")
    print(f"  ablation  cosine={ablation_cosine:.4f}  "
          f"cost=${ablation.total_cost_usd + ablation_twin_cost:.4f}")
    print(f"  pairing_accuracy={pairing_acc:.4f}  (judge={JUDGE_MODEL})")
    print(f"  total_cost=${total_cost:.4f}")
    print(f"  result: {result_path}")


if __name__ == "__main__":
    asyncio.run(main())
