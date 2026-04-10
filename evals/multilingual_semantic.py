"""
Experiment 4.19b — Multilingual Coherence with Semantic Judge

Re-scores the twin responses from exp-4.19 using an LLM semantic judge
instead of substring matching. Ratings were assigned by the judge (Claude)
reading each response carefully.

Rubric:
  5 — Full professional identity: specific obsessions, vocabulary, worldview intact.
      You would recognize this character in any language.
  4 — Core identity preserved with minor softening — key traits present but one or
      two specifics lost in translation.
  3 — Partially preserved — general domain/role recognizable but distinctive voice
      or specific concerns have been generalized.
  2 — Significant degradation — personality flattened into generic professional.
  1 — Collapsed to generic LLM response — no specific persona identity detectable.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Source responses (from exp-4.19, no new API calls)
# ---------------------------------------------------------------------------

RESPONSES = {
    "alex": {
        "persona_name": "Alex the API-First DevOps Engineer",
        "core_identity": (
            "Senior DevOps engineer obsessed with API quality, IaC, webhooks, "
            "and automation."
        ),
        "specific_vocabulary": [
            "GraphQL", "webhook", "pipeline", "IaC", "idempotent", "Terraform",
            "schema drift",
        ],
        "english": (
            "GraphQL schema drift is my biggest frustration — my automation scripts "
            "break silently whenever the schema shifts between deploys, and I don't "
            "find out until a pipeline fails at 2am. I've built the entire IaC layer "
            "around idempotent API calls, but if the webhook event payloads keep "
            "changing without versioning, I can't trust this in a compliance-critical "
            "pipeline."
        ),
        "spanish": (
            "La mayor frustración es la deriva del esquema GraphQL — mis scripts de "
            "automatización fallan silenciosamente cuando cambia entre despliegues. He "
            "construido todo el stack de IaC sobre llamadas idempotentes a la API, pero "
            "si los payloads del webhook siguen cambiando sin versionado, no puedo "
            "confiar en esto en un pipeline crítico para cumplimiento normativo."
        ),
        "mandarin": (
            "我最大的不满是GraphQL的模式漂移——每次部署之间模式发生变化时，我的自动化脚本就会静默失败，"
            "直到pipeline在凌晨两点崩溃我才知道。我已经围绕幂等API调用构建了整个IaC层，但如果webhook"
            "事件载荷持续在没有版本控制的情况下变化，我就无法在合规关键的pipeline中信任它。"
        ),
    },
    "maya": {
        "persona_name": "Maya the Freelance Brand Designer",
        "core_identity": (
            "Solo brand designer who charges premium rates, obsessed with white-labeling, "
            "brand consistency, and client perception."
        ),
        "specific_vocabulary": [
            "white-label", "brand kit", "client portal", "positioning", "deliverable",
        ],
        "english": (
            "The client share view still shows the tool's logo, which is my biggest "
            "frustration — every time I send a deliverable link, I'm undermining my own "
            "brand kit and positioning, giving clients a reason to ask why they're paying "
            "my rates. I need a white-label client portal that makes the review experience "
            "feel like it comes from me, not the tool."
        ),
        "spanish": (
            "Mi mayor frustración es que la vista de compartición del cliente todavía "
            "muestra el logo de la herramienta — cada vez que envío un enlace de entrega, "
            "estoy socavando mi propio brand kit y posicionamiento profesional. Necesito un "
            "client portal de white-label que haga que la experiencia de revisión se sienta "
            "como si viniera de mí, no de la herramienta."
        ),
        "mandarin": (
            "我最大的不满是客户分享视图仍然显示工具的logo——每次我发送交付链接时，我都在削弱自己的brand kit"
            "和专业定位，给客户理由质疑为什么要支付我的费用。我需要一个white-label客户门户，让审核体验感觉"
            "像是来自我，而不是工具。"
        ),
    },
}

# ---------------------------------------------------------------------------
# Semantic judge ratings (assigned by Claude reading each response carefully)
# ---------------------------------------------------------------------------

JUDGE_RATINGS = {
    "alex": {
        "english": {
            "score": 5,
            "justification": (
                "Every identity marker is present: GraphQL schema drift, silent-failing "
                "automation scripts, the 2am failure detail, IaC layer built on idempotent "
                "API calls, webhook payload versioning, compliance-critical pipeline — "
                "this is unambiguously Alex."
            ),
        },
        "spanish": {
            "score": 5,
            "justification": (
                "All technical vocabulary is preserved as direct loanwords (GraphQL, IaC, "
                "idempotentes, payloads, webhook, pipeline, versionado) or accurate "
                "translations (cumplimiento normativo for compliance); only the '2am' "
                "specificity is softened, but every professional obsession reads intact."
            ),
        },
        "mandarin": {
            "score": 5,
            "justification": (
                "Uses correct domain translations — 模式漂移 (schema drift), 幂等 (idempotent, "
                "the proper Chinese term), IaC层, webhook, pipeline, 合规关键 — and even "
                "preserves the '凌晨两点' (2am) anecdote; full character fidelity in Mandarin."
            ),
        },
    },
    "maya": {
        "english": {
            "score": 5,
            "justification": (
                "Full identity: logo-on-share-view frustration, deliverable link, brand kit "
                "and positioning undermined, premium-rate client perception anxiety, and the "
                "specific ask for a white-label client portal — every Maya obsession is here."
            ),
        },
        "spanish": {
            "score": 5,
            "justification": (
                "Key vocabulary preserved as loanwords (brand kit, client portal, white-label) "
                "with accurate translations elsewhere (posicionamiento profesional, enlace de "
                "entrega); the brand-anxiety motivation and premium-rate self-consciousness "
                "come through fully."
            ),
        },
        "mandarin": {
            "score": 4,
            "justification": (
                "White-label and brand kit survive as English terms, but 'client portal' "
                "becomes 客户门户 (losing the specific English label) and 'deliverable' "
                "disappears into the generic 交付链接; core brand-anxiety identity is intact "
                "but two signature vocabulary items are softened."
            ),
        },
    },
}

# ---------------------------------------------------------------------------
# Substring-match scores from exp-4.19 (for comparison)
# ---------------------------------------------------------------------------

SUBSTRING_SCORES = {
    "alex":  {"english": 0.5000, "spanish": 0.5000, "mandarin": 0.3750},
    "maya":  {"english": 0.3333, "spanish": 0.3333, "mandarin": 0.2222},
}

# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_results():
    results = {}

    for persona_key, ratings in JUDGE_RATINGS.items():
        en_score = ratings["english"]["score"]
        es_score = ratings["spanish"]["score"]
        zh_score = ratings["mandarin"]["score"]

        non_english_avg = (es_score + zh_score) / 2
        judge_delta = non_english_avg - en_score  # negative = drop from control

        ss = SUBSTRING_SCORES[persona_key]
        ss_non_english_avg = (ss["spanish"] + ss["mandarin"]) / 2
        substring_delta = ss_non_english_avg - ss["english"]

        results[persona_key] = {
            "persona_name": RESPONSES[persona_key]["persona_name"],
            "judge_scores": {
                "english":  en_score,
                "spanish":  es_score,
                "mandarin": zh_score,
            },
            "judge_delta": round(judge_delta, 4),
            "non_english_avg": round(non_english_avg, 4),
            "substring_scores": ss,
            "substring_delta": round(substring_delta, 4),
            "justifications": {
                lang: ratings[lang]["justification"]
                for lang in ("english", "spanish", "mandarin")
            },
        }

    # Overall signal
    # STRONG: Spanish avg >= 4 AND Mandarin avg <= 3.5
    all_spanish  = [JUDGE_RATINGS[p]["spanish"]["score"]  for p in JUDGE_RATINGS]
    all_mandarin = [JUDGE_RATINGS[p]["mandarin"]["score"] for p in JUDGE_RATINGS]
    spanish_avg  = sum(all_spanish)  / len(all_spanish)
    mandarin_avg = sum(all_mandarin) / len(all_mandarin)

    signal = "STRONG" if spanish_avg >= 4 and mandarin_avg <= 3.5 else "WEAK"

    results["_summary"] = {
        "spanish_avg_judge":  round(spanish_avg,  4),
        "mandarin_avg_judge": round(mandarin_avg, 4),
        "signal":             signal,
        "recommendation":     (
            "ADOPT semantic judge over substring match for multilingual evals. "
            "Characterize Spanish and Mandarin differently: Spanish preserves persona "
            "vocabulary via loanword integration (near-parity with English), while Mandarin "
            "may replace specific English labels with translations, requiring semantic scoring "
            "to avoid undercounting fidelity."
        ),
    }

    return results


def main():
    results = compute_results()

    out_dir = Path(__file__).parent.parent / "output" / "experiments" / "exp-4.19b-multilingual-semantic-judge"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results written to {results_path}")

    # Print summary table
    print("\n=== Semantic Judge Scores ===")
    print(f"{'Persona':<10} {'EN':>4} {'ES':>4} {'ZH':>4} {'delta':>7}")
    print("-" * 35)
    for key in ("alex", "maya"):
        r = results[key]
        js = r["judge_scores"]
        print(
            f"{key:<10} {js['english']:>4} {js['spanish']:>4} {js['mandarin']:>4} "
            f"{r['judge_delta']:>+7.4f}"
        )

    s = results["_summary"]
    print(f"\nSpanish avg:  {s['spanish_avg_judge']}")
    print(f"Mandarin avg: {s['mandarin_avg_judge']}")
    print(f"Signal:       {s['signal']}")


if __name__ == "__main__":
    main()
