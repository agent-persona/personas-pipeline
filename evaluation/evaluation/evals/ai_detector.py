"""
Adversarial detector for experiment 5.08.

Classifies personas as AI-generated or human-written using a scoring rubric.
Lower detectability score = more realistic persona.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Telltale signs of AI-generated personas (rubric dimensions).
# Every entry MUST have a corresponding check in score_persona — denominator is len(AI_TELLS).
AI_TELLS = [
    "overly_balanced",        # goals and pains identical count (exact match, ≥4 each)
    "generic_vocabulary",     # Words like "leverage", "synergy", "seamless", "robust"
    "template_structure",     # All four array fields (goals/pains/motivations/objections) same length ≥3
    "bland_quotes",           # Sample quotes contain marketing copy words
    "abstract_goals",         # Goals lack specific quantified markers (%, $, minutes, hours, days)
    "missing_specificity",    # Summary contains no digits
]

@dataclass
class DetectorResult:
    persona_name: str
    ai_tells_detected: list[str]
    ai_tell_count: int
    detectability_score: float   # 0.0 = undetectable (realistic), 1.0 = obvious AI
    verdict: str                  # "likely_ai" | "ambiguous" | "likely_human"
    reasoning: str

def score_persona(persona: dict[str, Any]) -> DetectorResult:
    """
    Score a persona for AI-generation detectability.
    Returns DetectorResult with detectability_score.
    """
    tells = []

    # Check overly balanced structure — exact equal length is the AI signal, not just close
    goals = persona.get("goals", [])
    pains = persona.get("pains", [])
    if len(goals) == len(pains) and len(goals) >= 4:
        tells.append("overly_balanced")

    # Check generic vocabulary
    vocab = " ".join(str(v) for v in persona.get("vocabulary", []))
    generic_words = ["leverage", "synergy", "seamless", "robust", "scalable",
                     "innovative", "streamline", "optimize", "holistic"]
    if any(w in vocab.lower() for w in generic_words):
        tells.append("generic_vocabulary")

    # Check template structure (all arrays same length)
    array_fields = [goals, pains, persona.get("motivations", []),
                    persona.get("objections", [])]
    lengths = [len(f) for f in array_fields if f]
    if len(set(lengths)) == 1 and lengths[0] >= 3:
        tells.append("template_structure")

    # Check bland quotes
    quotes = persona.get("sample_quotes", [])
    marketing_words = ["solution", "challenge", "journey", "empower", "transform"]
    bland_count = sum(1 for q in quotes
                     if any(w in str(q).lower() for w in marketing_words))
    if bland_count >= len(quotes) / 2 and quotes:
        tells.append("bland_quotes")

    # Check abstract goals — only genuine specificity signals, not common prepositions
    goal_texts = [str(g) for g in goals]
    specific_markers = ["%", "$", "minutes", "hours", "days", "per "]
    has_specific = any(m in " ".join(goal_texts) for m in specific_markers)
    if not has_specific and goals:
        tells.append("abstract_goals")

    # Check missing specificity in summary
    summary = persona.get("summary", "")
    if not any(c.isdigit() for c in summary):
        tells.append("missing_specificity")

    count = len(tells)
    score = count / len(AI_TELLS)

    if score >= 0.6:
        verdict = "likely_ai"
    elif score >= 0.3:
        verdict = "ambiguous"
    else:
        verdict = "likely_human"

    return DetectorResult(
        persona_name=persona.get("name", "unknown"),
        ai_tells_detected=tells,
        ai_tell_count=count,
        detectability_score=round(score, 3),
        verdict=verdict,
        reasoning=f"Detected {count}/{len(AI_TELLS)} AI tells: {', '.join(tells) if tells else 'none'}"
    )

def run_detector(output_dir: str = "output") -> dict:
    """Run detector on all persona files in output_dir."""
    results = []
    output_path = Path(output_dir)

    for persona_file in sorted(output_path.glob("persona_*.json")):
        data = json.loads(persona_file.read_text())
        persona = data.get("persona", data)
        result = score_persona(persona)
        results.append({
            "file": persona_file.name,
            "persona_name": result.persona_name,
            "ai_tells_detected": result.ai_tells_detected,
            "ai_tell_count": result.ai_tell_count,
            "detectability_score": result.detectability_score,
            "verdict": result.verdict,
            "reasoning": result.reasoning,
        })

    if not results:
        return {"error": "no persona files found", "results": []}

    mean_score = sum(r["detectability_score"] for r in results) / len(results)
    likely_ai = sum(1 for r in results if r["verdict"] == "likely_ai")

    return {
        "personas_evaluated": len(results),
        "mean_detectability_score": round(mean_score, 3),
        "likely_ai_count": likely_ai,
        "ambiguous_count": sum(1 for r in results if r["verdict"] == "ambiguous"),
        "likely_human_count": sum(1 for r in results if r["verdict"] == "likely_human"),
        "classifier_accuracy": round(likely_ai / len(results), 3),
        "realism_proxy": round(1.0 - mean_score, 3),
        "per_persona": results,
    }

if __name__ == "__main__":
    result = run_detector()
    print(json.dumps(result, indent=2))
