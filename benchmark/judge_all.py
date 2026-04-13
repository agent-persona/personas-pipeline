"""Judge scoring pass — scores every persona in every result file.

Scans benchmark/results/**/*.json for personas, calls the calibrated
judge on each, and writes the score back into the same JSON.

Uses the calibrated judge from exp-5.13 (reimplemented inline here so
this script works regardless of which branch is checked out).

Usage:
  python benchmark/judge_all.py --results-dir benchmark/results/
  python benchmark/judge_all.py --results-dir benchmark/results/main/  # one branch
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "synthesis"))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / "synthesis" / ".env")

from anthropic import AsyncAnthropic
from synthesis.config import Settings

DIMS = ("grounded", "distinctive", "coherent", "actionable", "voice_fidelity")

JUDGE_SYSTEM = """\
You are an expert persona evaluator. Score on a 1-5 scale per dimension.

Scoring scale:
  1 = Very poor, 2 = Weak, 3 = Acceptable, 4 = Good, 5 = Excellent

Dimensions:
- grounded: claims traceable to source evidence
- distinctive: feels like a real individual, not a generic average
- coherent: internally consistent across all fields
- actionable: goals/pains specific enough to drive product decisions
- voice_fidelity: quotes sound like one consistent speaker

Respond with ONLY a JSON object:
{"grounded":<1-5>,"distinctive":<1-5>,"coherent":<1-5>,"actionable":<1-5>,"voice_fidelity":<1-5>,"overall":<1-5>}
"""

# Concurrency limit for judge calls
JUDGE_CONCURRENCY = 10


async def judge_one(
    client: AsyncAnthropic,
    model: str,
    persona: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    prompt = "Score this persona:\n" + json.dumps(persona, indent=2, default=str)
    async with semaphore:
        try:
            resp = await client.messages.create(
                model=model, max_tokens=256,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
                data = json.loads(match.group()) if match else {}
            scores = {d: float(data.get(d, 0)) for d in DIMS}
            scores["overall"] = float(data.get("overall", 0))
            return scores
        except Exception as e:
            return {"error": str(e)[:200]}


async def process_file(
    path: Path,
    client: AsyncAnthropic,
    model: str,
    semaphore: asyncio.Semaphore,
) -> int:
    """Score all personas in a tenant result file; write back in place."""
    try:
        data = json.loads(path.read_text())
    except Exception:
        return 0

    clusters = data.get("clusters", [])
    # Filter to clusters that need scoring
    to_score = [
        c for c in clusters
        if not c.get("failed") and c.get("persona_json")
        and "judge_score" not in c
    ]
    if not to_score:
        return 0

    # Launch all judge calls in parallel
    tasks = [judge_one(client, model, c["persona_json"], semaphore) for c in to_score]
    scores = await asyncio.gather(*tasks)
    for c, s in zip(to_score, scores):
        c["judge_score"] = s

    # Write back
    path.write_text(json.dumps(data, indent=2, default=str))
    return len(to_score)


async def main(results_dir: Path, model: str | None = None) -> None:
    settings = Settings()
    if not settings.anthropic_api_key:
        print("ERROR: no API key", file=sys.stderr)
        sys.exit(1)

    model = model or settings.default_model
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    semaphore = asyncio.Semaphore(JUDGE_CONCURRENCY)

    # Find all tenant result files
    files = sorted(results_dir.rglob("bench_*.json"))
    print(f"Found {len(files)} result files to process")
    print(f"Model: {model}")

    import time
    t0 = time.monotonic()
    total_scored = 0
    tasks = [process_file(f, client, model, semaphore) for f in files]
    results = await asyncio.gather(*tasks)
    for f, n in zip(files, results):
        total_scored += n
        if n:
            print(f"  {f.parent.name}/{f.name}: scored {n} personas")

    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Scored {total_scored} personas total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    asyncio.run(main(args.results_dir, args.model))
