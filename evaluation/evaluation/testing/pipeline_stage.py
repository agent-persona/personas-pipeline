"""Orchestration integration: run the tiered scorer suite against freshly
synthesized personas inside the pipeline's DAG.

Keeps the orchestration layer dependency-light: this module is the only place
that imports the scorer registry, so stages that don't want the testing phase
never pay for it.
"""

from __future__ import annotations

from typing import Any

from evaluation.testing.adapter import persona_v1_to_testing, source_context_from_records
from evaluation.testing.schemas import EvalResult
from evaluation.testing.source_context import SourceContext
from evaluation.testing.suite_runner import SuiteRunner


def _default_scorers(tier_filter: int | None):
    """Lazy import so that the scorer tree (which pulls scipy/numpy/etc.) is
    only imported when the testing stage is actually used."""
    from evaluation.testing.scorers.all import get_all_scorers

    scorers = get_all_scorers()
    if tier_filter is None:
        return scorers
    return [s for s in scorers if s.tier <= tier_filter]


def run_testing_stage(
    personas: list[dict[str, Any]],
    *,
    tier_filter: int | None = 1,
    source_records_by_cluster: dict[str, list[Any]] | None = None,
) -> list[dict[str, Any]]:
    """Score each synthesized persona with the tiered suite.

    ``personas`` is the output of ``stage_synthesize`` — a list of dicts each
    with ``cluster_id`` and ``persona`` (a PersonaV1 dump).

    Mutates each entry in-place, adding a ``testing`` dict:

        {
          "tier_filter": 1,
          "passed": bool,       # all non-skipped scorers passed
          "results": [ ... ],   # list[EvalResult.model_dump()]
        }

    ``tier_filter=1`` is the default — Tier 1 (structural) is cheap and is
    what the gating layer cares about. Pass ``None`` to run the full suite,
    or an integer to run tiers 1..N.

    ``source_records_by_cluster`` lets callers pass the per-cluster raw
    records the persona was synthesized from, so scorers that need retrieval
    context (Tier 2+) can run. When absent, scorers get an empty context —
    structural scorers still work; semantic scorers degrade gracefully.
    """
    if not personas:
        return personas

    scorers = _default_scorers(tier_filter)
    runner = SuiteRunner(scorers)

    for entry in personas:
        persona_v1 = entry.get("persona") or {}
        cluster_id = entry.get("cluster_id", "unknown")

        persona = persona_v1_to_testing(persona_v1)
        if source_records_by_cluster and cluster_id in source_records_by_cluster:
            ctx = source_context_from_records(cluster_id, source_records_by_cluster[cluster_id])
        else:
            ctx = SourceContext(id=cluster_id, text="")

        results: list[EvalResult] = runner.run(persona, ctx, tier_filter=tier_filter)
        passed = all(r.passed for r in results if not r.details.get("skipped"))
        entry["testing"] = {
            "tier_filter": tier_filter,
            "passed": passed,
            "results": [r.model_dump() for r in results],
        }

    return personas


def summarize_testing(personas: list[dict[str, Any]]) -> str:
    """One-line human summary per persona, for orchestration logs."""
    lines: list[str] = []
    for entry in personas:
        t = entry.get("testing")
        if not t:
            continue
        results = t["results"]
        n_fail = sum(1 for r in results if not r["passed"] and not r["details"].get("skipped"))
        n_total = sum(1 for r in results if not r["details"].get("skipped"))
        name = (entry.get("persona") or {}).get("name", entry.get("cluster_id", "?"))
        verdict = "PASS" if t["passed"] else "FAIL"
        lines.append(f"  [{verdict}] {name}: {n_total - n_fail}/{n_total} scorers passed")
    return "\n".join(lines)
