"""Experiment 3.06 — Branch comparison: main (baseline) vs exp-3.06.

Deterministic test that compares how each branch's groundedness logic
handles personas with varying evidence quality at different data densities.
No API key required — uses synthetic personas with controlled evidence gaps.

Usage:
    python evals/compare_branches.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from synthesis.engine.groundedness import (
    DEFAULT_THRESHOLD,
    GroundednessReport,
    check_groundedness,
    scaled_threshold,
)
from synthesis.models.cluster import (
    ClusterData,
    ClusterSummary,
    SampleRecord,
    TenantContext,
)
from synthesis.models.evidence import SourceEvidence
from synthesis.models.persona import (
    Demographics,
    Firmographics,
    JourneyStage,
    PersonaV1,
)


@dataclass
class ScenarioResult:
    name: str
    n_records: int
    n_evidence_fields: int  # how many of the 7 required fields have evidence
    total_required: int
    groundedness_score: float
    # Baseline (main) behavior: fixed 0.9 threshold
    baseline_passed: bool
    baseline_would_fail: bool  # True if synthesizer would raise SynthesisError
    # Exp-3.06 behavior: scaled threshold
    exp_threshold: float
    exp_passed: bool
    exp_degraded: bool


def build_cluster(n_records: int) -> ClusterData:
    """Build a cluster with n sample records."""
    records = [
        SampleRecord(
            record_id=f"r{i}",
            source="dense_fixture",
            payload={"event": f"action_{i}", "session_duration": 1000 + i * 100},
        )
        for i in range(n_records)
    ]
    return ClusterData(
        cluster_id=f"cluster_{n_records}rec",
        tenant=TenantContext(
            tenant_id="test_tenant",
            industry="B2B SaaS",
            product_description="Project management tool",
        ),
        summary=ClusterSummary(
            cluster_size=max(1, n_records // 3),
            top_behaviors=["api_setup", "webhook_config", "dashboard_view"],
        ),
        sample_records=records,
    )


def build_persona(
    record_ids: list[str],
    covered_fields: int,
) -> PersonaV1:
    """Build a persona where `covered_fields` out of 7 required items have evidence.

    The 7 required items are: 2 goals + 2 pains + 2 motivations + 1 objection.
    """
    # Always produce the full persona structure
    goals = ["Automate CI/CD pipelines", "Reduce context-switching across tools"]
    pains = ["Too many disconnected tools", "Slow deployment cycles"]
    motivations = ["Ship features faster", "Better developer experience"]
    objections = ["Migration cost is too high"]

    all_fields = [
        ("goals.0", goals[0]),
        ("goals.1", goals[1]),
        ("pains.0", pains[0]),
        ("pains.1", pains[1]),
        ("motivations.0", motivations[0]),
        ("motivations.1", motivations[1]),
        ("objections.0", objections[0]),
    ]

    evidence = []
    for i in range(min(covered_fields, len(all_fields))):
        field_path, claim = all_fields[i]
        # Reuse record IDs (cycle through available ones)
        rid = record_ids[i % len(record_ids)] if record_ids else "r0"
        evidence.append(
            SourceEvidence(
                claim=claim,
                record_ids=[rid],
                field_path=field_path,
                confidence=0.6 if len(record_ids) < 5 else 0.85,
            )
        )

    # Need at least 3 source_evidence entries for schema validity
    while len(evidence) < 3:
        rid = record_ids[0] if record_ids else "r0"
        evidence.append(
            SourceEvidence(
                claim="Filler evidence",
                record_ids=[rid],
                field_path=f"goals.{len(evidence)}",
                confidence=0.3,
            )
        )

    return PersonaV1(
        name="DevOps Dana",
        summary="A senior DevOps engineer focused on CI/CD automation.",
        demographics=Demographics(
            age_range="30-40",
            gender_distribution="mixed",
            location_signals=["US West Coast"],
        ),
        firmographics=Firmographics(
            company_size="50-200",
            industry="SaaS",
            role_titles=["DevOps Engineer"],
        ),
        goals=goals,
        pains=pains,
        motivations=motivations,
        objections=objections,
        channels=["GitHub", "Slack", "Stack Overflow"],
        vocabulary=["pipeline", "deploy", "infra-as-code", "CI/CD"],
        decision_triggers=["Free trial with API access"],
        sample_quotes=[
            "If I can't automate it, I don't want it.",
            "We need everything in one dashboard.",
        ],
        journey_stages=[
            JourneyStage(
                stage="evaluation",
                mindset="Comparing tools",
                key_actions=["API docs review", "trial signup"],
                content_preferences=["technical docs", "comparison guides"],
            ),
            JourneyStage(
                stage="adoption",
                mindset="Proving ROI to team",
                key_actions=["integration setup", "team onboarding"],
                content_preferences=["case studies", "migration guides"],
            ),
        ],
        source_evidence=evidence,
    )


def run_scenario(
    name: str,
    n_records: int,
    covered_fields: int,
) -> ScenarioResult:
    """Run a single scenario and evaluate under both baseline and exp-3.06 logic."""
    cluster = build_cluster(n_records)
    record_ids = cluster.all_record_ids
    persona = build_persona(record_ids, covered_fields)

    report = check_groundedness(persona, cluster)
    total_required = 7  # 2 goals + 2 pains + 2 motivations + 1 objection

    # Baseline (main) logic: fixed 0.9 threshold
    baseline_passed = report.score >= DEFAULT_THRESHOLD
    baseline_would_fail = not baseline_passed  # after retries, SynthesisError

    # Exp-3.06 logic: scaled threshold (already computed in report)
    exp_threshold = report.threshold
    exp_passed = report.passed
    exp_degraded = report.degraded

    return ScenarioResult(
        name=name,
        n_records=n_records,
        n_evidence_fields=covered_fields,
        total_required=total_required,
        groundedness_score=report.score,
        baseline_passed=baseline_passed,
        baseline_would_fail=baseline_would_fail,
        exp_threshold=exp_threshold,
        exp_passed=exp_passed,
        exp_degraded=exp_degraded,
    )


def main() -> None:
    # Scenarios: (name, n_records, covered_fields_out_of_7)
    scenarios = [
        # Dense data — full evidence
        ("dense-full", 12, 7),
        ("dense-partial", 12, 5),
        ("dense-thin", 12, 4),
        # Medium data — moderate evidence
        ("medium-full", 7, 7),
        ("medium-partial", 7, 5),
        ("medium-thin", 7, 4),
        # Sparse data — limited evidence
        ("sparse-full", 3, 7),
        ("sparse-partial", 3, 5),
        ("sparse-thin", 3, 3),
        ("sparse-minimal", 3, 2),
        # Very sparse — barely any data
        ("very-sparse-full", 2, 7),
        ("very-sparse-partial", 2, 4),
        ("very-sparse-thin", 2, 2),
        # Edge: single record
        ("single-record", 1, 3),
        ("single-minimal", 1, 1),
    ]

    results = [run_scenario(name, n, cov) for name, n, cov in scenarios]

    # Print comparison table
    print()
    print("=" * 100)
    print("EXPERIMENT 3.06 — BRANCH COMPARISON: main (baseline) vs exp-3.06")
    print("=" * 100)
    print()
    print(
        f"{'Scenario':<20} {'Recs':>4} {'Evid':>4} {'Score':>6} "
        f"{'| MAIN':>8} {'Result':>10} "
        f"{'| EXP':>6} {'Thresh':>6} {'Result':>10}"
    )
    print("-" * 100)

    baseline_pass = 0
    baseline_fail = 0
    exp_pass = 0
    exp_degraded = 0
    exp_fail = 0

    for r in results:
        if r.baseline_passed:
            b_result = "PASS"
            baseline_pass += 1
        else:
            b_result = "FAIL"
            baseline_fail += 1

        if r.exp_passed and not r.exp_degraded:
            e_result = "PASS"
            exp_pass += 1
        elif r.exp_passed and r.exp_degraded:
            e_result = "DEGRADED"
            exp_degraded += 1
        else:
            e_result = "FAIL"
            exp_fail += 1

        print(
            f"{r.name:<20} {r.n_records:>4} {r.n_evidence_fields:>3}/7 {r.groundedness_score:>6.2f} "
            f"{'|':>2} {DEFAULT_THRESHOLD:>5.1f}  {b_result:>10} "
            f"{'|':>2} {r.exp_threshold:>5.1f}  {e_result:>10}"
        )

    print("-" * 100)
    print()
    print("SUMMARY")
    print(f"  Total scenarios: {len(results)}")
    print()
    print(f"  MAIN (baseline, fixed 0.9 threshold):")
    print(f"    Pass: {baseline_pass}    Fail: {baseline_fail}")
    print(f"    Persona generation rate: {baseline_pass}/{len(results)} ({100*baseline_pass/len(results):.0f}%)")
    print()
    print(f"  EXP-3.06 (scaled threshold):")
    print(f"    Pass: {exp_pass}    Degraded: {exp_degraded}    Fail: {exp_fail}")
    print(f"    Persona generation rate: {exp_pass + exp_degraded}/{len(results)} ({100*(exp_pass + exp_degraded)/len(results):.0f}%)")
    print()

    rescued = (exp_pass + exp_degraded) - baseline_pass
    print(f"  RESCUED BY EXP-3.06: {rescued} scenarios that would have failed on main")
    print(f"    now produce a persona (albeit some degraded).")
    print()

    # Detail the rescued scenarios
    rescued_scenarios = [
        r for r in results if not r.baseline_passed and r.exp_passed
    ]
    if rescued_scenarios:
        print("  Rescued scenarios:")
        for r in rescued_scenarios:
            label = "DEGRADED" if r.exp_degraded else "FULL"
            print(
                f"    - {r.name}: {r.n_records} records, {r.n_evidence_fields}/7 evidence, "
                f"score={r.groundedness_score:.2f} -> {label} (threshold={r.exp_threshold:.1f})"
            )
    print()


if __name__ == "__main__":
    main()
