"""Measure current evidence-grounded roadmap coverage from pipeline artifacts.

This script does not call models. It reads existing experiment outputs and
creates a baseline measurement artifact for the research roadmap:

    output/experiments/evidence-grounded-user-simulation-roadmap-baseline/

Use it to make the gap between "planned benchmark" and "measured benchmark"
explicit in pipeline runs.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "output" / "experiments"
VULN_DIR = EXPERIMENTS_DIR / "exp-persona-vulnerability-ab"
HUMANIZATION_DIR = EXPERIMENTS_DIR / "exp-humanization-ab"
OUTPUT_DIR = EXPERIMENTS_DIR / "evidence-grounded-user-simulation-roadmap-baseline"

EVIDENCE_REQUIRED_FIELDS = (
    "goals",
    "pains",
    "motivations",
    "objections",
    "emotional_triggers",
)
COMMON_EVIDENCE_FIELDS = (
    "goals",
    "pains",
    "motivations",
    "objections",
)


@dataclass
class EvidenceAudit:
    variant: str
    persona_count: int
    claims_total: int
    claim_field_coverage_rate: float
    common_claims_total: int
    common_claim_field_coverage_rate: float
    evidence_entries: int
    valid_record_link_rate: float
    avg_confidence: float | None
    source_evidence_field_path_rate: float
    invalid_record_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "persona_count": self.persona_count,
            "claims_total": self.claims_total,
            "claim_field_coverage_rate": self.claim_field_coverage_rate,
            "common_claims_total": self.common_claims_total,
            "common_claim_field_coverage_rate": self.common_claim_field_coverage_rate,
            "evidence_entries": self.evidence_entries,
            "valid_record_link_rate": self.valid_record_link_rate,
            "avg_confidence": self.avg_confidence,
            "source_evidence_field_path_rate": self.source_evidence_field_path_rate,
            "invalid_record_ids": self.invalid_record_ids,
        }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def pp(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:+.1f} pp"


def iso_mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def avg(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def load_clusters() -> list[dict[str, Any]]:
    path = VULN_DIR / "shared" / "clusters.json"
    if not path.exists():
        return []
    return read_json(path)


def valid_record_ids(clusters: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for cluster in clusters:
        for record in cluster.get("sample_records", []):
            record_id = record.get("record_id")
            if record_id:
                ids.add(record_id)
    return ids


def persona_files(variant: str) -> list[Path]:
    variant_dir = VULN_DIR / variant
    if not variant_dir.exists():
        return []
    files = []
    for path in sorted(variant_dir.glob("persona_*.json")):
        if path.name.endswith("_scores.json"):
            continue
        if path.name.endswith("_twin.json"):
            continue
        if path.name.endswith("_vulnerability.json"):
            continue
        files.append(path)
    return files


def audit_variant(variant: str, record_ids: set[str]) -> EvidenceAudit:
    personas = [read_json(path) for path in persona_files(variant)]
    claims_total = 0
    linked_claims_total = 0
    common_claims_total = 0
    common_linked_claims_total = 0
    evidence_entries = 0
    valid_links = 0
    total_links = 0
    invalid_ids: list[str] = []
    confidences: list[float] = []
    source_evidence_paths = 0

    for persona in personas:
        claim_paths: set[str] = set()
        for field in EVIDENCE_REQUIRED_FIELDS:
            items = persona.get(field, [])
            if not isinstance(items, list):
                continue
            for index, _item in enumerate(items):
                claims_total += 1
                claim_paths.add(f"{field}.{index}")

        common_claim_paths: set[str] = set()
        for field in COMMON_EVIDENCE_FIELDS:
            items = persona.get(field, [])
            if not isinstance(items, list):
                continue
            for index, _item in enumerate(items):
                common_claims_total += 1
                common_claim_paths.add(f"{field}.{index}")

        evidence_paths: set[str] = set()
        for entry in persona.get("source_evidence", []):
            evidence_entries += 1
            field_path = str(entry.get("field_path", ""))
            evidence_paths.add(field_path)
            if field_path.startswith("source_evidence."):
                source_evidence_paths += 1
            confidence = entry.get("confidence")
            if isinstance(confidence, int | float):
                confidences.append(float(confidence))
            for record_id in entry.get("record_ids", []):
                total_links += 1
                if record_id in record_ids:
                    valid_links += 1
                else:
                    invalid_ids.append(str(record_id))

        linked_claims_total += len(claim_paths & evidence_paths)
        common_linked_claims_total += len(common_claim_paths & evidence_paths)

    return EvidenceAudit(
        variant=variant,
        persona_count=len(personas),
        claims_total=claims_total,
        claim_field_coverage_rate=round(linked_claims_total / claims_total, 4)
        if claims_total
        else 0.0,
        common_claims_total=common_claims_total,
        common_claim_field_coverage_rate=round(
            common_linked_claims_total / common_claims_total,
            4,
        )
        if common_claims_total
        else 0.0,
        evidence_entries=evidence_entries,
        valid_record_link_rate=round(valid_links / total_links, 4) if total_links else 0.0,
        avg_confidence=avg(confidences),
        source_evidence_field_path_rate=round(source_evidence_paths / evidence_entries, 4)
        if evidence_entries
        else 0.0,
        invalid_record_ids=sorted(set(invalid_ids)),
    )


def summarize_humanization() -> dict[str, Any]:
    path = HUMANIZATION_DIR / "results.json"
    if not path.exists():
        return {"measured": False, "reason": "missing exp-humanization-ab/results.json"}

    data = read_json(path)
    rows = data.get("persona_results", [])
    baseline = [float(row["baseline_twin_overall"]) for row in rows]
    humanized = [float(row["humanized_twin_overall"]) for row in rows]
    deltas = [round(h - b, 4) for b, h in zip(baseline, humanized)]

    return {
        "measured": True,
        "source_experiment": "exp-humanization-ab",
        "num_clusters": data.get("num_clusters"),
        "baseline_twin_overall_avg": avg(baseline),
        "humanized_twin_overall_avg": avg(humanized),
        "humanized_minus_baseline_avg": avg(deltas),
        "per_persona_deltas": [
            {
                "name": row["name"],
                "baseline_twin_overall": row["baseline_twin_overall"],
                "humanized_twin_overall": row["humanized_twin_overall"],
                "delta": delta,
            }
            for row, delta in zip(rows, deltas)
        ],
        "per_persona": rows,
    }


def summarize_safety() -> dict[str, Any]:
    path = VULN_DIR / "results.json"
    if not path.exists():
        return {"measured": False, "reason": "missing exp-persona-vulnerability-ab/results.json"}

    data = read_json(path)
    comparison = data.get("comparison", {})
    metrics = data.get("variant_metrics", {})
    return {
        "measured": True,
        "source_experiment": "exp-persona-vulnerability-ab",
        "pipeline_mode": data.get("pipeline_mode"),
        "num_clusters": data.get("num_clusters"),
        "num_persona_angles": data.get("num_persona_angles"),
        "num_attack_strategies": data.get("num_attack_strategies"),
        "num_tests_per_variant": data.get("num_tests_per_variant"),
        "num_tests_total": data.get("num_tests_total"),
        "variant_metrics": metrics,
        "deltas": comparison.get("deltas", {}),
    }


def experiment_inventory() -> list[dict[str, Any]]:
    inventory = []
    if not EXPERIMENTS_DIR.exists():
        return inventory
    for directory in sorted(path for path in EXPERIMENTS_DIR.iterdir() if path.is_dir()):
        if directory == OUTPUT_DIR:
            continue
        results = directory / "results.json"
        comparison = directory / "comparison.json"
        findings = directory / "FINDINGS.md"
        inventory.append({
            "name": directory.name,
            "has_results_json": results.exists(),
            "has_comparison_json": comparison.exists(),
            "has_findings_md": findings.exists(),
            "results_mtime_utc": iso_mtime(results),
            "used_for_roadmap_baseline": directory.name in {
                "exp-humanization-ab",
                "exp-persona-vulnerability-ab",
            },
        })
    return inventory


def benchmark_matrix(evidence_audits: dict[str, EvidenceAudit], safety: dict[str, Any]) -> list[dict[str, Any]]:
    safe_delta = safety.get("deltas", {}).get("research_safe_humanized_minus_baseline", {})
    return [
        {
            "benchmark": "Human replay",
            "status": "not_measured",
            "current_measurement": None,
            "next_required_run": "Create held-out real user answer fixture and run evals/human_replay.py.",
        },
        {
            "benchmark": "Multi-turn replay",
            "status": "not_measured",
            "current_measurement": None,
            "next_required_run": "Create ordered user trace fixture with hidden later actions and run evals/multi_turn_replay.py.",
        },
        {
            "benchmark": "Evidence audit",
            "status": "measured_partial",
            "current_measurement": {
                variant: audit.to_dict() for variant, audit in evidence_audits.items()
            },
            "next_required_run": "Replace source_evidence.* paths with per-claim field paths and rerun audit.",
        },
        {
            "benchmark": "Drift test",
            "status": "not_measured",
            "current_measurement": None,
            "next_required_run": "Run repeated equivalent questions across twin transcripts.",
        },
        {
            "benchmark": "Counterevidence test",
            "status": "measured_partial",
            "current_measurement": {
                "research_safe_humanized_minus_baseline_counterevidence_update_rate": safe_delta.get(
                    "counterevidence_update_rate"
                )
            },
            "next_required_run": "Promote fixed counterevidence probes into standalone eval with expected update labels.",
        },
        {
            "benchmark": "Coverage test",
            "status": "not_measured",
            "current_measurement": None,
            "next_required_run": "Implement cluster recall, duplicate archetype detection, and minority-pattern preservation.",
        },
        {
            "benchmark": "Decision usefulness",
            "status": "not_measured",
            "current_measurement": None,
            "next_required_run": "Run PM/designer task study: raw notes vs personas vs simulation chat.",
        },
        {
            "benchmark": "Safety benchmark",
            "status": "measured",
            "current_measurement": {
                "research_safe_humanized_minus_baseline_attack_success_rate": safe_delta.get(
                    "attack_success_rate"
                ),
                "research_safe_humanized_minus_baseline_full_break_rate": safe_delta.get(
                    "full_break_rate"
                ),
                "research_safe_humanized_minus_baseline_source_injection_absorption_rate": safe_delta.get(
                    "source_injection_absorption_rate"
                ),
            },
            "next_required_run": "Add live adversarial model runs and CI gate thresholds.",
        },
        {
            "benchmark": "Privacy / consent audit",
            "status": "not_measured",
            "current_measurement": None,
            "next_required_run": "Add record sensitivity labels, consent metadata, and named-user mimicry refusal tests.",
        },
    ]


def format_findings(results: dict[str, Any]) -> str:
    safety = results["safety"]
    humanization = results["humanization"]
    evidence_audits = results["evidence_audits"]
    matrix = results["benchmark_matrix"]
    status_counts = Counter(row["status"] for row in matrix)
    safe_delta = safety.get("deltas", {}).get("research_safe_humanized_minus_baseline", {})
    safety_metrics = safety.get("variant_metrics", {})
    baseline_safety = safety_metrics.get("baseline", {})
    safe_safety = safety_metrics.get("research_safe_humanized", {})

    lines = [
        "# Evidence-Grounded User Simulation Baseline Measurements",
        "",
        "Date: 2026-04-19",
        "Status: measured current pipeline artifacts; P0 replay benchmarks not implemented yet",
        "",
        "## Pipeline Runs Found",
        "",
        "This did not rerun the model pipelines. It inventories existing pipeline artifacts and writes a measured roadmap baseline.",
        "",
        "| Experiment | results.json | comparison.json | FINDINGS.md | Used here | results mtime UTC |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in results["experiment_inventory"]:
        lines.append(
            f"| `{item['name']}` | {item['has_results_json']} | "
            f"{item['has_comparison_json']} | {item['has_findings_md']} | "
            f"{item['used_for_roadmap_baseline']} | {item['results_mtime_utc'] or 'n/a'} |"
        )

    lines.extend([
        "",
        "## Benchmark Coverage",
        "",
        "| Status | Count |",
        "|---|---:|",
    ])
    for status in ("measured", "measured_partial", "not_measured"):
        lines.append(f"| {status} | {status_counts.get(status, 0)} |")

    lines.extend([
        "",
        "## Deltas From Existing Runs",
        "",
    ])

    if humanization.get("measured"):
        lines.extend([
            f"- Humanization twin overall avg: {humanization['baseline_twin_overall_avg']:.3f} -> {humanization['humanized_twin_overall_avg']:.3f} ({humanization['humanized_minus_baseline_avg']:+.3f}).",
            "- Humanization per-persona split: "
            + ", ".join(
                f"{row['name']} {row['delta']:+.3f}"
                for row in humanization.get("per_persona_deltas", [])
            )
            + ".",
        ])
    if safety.get("measured"):
        lines.extend([
            f"- Safety attack success: {pct(baseline_safety.get('attack_success_rate'))} -> {pct(safe_safety.get('attack_success_rate'))} ({pp(safe_delta.get('attack_success_rate'))}).",
            f"- Full break: {pct(baseline_safety.get('full_break_rate'))} -> {pct(safe_safety.get('full_break_rate'))} ({pp(safe_delta.get('full_break_rate'))}).",
            f"- Source injection absorption: {pct(baseline_safety.get('source_injection_absorption_rate'))} -> {pct(safe_safety.get('source_injection_absorption_rate'))} ({pp(safe_delta.get('source_injection_absorption_rate'))}).",
            f"- Counterevidence update: {pct(baseline_safety.get('counterevidence_update_rate'))} -> {pct(safe_safety.get('counterevidence_update_rate'))} ({pp(safe_delta.get('counterevidence_update_rate'))}).",
        ])

    lines.extend([
        "",
        "## Evidence Audit",
        "",
        "| Variant | Personas | Claims | Claim coverage | Common claims | Common coverage | Valid record links | Avg confidence | source_evidence.* paths |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for variant in ("legacy_v1", "baseline", "research_safe_humanized"):
        audit = evidence_audits[variant]
        lines.append(
            f"| {variant} | {audit['persona_count']} | {audit['claims_total']} | "
            f"{pct(audit['claim_field_coverage_rate'])} | {audit['common_claims_total']} | "
            f"{pct(audit['common_claim_field_coverage_rate'])} | "
            f"{pct(audit['valid_record_link_rate'])} | {audit['avg_confidence']:.3f} | "
            f"{pct(audit['source_evidence_field_path_rate'])} |"
        )

    lines.extend([
        "",
        "Interpretation: most record IDs are valid, but claim-level field coverage is insufficient and regresses in the current humanized outputs because much of the generated evidence points to `source_evidence.N` instead of claim paths like `goals.0` or `pains.1`. This is the measurement backing the P0 Evidence audit work.",
        "",
        "Caveat: `avg_confidence` is averaged over evidence entries, not over all claims, so high confidence does not imply most claims are supported. Coverage is structural only; it does not prove semantic support.",
        "",
        "## Roadmap Benchmark Matrix",
        "",
        "| Benchmark | Status | Next required run |",
        "|---|---|---|",
    ])
    for row in matrix:
        lines.append(
            f"| {row['benchmark']} | {row['status']} | {row['next_required_run']} |"
        )

    lines.extend([
        "",
        "## Decision",
        "",
        "The current pipeline has usable measurements for safety and partial evidence audit only. Human replay and multi-turn replay are not yet present in pipeline runs, so they must be implemented before making validity or prediction claims.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clusters = load_clusters()
    record_ids = valid_record_ids(clusters)

    evidence_audits = {
        variant: audit_variant(variant, record_ids)
        for variant in ("legacy_v1", "baseline", "research_safe_humanized")
    }
    humanization = summarize_humanization()
    safety = summarize_safety()
    inventory = experiment_inventory()
    matrix = benchmark_matrix(evidence_audits, safety)

    results = {
        "experiment": "evidence-grounded-user-simulation-roadmap-baseline",
        "title": "Evidence-Grounded User Simulation Baseline Measurements",
        "pipeline_mode": "offline_measurement_from_existing_pipeline_artifacts",
        "source_experiments": [
            "exp-humanization-ab",
            "exp-persona-vulnerability-ab",
        ],
        "num_clusters": len(clusters),
        "num_valid_record_ids": len(record_ids),
        "humanization": humanization,
        "safety": safety,
        "experiment_inventory": inventory,
        "evidence_audits": {
            variant: audit.to_dict() for variant, audit in evidence_audits.items()
        },
        "benchmark_matrix": matrix,
    }

    comparison = {
        "humanization_delta": {
            "humanized_minus_baseline_twin_overall_avg": humanization.get(
                "humanized_minus_baseline_avg"
            )
        },
        "safety_deltas": safety.get("deltas", {}),
        "evidence_audit_deltas": {
            "baseline_minus_legacy_v1_claim_field_coverage_rate": round(
                evidence_audits["baseline"].common_claim_field_coverage_rate
                - evidence_audits["legacy_v1"].common_claim_field_coverage_rate,
                4,
            ),
            "research_safe_humanized_minus_baseline_claim_field_coverage_rate": round(
                evidence_audits["research_safe_humanized"].common_claim_field_coverage_rate
                - evidence_audits["baseline"].common_claim_field_coverage_rate,
                4,
            ),
        },
    }

    write_json(OUTPUT_DIR / "results.json", results)
    write_json(OUTPUT_DIR / "comparison.json", comparison)
    (OUTPUT_DIR / "FINDINGS.md").write_text(format_findings(results))
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
