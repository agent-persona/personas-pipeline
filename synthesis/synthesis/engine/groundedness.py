from __future__ import annotations

from dataclasses import dataclass, field

from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1

# Fields that require at least one source_evidence entry
EVIDENCE_REQUIRED_FIELDS = ("goals", "pains", "motivations", "objections")
EVIDENCE_REQUIRED_FIELDS_V2 = EVIDENCE_REQUIRED_FIELDS + ("emotional_triggers",)


@dataclass
class GroundednessReport:
    score: float
    violations: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.score >= 0.9


def check_groundedness(
    persona: PersonaV1,
    cluster: ClusterData,
    evidence_fields: tuple[str, ...] | None = None,
) -> GroundednessReport:
    """Run deterministic groundedness checks on a synthesized persona.

    Checks:
    1. Every source_evidence entry references valid record IDs from the cluster.
    2. Every item in evidence-required fields has at least one evidence entry.

    Returns a GroundednessReport with a score (0.0-1.0) and any violations.
    """
    valid_ids = set(cluster.all_record_ids)
    violations: list[str] = []

    # Check 1: All referenced record IDs exist in the cluster
    invalid_evidence_indices: set[int] = set()
    for i, ev in enumerate(persona.source_evidence):
        bad_ids = [rid for rid in ev.record_ids if rid not in valid_ids]
        if bad_ids:
            violations.append(
                f"source_evidence[{i}] references unknown record IDs: {bad_ids}"
            )
            invalid_evidence_indices.add(i)

    # Check 2: Every item in required fields has a corresponding evidence entry
    # with valid record IDs (evidence pointing to bad records doesn't count)
    valid_evidence_paths: set[str] = set()
    for i, ev in enumerate(persona.source_evidence):
        if i not in invalid_evidence_indices:
            valid_evidence_paths.add(ev.field_path)

    total_required = 0
    covered = 0
    for field_name in (evidence_fields if evidence_fields is not None else EVIDENCE_REQUIRED_FIELDS):
        items = getattr(persona, field_name)
        for idx in range(len(items)):
            total_required += 1
            path = f"{field_name}.{idx}"
            if path in valid_evidence_paths:
                covered += 1
            else:
                violations.append(
                    f"No valid source_evidence entry for {path}: {items[idx]!r}"
                )

    if total_required == 0:
        score = 1.0
    else:
        score = covered / total_required

    return GroundednessReport(score=score, violations=violations)
