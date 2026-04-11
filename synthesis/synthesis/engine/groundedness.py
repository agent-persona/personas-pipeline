from __future__ import annotations

from dataclasses import dataclass, field

from synthesis.models.cluster import ClusterData
from synthesis.models.persona import PersonaV1

# Fields that require at least one source_evidence entry
EVIDENCE_REQUIRED_FIELDS = ("goals", "pains", "motivations", "objections")

# ── Experiment 3.20: corroboration thresholds ────────────────────────
# Claims with confidence >= HIGH_CONFIDENCE_THRESHOLD but only 1 record
# are flagged as over-confident.
HIGH_CONFIDENCE_THRESHOLD = 0.8
MIN_CORROBORATION_FOR_HIGH_CONFIDENCE = 2


@dataclass
class CorroborationReport:
    """Experiment 3.20: confidence vs corroboration analysis."""
    total_evidence: int = 0
    over_confident_count: int = 0       # high confidence, low corroboration
    well_corroborated_count: int = 0    # confidence matches corroboration
    under_confident_count: int = 0      # low confidence, high corroboration
    avg_confidence: float = 0.0
    avg_corroboration: float = 0.0
    calibration_score: float = 0.0      # 1.0 = perfect calibration
    violations: list[str] = field(default_factory=list)


@dataclass
class GroundednessReport:
    score: float
    violations: list[str] = field(default_factory=list)
    corroboration: CorroborationReport | None = None

    @property
    def passed(self) -> bool:
        return self.score >= 0.9


def check_corroboration(persona: PersonaV1) -> CorroborationReport:
    """Experiment 3.20: Cross-check confidence against corroboration depth.

    Flags over-confident claims (confidence >= 0.8 but only 1 source record)
    and computes calibration score between confidence and corroboration.
    """
    evidence = persona.source_evidence
    if not evidence:
        return CorroborationReport()

    total = len(evidence)
    over_confident = 0
    well_corroborated = 0
    under_confident = 0
    violations: list[str] = []

    confidences: list[float] = []
    corroborations: list[int] = []

    for ev in evidence:
        depth = ev.corroboration_depth
        conf = ev.confidence
        confidences.append(conf)
        corroborations.append(depth)

        if conf >= HIGH_CONFIDENCE_THRESHOLD and depth < MIN_CORROBORATION_FOR_HIGH_CONFIDENCE:
            over_confident += 1
            violations.append(
                f"Over-confident: {ev.field_path} has confidence={conf:.2f} "
                f"but only {depth} record(s) — needs >= {MIN_CORROBORATION_FOR_HIGH_CONFIDENCE}"
            )
        elif conf < 0.5 and depth >= 3:
            under_confident += 1
        else:
            well_corroborated += 1

    avg_conf = sum(confidences) / total
    avg_corr = sum(corroborations) / total

    # Calibration: how well does confidence track corroboration?
    # Normalize corroboration to 0-1 scale (cap at 5 records = 1.0)
    max_corr = 5.0
    norm_corr = [min(c / max_corr, 1.0) for c in corroborations]
    # Mean absolute error between confidence and normalized corroboration
    mae = sum(abs(c - nc) for c, nc in zip(confidences, norm_corr)) / total
    calibration = 1.0 - mae  # 1.0 = perfect, 0.0 = worst

    return CorroborationReport(
        total_evidence=total,
        over_confident_count=over_confident,
        well_corroborated_count=well_corroborated,
        under_confident_count=under_confident,
        avg_confidence=avg_conf,
        avg_corroboration=avg_corr,
        calibration_score=calibration,
        violations=violations,
    )


def check_groundedness(
    persona: PersonaV1,
    cluster: ClusterData,
    enforce_corroboration: bool = False,
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
    for field_name in EVIDENCE_REQUIRED_FIELDS:
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

    # Experiment 3.20: corroboration check
    corroboration = check_corroboration(persona)
    if enforce_corroboration and corroboration.over_confident_count > 0:
        violations.extend(corroboration.violations)
        # Penalize score proportionally to over-confident claims
        penalty = corroboration.over_confident_count / max(len(persona.source_evidence), 1)
        score = max(0.0, score - penalty * 0.1)

    return GroundednessReport(
        score=score,
        violations=violations,
        corroboration=corroboration,
    )
