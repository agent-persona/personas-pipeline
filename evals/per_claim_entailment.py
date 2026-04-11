"""Per-claim entailment analysis for experiment 3.05."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from statistics import mean

from synthesis.engine.groundedness import check_groundedness
from synthesis.models.cluster import ClusterData, SampleRecord
from synthesis.models.persona import PersonaV1

from evals.judge_helper_3_05 import ClaimEntailmentScore, LLMJudge

CLAIM_FIELDS = ("goals", "pains", "motivations", "objections")


@dataclass
class ClaimTarget:
    persona_name: str
    cluster_id: str
    field_name: str
    field_index: int
    claim: str
    field_path: str
    record_ids: list[str]
    confidence: float


@dataclass
class ClaimEvaluation:
    persona_name: str
    cluster_id: str
    field_name: str
    field_index: int
    field_path: str
    claim: str
    record_ids: list[str]
    evidence_context: str
    structural_grounded: bool
    structural_groundedness: float
    label: str
    confidence: float
    rationale: str
    judge_model: str


@dataclass
class PersonaEntailmentSummary:
    persona_name: str
    cluster_id: str
    judge_model: str
    n_claims: int
    n_entailed: int
    n_neutral: int
    n_contradicted: int
    entailment_rate: float
    neutral_rate: float
    contradiction_rate: float
    false_positive_grounding_rate: float
    structural_groundedness: float
    mean_judge_confidence: float
    claim_coverage: float


def iter_claim_targets(persona: PersonaV1, cluster: ClusterData) -> list[ClaimTarget]:
    targets: list[ClaimTarget] = []
    for field_name in CLAIM_FIELDS:
        items = getattr(persona, field_name)
        evidence = {
            ev.field_path: ev
            for ev in persona.source_evidence
            if ev.field_path.startswith(f"{field_name}.")
        }
        for idx, item in enumerate(items):
            field_path = f"{field_name}.{idx}"
            ev = evidence.get(field_path)
            targets.append(
                ClaimTarget(
                    persona_name=persona.name,
                    cluster_id=cluster.cluster_id,
                    field_name=field_name,
                    field_index=idx,
                    claim=str(item),
                    field_path=field_path,
                    record_ids=list(ev.record_ids) if ev else [],
                    confidence=float(ev.confidence) if ev else 0.0,
                )
            )
    return targets


def _record_text(record: SampleRecord) -> str:
    payload = json.dumps(record.payload, indent=2, sort_keys=True, default=str)
    return (
        f"- record_id: {record.record_id}\n"
        f"  source: {record.source}\n"
        f"  payload: {payload}"
    )


def build_evidence_context(cluster: ClusterData, record_ids: list[str]) -> str:
    records_by_id = {record.record_id: record for record in cluster.sample_records}
    lines: list[str] = []
    for record_id in sorted(dict.fromkeys(record_ids)):
        record = records_by_id.get(record_id)
        if record is None:
            lines.append(f"- record_id: {record_id} (missing from cluster)")
            continue
        lines.append(_record_text(record))
    return "\n".join(lines) if lines else "- no valid cited records"


async def evaluate_claims_async(
    persona: PersonaV1,
    cluster: ClusterData,
    judge: LLMJudge,
) -> list[ClaimEvaluation]:
    claim_targets = iter_claim_targets(persona, cluster)
    cluster_ids = set(cluster.all_record_ids)
    structural = check_groundedness(persona, cluster)
    claim_results: list[ClaimEvaluation] = []

    for target in claim_targets:
        evidence_context = build_evidence_context(cluster, target.record_ids)
        score: ClaimEntailmentScore = await judge.score_claim(
            persona_name=target.persona_name,
            field_path=target.field_path,
            claim=target.claim,
            evidence_context=evidence_context,
        )
        claim_results.append(
            ClaimEvaluation(
                persona_name=target.persona_name,
                cluster_id=target.cluster_id,
                field_name=target.field_name,
                field_index=target.field_index,
                field_path=target.field_path,
                claim=target.claim,
                record_ids=target.record_ids,
                evidence_context=evidence_context,
                structural_grounded=all(record_id in cluster_ids for record_id in target.record_ids),
                structural_groundedness=structural.score,
                label=score.label,
                confidence=score.confidence,
                rationale=score.rationale,
                judge_model=score.judge_model,
            )
        )

    return claim_results


def summarize_persona_claims(persona: PersonaV1, evaluations: list[ClaimEvaluation], cluster: ClusterData) -> PersonaEntailmentSummary:
    total = len(evaluations)
    entailed = sum(1 for ev in evaluations if ev.label == "entailed")
    neutral = sum(1 for ev in evaluations if ev.label == "neutral")
    contradicted = sum(1 for ev in evaluations if ev.label == "contradicted")
    valid_evidence = sum(1 for ev in evaluations if ev.structural_grounded)
    false_positive = sum(1 for ev in evaluations if ev.structural_grounded and ev.label != "entailed")
    structural = check_groundedness(persona, cluster)
    return PersonaEntailmentSummary(
        persona_name=persona.name,
        cluster_id=cluster.cluster_id,
        judge_model=evaluations[0].judge_model if evaluations else "",
        n_claims=total,
        n_entailed=entailed,
        n_neutral=neutral,
        n_contradicted=contradicted,
        entailment_rate=(entailed / total) if total else 0.0,
        neutral_rate=(neutral / total) if total else 0.0,
        contradiction_rate=(contradicted / total) if total else 0.0,
        false_positive_grounding_rate=(false_positive / valid_evidence) if valid_evidence else 0.0,
        structural_groundedness=structural.score,
        mean_judge_confidence=mean(ev.confidence for ev in evaluations) if evaluations else 0.0,
        claim_coverage=(valid_evidence / total) if total else 0.0,
    )


def safe_pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = mean(xs)
    mean_y = mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    denom_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def results_to_dict(
    persona_evaluations: list[dict],
    persona_summaries: list[PersonaEntailmentSummary],
) -> dict:
    return {
        "persona_evaluations": persona_evaluations,
        "persona_summaries": [asdict(summary) for summary in persona_summaries],
    }
