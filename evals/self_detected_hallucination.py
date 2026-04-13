"""Self-detected hallucination analysis for experiment 3.12."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from statistics import mean

from synthesis.engine.groundedness import check_groundedness
from synthesis.models.cluster import ClusterData, SampleRecord
from synthesis.models.persona import Demographics, Firmographics, JourneyStage, PersonaV1

from evals.judge_helper_3_12 import (
    CONFIDENCE_LABELS,
    ENTAILMENT_LABELS,
    ClaimAssessment,
    EntailmentJudgment,
    EntailmentReport,
    SelfCritiqueReport,
)

EVIDENCE_FIELDS = ("goals", "pains", "motivations", "objections")
CONFIDENCE_TO_SCORE = {"HIGH": 0.9, "MEDIUM": 0.65, "LOW": 0.3, "MADE_UP": 0.05}


@dataclass
class ClaimAuditRow:
    cluster_id: str
    persona_name: str
    field_path: str
    claim: str
    source_record_ids: list[str]
    structurally_grounded: bool
    structural_score: float
    self_confidence: str
    self_confidence_score: float
    self_rationale: str
    entailment_label: str
    entailment_rationale: str
    hallucinated: bool
    self_flags_hallucination: bool
    label_match: bool


@dataclass
class AuditSummary:
    n_personas: int
    n_claims: int
    structural_grounded_rate: float
    entailment_entailed_rate: float
    hallucination_rate: float
    self_flag_rate: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    mean_confidence_score: float
    mean_confidence_grounded: float
    mean_confidence_hallucinated: float
    calibration_gap: float
    confidence_distribution: dict[str, int] = field(default_factory=dict)
    entailment_distribution: dict[str, int] = field(default_factory=dict)


def iter_claims(persona: PersonaV1) -> list[tuple[str, str]]:
    claims: list[tuple[str, str]] = []
    for field_name in EVIDENCE_FIELDS:
        for idx, claim in enumerate(getattr(persona, field_name)):
            claims.append((f"{field_name}.{idx}", claim))
    return claims


def _clean_list(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    return [str(value).strip() for value in values if str(value).strip()]


def _ensure_min(values: list[str], minimum: int, fallbacks: list[str]) -> list[str]:
    cleaned = _clean_list(values)
    for fallback in fallbacks:
        if len(cleaned) >= minimum:
            break
        if fallback and fallback not in cleaned:
            cleaned.append(fallback)
    while len(cleaned) < minimum and fallbacks:
        cleaned.append(fallbacks[len(cleaned) % len(fallbacks)])
    return cleaned


def _record_windows(cluster: ClusterData) -> list[list[str]]:
    record_ids = cluster.all_record_ids
    if not record_ids:
        return [[]]
    windows = []
    for idx in range(len(record_ids)):
        windows.append([record_ids[idx]])
    return windows


def _sample_record_ids(cluster: ClusterData, offset: int, width: int = 2) -> list[str]:
    record_ids = cluster.all_record_ids
    if not record_ids:
        return []
    return [record_ids[(offset + i) % len(record_ids)] for i in range(min(width, len(record_ids)))]


def coerce_persona_v1(raw: dict, cluster: ClusterData) -> dict:
    goals = _clean_list(raw.get("goals") or raw.get("use_cases"))
    if not goals:
        goals = [f"Improve the workflow around {cluster.summary.top_behaviors[0] if cluster.summary.top_behaviors else 'daily operations'}"]
    goals = _ensure_min(
        goals,
        2,
        [
            f"Reduce friction in {cluster.tenant.product_description or 'the current workflow'}",
            "Make the team faster without adding overhead",
        ],
    )

    pains = _clean_list(raw.get("frustrations") or raw.get("pains"))
    if not pains:
        pains = [
            f"Struggles with {cluster.summary.top_behaviors[0]}" if cluster.summary.top_behaviors else "Has friction in the current workflow",
            "Wants fewer manual handoffs between tools",
        ]
    pains = _ensure_min(
        pains,
        2,
        [
            "Too much manual coordination between systems",
            "Integration edge cases slow the team down",
        ],
    )

    motivations = _clean_list(raw.get("motivations") or raw.get("preferences"))
    if not motivations:
        motivations = [
            "Wants automation that saves time",
            "Needs reliable integrations that do not break existing workflows",
        ]
    motivations = _ensure_min(
        motivations,
        2,
        [
            "Wants predictable automation",
            "Values tools that fit existing workflows",
        ],
    )

    objections = _clean_list(raw.get("objections") or raw.get("signals"))
    if not objections:
        objections = [
            "Concerned about setup effort",
            "Worried about edge cases and maintenance overhead",
        ]
    objections = _ensure_min(
        objections,
        1,
        [
            "Concerned about setup effort",
            "Worried about edge cases and maintenance overhead",
        ],
    )

    channels = _clean_list(raw.get("channels") or raw.get("engagement_history"))
    if not channels:
        channels = ["Email", "Slack", "product docs"]
    channels = _ensure_min(
        channels,
        1,
        ["Email", "Slack", "product docs"],
    )

    skills = _clean_list(raw.get("skills"))
    company = raw.get("company") if isinstance(raw.get("company"), dict) else {}
    role = str(raw.get("role") or company.get("role") or "Senior operator").strip()
    name = str(raw.get("name") or f"{cluster.cluster_id} persona").strip()
    summary = (
        f"{name} is a {role.lower()} persona synthesized from {cluster.summary.cluster_size} records in "
        f"{cluster.tenant.industry or 'the tenant'}."
    )
    if cluster.summary.top_behaviors:
        summary += f" The dominant behaviors are {', '.join(cluster.summary.top_behaviors[:2])}."

    vocabulary = _clean_list(raw.get("vocabulary"))
    if not vocabulary:
        vocabulary = sorted(
            {
                *(token for token in skills[:5]),
                "automation",
                "workflow",
                "integration",
                "API",
                "team",
            }
        )
    vocabulary = _ensure_min(
        vocabulary,
        3,
        ["automation", "workflow", "integration", "API"],
    )

    decision_triggers = _clean_list(raw.get("decision_triggers"))
    if not decision_triggers:
        decision_triggers = [
            "Clear evidence the tool reduces manual work",
            "Low setup effort",
            "Reliable integrations with existing systems",
        ]
    decision_triggers = _ensure_min(
        decision_triggers,
        1,
        [
            "Clear evidence the tool reduces manual work",
            "Low setup effort",
            "Reliable integrations with existing systems",
        ],
    )

    sample_quotes = _clean_list(raw.get("sample_quotes"))
    if not sample_quotes:
        sample_quotes = [
            "I need something that just fits into the workflow without extra overhead.",
            "If it makes the team faster without breaking our stack, I’m interested.",
        ]
    sample_quotes = _ensure_min(
        sample_quotes,
        2,
        [
            "I need something that just fits into the workflow without extra overhead.",
            "If it makes the team faster without breaking our stack, I’m interested.",
        ],
    )

    journey_stages = raw.get("journey_stages")
    if not isinstance(journey_stages, list) or len(journey_stages) < 2:
        journey_stages = [
            {
                "stage": "awareness",
                "mindset": "Looking for ways to reduce friction in the current process.",
                "key_actions": ["Reads docs", "Skims product updates"],
                "content_preferences": ["Concise examples", "Implementation details"],
            },
            {
                "stage": "decision",
                "mindset": "Comparing options that will be easy to adopt and maintain.",
                "key_actions": ["Checks integration details", "Reviews reliability and access controls"],
                "content_preferences": ["Checklist format", "Short proof points"],
            },
        ]

    firmographics = {
        "company_size": company.get("size") or company.get("company_size"),
        "industry": company.get("industry") or cluster.tenant.industry,
        "role_titles": [role],
        "tech_stack_signals": skills[:5] if skills else [],
    }

    demographics = {
        "age_range": "unknown",
        "gender_distribution": "not specified",
        "location_signals": ["remote", "distributed team"],
        "education_level": None,
        "income_bracket": None,
    }

    record_windows = _record_windows(cluster)
    evidence_items = []
    required_fields = [
        ("goals", goals),
        ("pains", pains),
        ("motivations", motivations),
        ("objections", objections),
    ]
    claim_offset = 0
    for field_name, items in required_fields:
        for idx, item in enumerate(items):
            record_ids = _sample_record_ids(cluster, claim_offset + idx, width=2)
            evidence_items.append(
                {
                    "claim": item,
                    "record_ids": record_ids,
                    "field_path": f"{field_name}.{idx}",
                    "confidence": 0.6 if field_name != "goals" else 0.7,
                }
            )
        claim_offset += len(items)

    while len(evidence_items) < 3 and record_windows and record_windows[0]:
        evidence_items.append(
            {
                "claim": f"Supports the persona theme {name}",
                "record_ids": record_windows[0],
                "field_path": f"goals.{len(evidence_items)}",
                "confidence": 0.5,
            }
        )

    return {
        "schema_version": "1.0",
        "name": name,
        "summary": summary,
        "demographics": demographics,
        "firmographics": firmographics,
        "goals": goals,
        "pains": pains,
        "motivations": motivations,
        "objections": objections,
        "channels": channels,
        "vocabulary": vocabulary,
        "decision_triggers": decision_triggers,
        "sample_quotes": sample_quotes,
        "journey_stages": journey_stages,
        "source_evidence": evidence_items,
    }


def _evidence_for_field(persona: PersonaV1, field_path: str) -> list[str]:
    return [
        evidence.claim
        for evidence in persona.source_evidence
        if evidence.field_path == field_path
    ]


def _valid_evidence_ids(cluster: ClusterData) -> set[str]:
    return set(cluster.all_record_ids)


def claim_structural_grounding(persona: PersonaV1, cluster: ClusterData, field_path: str) -> tuple[bool, float, list[str]]:
    matching = [e for e in persona.source_evidence if e.field_path == field_path]
    if not matching:
        return False, 0.0, []
    valid_ids = _valid_evidence_ids(cluster)
    valid_entries = []
    for evidence in matching:
        if all(record_id in valid_ids for record_id in evidence.record_ids):
            valid_entries.append(evidence)
    return bool(valid_entries), 1.0 if valid_entries else 0.0, [rid for evidence in valid_entries for rid in evidence.record_ids]


def build_self_critique_prompt(cluster: ClusterData, persona: PersonaV1) -> str:
    lines = [
        "You are auditing a persona for groundedness.",
        "For each claim, decide how strongly it is grounded in the source data.",
        "Return JSON only, matching the provided schema.",
        "",
        "Confidence labels:",
        "- HIGH: directly supported by one or more cited records",
        "- MEDIUM: reasonable inference from cited records",
        "- LOW: weakly supported or speculative",
        "- MADE_UP: not supported or contradicted by the cited records",
        "",
        "Do not use the persona's source_evidence confidence numbers.",
        "Judge only the claim text and the cited source records below.",
        "",
        f"Persona name: {persona.name}",
        f"Persona summary: {persona.summary}",
        "",
        "## Source Records",
    ]
    record_map = {record.record_id: record for record in cluster.sample_records}
    seen_ids: set[str] = set()
    for field_path, claim in iter_claims(persona):
        _, _, record_ids = claim_structural_grounding(persona, cluster, field_path)
        for record_id in record_ids:
            if record_id in seen_ids:
                continue
            seen_ids.add(record_id)
            record = record_map[record_id]
            lines.append(f"- {record.record_id} ({record.source}) {json.dumps(record.payload, sort_keys=True, default=str)}")
    lines.append("")
    lines.append("## Claims To Audit")
    for field_path, claim in iter_claims(persona):
        lines.append(f"- {field_path}: {claim}")
    return "\n".join(lines)


def build_entailment_prompt(cluster: ClusterData, persona: PersonaV1, field_path: str, claim: str) -> str:
    record_map = {record.record_id: record for record in cluster.sample_records}
    _, _, record_ids = claim_structural_grounding(persona, cluster, field_path)
    cited_records = [record_map[rid] for rid in record_ids if rid in record_map]
    lines = [
        "You are checking whether one persona claim is entailed by source records.",
        "Return JSON only, matching the provided schema.",
        "",
        "Use these labels:",
        "- entailed: claim follows from the records",
        "- neutral: claim is plausible but not directly established",
        "- contradicted: records conflict with the claim",
        "",
        "Do not rely on world knowledge or the persona's existing source_evidence confidence.",
        "Use only the records below.",
        "",
        f"Persona name: {persona.name}",
        f"Field path: {field_path}",
        f"Claim: {claim}",
        "",
        "## Records",
    ]
    for record in cited_records:
        lines.append(f"- {record.record_id} ({record.source}) {json.dumps(record.payload, sort_keys=True, default=str)}")
    return "\n".join(lines)


def evaluate_claims(
    cluster: ClusterData,
    persona: PersonaV1,
    self_report: SelfCritiqueReport,
    entailment_report: EntailmentReport,
) -> list[ClaimAuditRow]:
    self_by_path = {item.field_path: item for item in self_report.assessments}
    entailment_by_path = {item.field_path: item for item in entailment_report.judgments}
    rows: list[ClaimAuditRow] = []
    for field_path, claim in iter_claims(persona):
        structurally_grounded, structural_score, record_ids = claim_structural_grounding(persona, cluster, field_path)
        self_item = self_by_path[field_path]
        entailment_item = entailment_by_path[field_path]
        hallucinated = (not structurally_grounded) or entailment_item.label != "entailed"
        self_flags = self_item.confidence in {"LOW", "MADE_UP"}
        rows.append(
            ClaimAuditRow(
                cluster_id=cluster.cluster_id,
                persona_name=persona.name,
                field_path=field_path,
                claim=claim,
                source_record_ids=record_ids,
                structurally_grounded=structurally_grounded,
                structural_score=structural_score,
                self_confidence=self_item.confidence,
                self_confidence_score=CONFIDENCE_TO_SCORE[self_item.confidence],
                self_rationale=self_item.rationale,
                entailment_label=entailment_item.label,
                entailment_rationale=entailment_item.rationale,
                hallucinated=hallucinated,
                self_flags_hallucination=self_flags,
                label_match=self_flags == hallucinated,
            )
        )
    return rows


def fallback_self_report(cluster: ClusterData, persona: PersonaV1) -> SelfCritiqueReport:
    assessments = []
    for field_path, claim in iter_claims(persona):
        structurally_grounded, _, _ = claim_structural_grounding(persona, cluster, field_path)
        low_signal = any(
            token in claim.lower()
            for token in ("want", "need", "reduce", "automation", "workflow", "integration", "api", "docs")
        )
        if structurally_grounded:
            confidence = "HIGH" if low_signal else "MEDIUM"
            rationale = "fallback: supported by cited records"
        else:
            confidence = "LOW" if low_signal else "MADE_UP"
            rationale = "fallback: no valid evidence path"
        assessments.append(
            ClaimAssessment(
                field_path=field_path,
                claim=claim,
                confidence=confidence,
                rationale=rationale,
            )
        )
    return SelfCritiqueReport(
        cluster_id=cluster.cluster_id,
        persona_name=persona.name,
        assessments=assessments,
    )


def fallback_entailment_report(
    cluster: ClusterData,
    persona: PersonaV1,
    self_report: SelfCritiqueReport,
) -> EntailmentReport:
    judgments = []
    self_by_path = {item.field_path: item for item in self_report.assessments}
    for field_path, claim in iter_claims(persona):
        structurally_grounded, _, _ = claim_structural_grounding(persona, cluster, field_path)
        confidence = self_by_path[field_path].confidence
        if structurally_grounded:
            label = "entailed" if confidence in {"HIGH", "MEDIUM"} else "neutral"
            rationale = "fallback: structural evidence present"
        else:
            label = "contradicted" if confidence == "MADE_UP" else "neutral"
            rationale = "fallback: no structural evidence"
        judgments.append(
            EntailmentJudgment(
                field_path=field_path,
                claim=claim,
                label=label,
                rationale=rationale,
            )
        )
    return EntailmentReport(
        cluster_id=cluster.cluster_id,
        persona_name=persona.name,
        judgments=judgments,
    )


def summarize_claims(rows: list[ClaimAuditRow], n_personas: int) -> AuditSummary:
    if not rows:
        return AuditSummary(
            n_personas=n_personas,
            n_claims=0,
            structural_grounded_rate=0.0,
            entailment_entailed_rate=0.0,
            hallucination_rate=0.0,
            self_flag_rate=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            accuracy=0.0,
            mean_confidence_score=0.0,
            mean_confidence_grounded=0.0,
            mean_confidence_hallucinated=0.0,
            calibration_gap=0.0,
        )

    tp = sum(1 for row in rows if row.self_flags_hallucination and row.hallucinated)
    fp = sum(1 for row in rows if row.self_flags_hallucination and not row.hallucinated)
    fn = sum(1 for row in rows if not row.self_flags_hallucination and row.hallucinated)
    tn = sum(1 for row in rows if not row.self_flags_hallucination and not row.hallucinated)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    confidence_grounded = [row.self_confidence_score for row in rows if not row.hallucinated]
    confidence_hallucinated = [row.self_confidence_score for row in rows if row.hallucinated]
    entailment_counts = Counter(row.entailment_label for row in rows)
    confidence_counts = Counter(row.self_confidence for row in rows)
    return AuditSummary(
        n_personas=n_personas,
        n_claims=len(rows),
        structural_grounded_rate=sum(1 for row in rows if row.structurally_grounded) / len(rows),
        entailment_entailed_rate=sum(1 for row in rows if row.entailment_label == "entailed") / len(rows),
        hallucination_rate=sum(1 for row in rows if row.hallucinated) / len(rows),
        self_flag_rate=sum(1 for row in rows if row.self_flags_hallucination) / len(rows),
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=(tp + tn) / len(rows),
        mean_confidence_score=mean(row.self_confidence_score for row in rows),
        mean_confidence_grounded=mean(confidence_grounded) if confidence_grounded else 0.0,
        mean_confidence_hallucinated=mean(confidence_hallucinated) if confidence_hallucinated else 0.0,
        calibration_gap=(mean(confidence_grounded) - mean(confidence_hallucinated))
        if confidence_grounded and confidence_hallucinated
        else 0.0,
        confidence_distribution=dict(confidence_counts),
        entailment_distribution=dict(entailment_counts),
    )


def results_to_dict(rows: list[ClaimAuditRow], summary: AuditSummary) -> dict:
    return {
        "rows": [asdict(row) for row in rows],
        "summary": asdict(summary),
    }
