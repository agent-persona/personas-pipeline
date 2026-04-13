"""Experiment 3.16: synthetic ground-truth injection helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from segmentation.models.record import RawRecord


@dataclass(frozen=True)
class FactSpec:
    fact_id: str
    cluster_label: str
    target_source: str
    target_user_id: str
    payload_key: str
    payload_value: str
    signals: tuple[str, ...]


@dataclass
class AppliedFact:
    fact_id: str
    cluster_label: str
    record_id: str
    source: str
    target_user_id: str
    payload_key: str
    payload_value: str
    signals: tuple[str, ...]


@dataclass
class FactOutcome:
    fact_id: str
    cluster_label: str
    record_id: str
    control_signal_score: float
    injected_signal_score: float
    control_surface_hit: bool
    injected_surface_hit: bool
    control_evidence_hit: bool
    injected_evidence_hit: bool
    control_excerpt: str
    injected_excerpt: str


@dataclass
class PersonaRun:
    cluster_id: str
    persona_name: str
    groundedness: float
    cost_usd: float
    attempts: int
    persona: dict[str, Any]


@dataclass
class ClusterComparison:
    cluster_id: str
    control: PersonaRun
    injected: PersonaRun
    fact_outcomes: list[FactOutcome]


@dataclass
class ExperimentSummary:
    tenant_id: str
    n_records: int
    n_clusters: int
    n_facts: int
    mean_control_groundedness: float
    mean_injected_groundedness: float
    mean_control_cost_usd: float
    mean_injected_cost_usd: float
    mean_control_fact_survival: float
    mean_injected_fact_survival: float
    injected_survival_rate: float
    control_survival_rate: float
    provider: str
    synthesis_model: str


FACT_SPECS: tuple[FactSpec, ...] = (
    FactSpec(
        fact_id="engineer_csv_board_exports",
        cluster_label="engineer",
        target_source="intercom",
        target_user_id="user_eng_c",
        payload_key="injected_note",
        payload_value="Needs CSV exports for quarterly board meetings.",
        signals=("csv export", "quarterly board", "board meeting"),
    ),
    FactSpec(
        fact_id="engineer_slack_webhook_alerts",
        cluster_label="engineer",
        target_source="ga4",
        target_user_id="user_eng_c",
        payload_key="injected_note",
        payload_value="Wants Slack webhook alerts for deployment metrics and standup reporting.",
        signals=("slack webhook", "deployment metrics", "standup"),
    ),
    FactSpec(
        fact_id="engineer_ex_google_vp",
        cluster_label="engineer",
        target_source="hubspot",
        target_user_id="user_eng_d",
        payload_key="injected_note",
        payload_value="Title is VP of Engineering, ex-Google.",
        signals=("vp of engineering", "ex-google", "google"),
    ),
    FactSpec(
        fact_id="designer_white_label_share",
        cluster_label="designer",
        target_source="intercom",
        target_user_id="user_des_a",
        payload_key="injected_note",
        payload_value="Needs a white-labeled client share view for client presentations.",
        signals=("white-label", "client share", "client presentations"),
    ),
    FactSpec(
        fact_id="designer_presentation_mode",
        cluster_label="designer",
        target_source="intercom",
        target_user_id="user_des_b",
        payload_key="injected_note",
        payload_value="Needs a presentation mode for non-technical clients to review work.",
        signals=("presentation mode", "non-technical clients", "review work"),
    ),
    FactSpec(
        fact_id="designer_unlimited_share_links",
        cluster_label="designer",
        target_source="intercom",
        target_user_id="user_des_d",
        payload_key="injected_note",
        payload_value="Prefers unlimited share links over per-seat pricing.",
        signals=("unlimited share links", "per-seat", "pricing"),
    ),
)


def _flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for key, item in value.items():
            out.extend(_flatten_text(key))
            out.extend(_flatten_text(item))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_text(item))
        return out
    return [str(value)]


def persona_text(persona: dict[str, Any]) -> str:
    return "\n".join(_flatten_text(persona)).lower()


def signal_score(text: str, signals: tuple[str, ...]) -> float:
    if not signals:
        return 0.0
    found = sum(1 for signal in signals if signal.lower() in text)
    return found / len(signals)


def signal_hit(text: str, signals: tuple[str, ...], threshold: float = 0.34) -> bool:
    return signal_score(text, signals) >= threshold


def evidence_hit(persona: dict[str, Any], signals: tuple[str, ...]) -> bool:
    evidence = persona.get("source_evidence", [])
    evidence_text = "\n".join(_flatten_text(evidence)).lower()
    return signal_hit(evidence_text, signals)


def find_record(records: list[RawRecord], spec: FactSpec) -> RawRecord:
    for record in records:
        if record.source == spec.target_source and record.user_id == spec.target_user_id:
            return record
    raise ValueError(
        f"Could not find record for fact {spec.fact_id}: "
        f"{spec.target_source}/{spec.target_user_id}"
    )


def inject_synthetic_facts(records: list[RawRecord]) -> tuple[list[RawRecord], list[AppliedFact]]:
    injected = [RawRecord.model_validate(r.model_dump()) for r in records]
    applied: list[AppliedFact] = []
    by_record_id = {record.record_id: record for record in injected}
    for spec in FACT_SPECS:
        source_record = find_record(injected, spec)
        source_record.payload[spec.payload_key] = spec.payload_value
        applied.append(
            AppliedFact(
                fact_id=spec.fact_id,
                cluster_label=spec.cluster_label,
                record_id=source_record.record_id,
                source=source_record.source,
                target_user_id=source_record.user_id or "",
                payload_key=spec.payload_key,
                payload_value=spec.payload_value,
                signals=spec.signals,
            )
        )
        by_record_id[source_record.record_id] = source_record
    return list(by_record_id.values()), applied


def assess_fact(persona: dict[str, Any], fact: AppliedFact) -> tuple[float, bool, bool, str]:
    text = persona_text(persona)
    score = signal_score(text, fact.signals)
    surface = signal_hit(text, fact.signals)
    evidence = evidence_hit(persona, fact.signals)
    excerpt = ""
    if surface:
        for signal in fact.signals:
            if signal.lower() in text:
                excerpt = signal
                break
    return score, surface, evidence, excerpt


def applied_facts_to_dict(facts: list[AppliedFact]) -> list[dict[str, Any]]:
    return [asdict(fact) for fact in facts]

