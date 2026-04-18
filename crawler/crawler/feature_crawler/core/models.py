from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


def _strip_none(value: Any) -> Any:
    if is_dataclass(value):
        return _strip_none(asdict(value))
    if isinstance(value, dict):
        return {
            key: _strip_none(item)
            for key, item in value.items()
            if item is not None and _strip_none(item) != {}
        }
    if isinstance(value, list):
        return [_strip_none(item) for item in value if item is not None]
    return value


@dataclass(slots=True)
class EvidencePointer:
    source_url: str
    fetched_at: str
    derived_from_message_id: str | None = None


@dataclass(slots=True)
class CrawlTarget:
    platform: str
    target_id: str
    url: str
    community_name: str
    collection_basis: str
    allow_persona_inference: bool = False
    allow_cross_linking: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AccountRecord:
    record_type: str
    platform: str
    platform_user_id: str
    username: str
    account_created_at: str | None
    first_observed_at: str
    crawl_run_id: str
    evidence_pointer: EvidencePointer


@dataclass(slots=True)
class ProfileSnapshotRecord:
    record_type: str
    platform: str
    platform_user_id: str
    snapshot_at: str
    crawl_run_id: str
    fields: dict[str, Any]
    evidence_pointer: EvidencePointer


@dataclass(slots=True)
class CommunityRecord:
    record_type: str
    platform: str
    community_id: str
    community_name: str
    community_type: str
    parent_community_id: str | None
    description: str | None
    member_count: int | None
    rules_summary: str | None
    observed_at: str
    crawl_run_id: str
    evidence_pointer: EvidencePointer


@dataclass(slots=True)
class ThreadRecord:
    record_type: str
    platform: str
    thread_id: str
    community_id: str
    title: str
    author_platform_user_id: str | None
    created_at: str
    observed_at: str
    crawl_run_id: str
    metadata: dict[str, Any]
    evidence_pointer: EvidencePointer


@dataclass(slots=True)
class MessageRecord:
    record_type: str
    platform: str
    message_id: str
    thread_id: str
    community_id: str
    author_platform_user_id: str | None
    body: str
    created_at: str
    observed_at: str
    crawl_run_id: str
    reply_to_message_id: str | None
    reply_to_user_id: str | None
    metadata: dict[str, Any]
    evidence_pointer: EvidencePointer


@dataclass(slots=True)
class InteractionRecord:
    record_type: str
    platform: str
    interaction_type: str
    source_user_id: str
    target_user_id: str
    message_id: str
    thread_id: str
    community_id: str
    created_at: str
    crawl_run_id: str
    evidence_pointer: EvidencePointer


@dataclass(slots=True)
class TombstoneRecord:
    record_type: str
    platform: str
    tombstone_type: str
    target_record_type: str
    target_record_id: str
    observed_at: str
    crawl_run_id: str
    previous_body_hash: str | None
    reason: str
    evidence_pointer: EvidencePointer


Record = (
    AccountRecord
    | ProfileSnapshotRecord
    | CommunityRecord
    | ThreadRecord
    | MessageRecord
    | InteractionRecord
    | TombstoneRecord
)


def record_to_dict(record: Record) -> dict[str, Any]:
    return _strip_none(record)
