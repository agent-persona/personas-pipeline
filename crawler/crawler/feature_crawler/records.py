from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class RecordValidationError(ValueError):
    """Raised when a record does not satisfy the bronze contract."""


def _require(value: bool, message: str) -> None:
    if not value:
        raise RecordValidationError(message)


def _non_empty(name: str, value: str | None) -> None:
    _require(isinstance(value, str) and bool(value.strip()), f"{name} must be non-empty")


def _isoish(name: str, value: str | None) -> None:
    _non_empty(name, value)
    _require("T" in value, f"{name} must look like an ISO-8601 timestamp")


@dataclass(frozen=True)
class EvidencePointer:
    source_url: str
    fetched_at: str
    derived_from_message_id: str | None = None

    def validate(self) -> None:
        _non_empty("source_url", self.source_url)
        _isoish("fetched_at", self.fetched_at)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class Record:
    record_type: str
    platform: str
    crawl_run_id: str

    def validate(self) -> None:
        _non_empty("record_type", self.record_type)
        _non_empty("platform", self.platform)
        _non_empty("crawl_run_id", self.crawl_run_id)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class AccountRecord(Record):
    platform_user_id: str
    username: str
    account_created_at: str | None
    first_observed_at: str
    evidence_pointer: EvidencePointer

    def validate(self) -> None:
        super().validate()
        _require(self.record_type == "account", "account record_type mismatch")
        _non_empty("platform_user_id", self.platform_user_id)
        _non_empty("username", self.username)
        if self.account_created_at is not None:
            _isoish("account_created_at", self.account_created_at)
        _isoish("first_observed_at", self.first_observed_at)
        self.evidence_pointer.validate()


@dataclass(frozen=True)
class ProfileSnapshotRecord(Record):
    platform_user_id: str
    snapshot_at: str
    fields: dict[str, Any]
    evidence_pointer: EvidencePointer

    def validate(self) -> None:
        super().validate()
        _require(self.record_type == "profile_snapshot", "profile_snapshot record_type mismatch")
        _non_empty("platform_user_id", self.platform_user_id)
        _isoish("snapshot_at", self.snapshot_at)
        _require(isinstance(self.fields, dict) and bool(self.fields), "fields must be a non-empty dict")
        self.evidence_pointer.validate()


@dataclass(frozen=True)
class CommunityRecord(Record):
    community_id: str
    community_name: str
    community_type: str
    parent_community_id: str | None
    description: str | None
    member_count: int | None
    rules_summary: str | None
    observed_at: str
    evidence_pointer: EvidencePointer

    def validate(self) -> None:
        super().validate()
        _require(self.record_type == "community", "community record_type mismatch")
        _non_empty("community_id", self.community_id)
        _non_empty("community_name", self.community_name)
        _non_empty("community_type", self.community_type)
        if self.parent_community_id is not None:
            _non_empty("parent_community_id", self.parent_community_id)
        _isoish("observed_at", self.observed_at)
        self.evidence_pointer.validate()


@dataclass(frozen=True)
class ThreadRecord(Record):
    thread_id: str
    community_id: str
    title: str
    author_platform_user_id: str
    created_at: str
    observed_at: str
    metadata: dict[str, Any]
    evidence_pointer: EvidencePointer

    def validate(self) -> None:
        super().validate()
        _require(self.record_type == "thread", "thread record_type mismatch")
        _non_empty("thread_id", self.thread_id)
        _non_empty("community_id", self.community_id)
        _non_empty("title", self.title)
        _non_empty("author_platform_user_id", self.author_platform_user_id)
        _isoish("created_at", self.created_at)
        _isoish("observed_at", self.observed_at)
        _require(isinstance(self.metadata, dict), "metadata must be a dict")
        self.evidence_pointer.validate()


@dataclass(frozen=True)
class MessageRecord(Record):
    message_id: str
    thread_id: str
    community_id: str
    author_platform_user_id: str
    body: str
    created_at: str
    observed_at: str
    reply_to_message_id: str | None
    reply_to_user_id: str | None
    metadata: dict[str, Any]
    evidence_pointer: EvidencePointer

    def validate(self) -> None:
        super().validate()
        _require(self.record_type == "message", "message record_type mismatch")
        _non_empty("message_id", self.message_id)
        _non_empty("thread_id", self.thread_id)
        _non_empty("community_id", self.community_id)
        _non_empty("author_platform_user_id", self.author_platform_user_id)
        _non_empty("body", self.body)
        _isoish("created_at", self.created_at)
        _isoish("observed_at", self.observed_at)
        if self.reply_to_message_id is not None:
            _non_empty("reply_to_message_id", self.reply_to_message_id)
        if self.reply_to_user_id is not None:
            _non_empty("reply_to_user_id", self.reply_to_user_id)
        _require(isinstance(self.metadata, dict), "metadata must be a dict")
        self.evidence_pointer.validate()


@dataclass(frozen=True)
class InteractionRecord(Record):
    interaction_type: str
    source_user_id: str
    target_user_id: str
    message_id: str
    thread_id: str
    community_id: str
    created_at: str
    evidence_pointer: EvidencePointer

    def validate(self) -> None:
        super().validate()
        _require(self.record_type == "interaction", "interaction record_type mismatch")
        _non_empty("interaction_type", self.interaction_type)
        _non_empty("source_user_id", self.source_user_id)
        _non_empty("target_user_id", self.target_user_id)
        _non_empty("message_id", self.message_id)
        _non_empty("thread_id", self.thread_id)
        _non_empty("community_id", self.community_id)
        _isoish("created_at", self.created_at)
        self.evidence_pointer.validate()


@dataclass(frozen=True)
class TombstoneRecord(Record):
    tombstone_type: str
    target_record_type: str
    target_record_id: str
    observed_at: str
    previous_body_hash: str | None
    reason: str
    evidence_pointer: EvidencePointer

    def validate(self) -> None:
        super().validate()
        _require(self.record_type == "tombstone", "tombstone record_type mismatch")
        _non_empty("tombstone_type", self.tombstone_type)
        _non_empty("target_record_type", self.target_record_type)
        _non_empty("target_record_id", self.target_record_id)
        _isoish("observed_at", self.observed_at)
        if self.previous_body_hash is not None:
            _non_empty("previous_body_hash", self.previous_body_hash)
        _non_empty("reason", self.reason)
        self.evidence_pointer.validate()


BronzeRecord = (
    AccountRecord
    | ProfileSnapshotRecord
    | CommunityRecord
    | ThreadRecord
    | MessageRecord
    | InteractionRecord
    | TombstoneRecord
)
