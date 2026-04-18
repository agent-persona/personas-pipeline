from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from .models import Record

BronzeLike = object

_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ai": (" ai ", " llm ", " gpt ", " claude ", " model "),
    "automation": (" automation ", " automate ", " workflow "),
    "community": (" community ", " forum ", " discord "),
    "data": (" data ", " analytics ", " pipeline "),
    "design": (" design ", " designer ", " ux ", " ui "),
    "engineering": (" engineer ", " engineering ", " developer ", " software "),
    "gaming": (" game ", " gaming ", " roblox ", " fortnite "),
    "growth": (" growth ", " acquisition ", " retention "),
    "marketing": (" marketing ", " brand ", " campaign "),
    "product": (" product ", " roadmap ", " feature "),
    "sales": (" sales ", " revenue ", " pipeline "),
}

_ROLE_PATTERNS: dict[str, tuple[str, ...]] = {
    "role_creator": (" creator ", " youtuber ", " influencer "),
    "role_designer": (" designer ", " ux ", " ui ", " design "),
    "role_engineer": (" engineer ", " developer ", " software ", " architect "),
    "role_founder": (" founder ", " co-founder ", " entrepreneur "),
    "role_marketer": (" marketer ", " marketing ", " growth "),
    "role_product": (" product manager ", " product ", " pm "),
    "role_recruiter": (" recruiter ", " talent ", " hiring "),
    "role_sales": (" account executive ", " bdr ", " seller ", " sales "),
}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "this",
    "to",
    "we",
    "with",
    "you",
    "your",
}


def bronze_to_flat(bronze_records: Iterable[BronzeLike], tenant_id: str) -> list[Record]:
    """Convert vendored crawler records into the flat contract segmentation reads."""
    flattened: list[Record] = []
    for bronze_record in bronze_records:
        flattened.extend(_flatten_payload(_coerce_payload(bronze_record), tenant_id))
    return flattened


def load_run_jsonl(run_path: str | Path, tenant_id: str) -> list[Record]:
    """Load one feature_crawler JSONL file or a run directory and flatten it."""
    payloads = _load_payloads(Path(run_path))
    return bronze_to_flat(payloads, tenant_id=tenant_id)


def _load_payloads(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        return _read_jsonl(path)
    if not path.exists():
        raise FileNotFoundError(path)

    payloads: list[dict[str, Any]] = []
    for jsonl_path in sorted(path.rglob("*.jsonl")):
        payloads.extend(_read_jsonl(jsonl_path))
    if not payloads:
        raise ValueError(f"no JSONL records found under {path}")
    return payloads


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            payloads.append(json.loads(stripped))
    return payloads


def _coerce_payload(record: BronzeLike) -> dict[str, Any]:
    if isinstance(record, dict):
        return record
    if is_dataclass(record):
        return asdict(record)
    raise TypeError(f"unsupported bronze record: {type(record)!r}")


def _flatten_payload(payload: dict[str, Any], tenant_id: str) -> list[Record]:
    record_type = str(payload.get("record_type") or "")
    if record_type == "message":
        return [_message_to_record(payload, tenant_id)]
    if record_type == "interaction":
        return [_interaction_to_record(payload, tenant_id)]
    if record_type == "profile_snapshot":
        return [_profile_to_record(payload, tenant_id)]
    if record_type == "account":
        return [_account_to_record(payload, tenant_id)]
    if record_type == "thread":
        return [_thread_to_record(payload, tenant_id)]
    if record_type in {"community", "tombstone"}:
        return []
    raise ValueError(f"unsupported record_type: {record_type!r}")


def _message_to_record(payload: dict[str, Any], tenant_id: str) -> Record:
    body = str(payload.get("body") or "")
    metadata = dict(payload.get("metadata") or {})
    behaviors = [
        "posted_message",
        *_topic_tags(body),
    ]
    if payload.get("reply_to_message_id"):
        behaviors.append("replied_to_message")
    if "?" in body:
        behaviors.append("asked_question")
    if _has_link(body):
        behaviors.append("shared_link")
    if len(_tokenize(body)) >= 40:
        behaviors.append("long_form_post")
    if body.count("\n") >= 2:
        behaviors.append("structured_post")

    message_id = str(payload.get("message_id") or "")
    pages = [_conversation_path(payload)]
    if source_path := _path_from_evidence(payload):
        pages.append(source_path)

    payload_copy = dict(payload)
    payload_copy["metadata"] = metadata

    return Record(
        record_id=_stable_id(payload, "message", message_id),
        tenant_id=tenant_id,
        source=str(payload.get("platform") or "unknown"),
        timestamp=_none_if_blank(payload.get("created_at") or payload.get("observed_at")),
        user_id=_none_if_blank(payload.get("author_platform_user_id")),
        behaviors=_unique(behaviors),
        pages=_unique([page for page in pages if page]),
        payload=payload_copy,
    )


def _interaction_to_record(payload: dict[str, Any], tenant_id: str) -> Record:
    interaction_type = _slug(str(payload.get("interaction_type") or "interaction"))
    behaviors = ["interaction", f"interaction_{interaction_type}"]
    if interaction_type == "reply":
        behaviors.append("replied_to_user")
    if "mention" in interaction_type:
        behaviors.append("mentioned_user")

    return Record(
        record_id=_stable_id(
            payload,
            "interaction",
            f"{payload.get('message_id')}-{payload.get('source_user_id')}-{interaction_type}",
        ),
        tenant_id=tenant_id,
        source=str(payload.get("platform") or "unknown"),
        timestamp=_none_if_blank(payload.get("created_at")),
        user_id=_none_if_blank(payload.get("source_user_id")),
        behaviors=_unique(behaviors),
        pages=[_conversation_path(payload)],
        payload=dict(payload),
    )


def _profile_to_record(payload: dict[str, Any], tenant_id: str) -> Record:
    fields = dict(payload.get("fields") or {})
    user_id = _none_if_blank(payload.get("platform_user_id"))
    behaviors = ["profile_snapshot", *_profile_behaviors(fields)]
    pages = [_profile_path(payload, fields)]
    if source_path := _path_from_evidence(payload):
        pages.append(source_path)

    return Record(
        record_id=_stable_id(payload, "profile_snapshot", f"{user_id}-{payload.get('snapshot_at')}"),
        tenant_id=tenant_id,
        source=str(payload.get("platform") or "unknown"),
        timestamp=_none_if_blank(payload.get("snapshot_at")),
        user_id=user_id,
        behaviors=_unique(behaviors),
        pages=_unique([page for page in pages if page]),
        payload=dict(payload),
    )


def _account_to_record(payload: dict[str, Any], tenant_id: str) -> Record:
    user_id = _none_if_blank(payload.get("platform_user_id"))
    username = str(payload.get("username") or "")
    behaviors = ["account_observed", *_topic_tags(username)]

    return Record(
        record_id=_stable_id(payload, "account", str(user_id or payload.get("username") or "unknown")),
        tenant_id=tenant_id,
        source=str(payload.get("platform") or "unknown"),
        timestamp=_none_if_blank(payload.get("first_observed_at")),
        user_id=user_id,
        behaviors=_unique(behaviors),
        pages=[],
        payload=dict(payload),
    )


def _thread_to_record(payload: dict[str, Any], tenant_id: str) -> Record:
    title = str(payload.get("title") or "")
    behaviors = ["started_thread", *_topic_tags(title)]
    if "?" in title:
        behaviors.append("asked_question")

    pages = [_conversation_path(payload)]
    if source_path := _path_from_evidence(payload):
        pages.append(source_path)

    return Record(
        record_id=_stable_id(payload, "thread", str(payload.get("thread_id") or "unknown")),
        tenant_id=tenant_id,
        source=str(payload.get("platform") or "unknown"),
        timestamp=_none_if_blank(payload.get("created_at") or payload.get("observed_at")),
        user_id=_none_if_blank(payload.get("author_platform_user_id")),
        behaviors=_unique(behaviors),
        pages=_unique([page for page in pages if page]),
        payload=dict(payload),
    )


def _profile_behaviors(fields: dict[str, Any]) -> list[str]:
    text = _normalize_text(" ".join(_string_values(fields)))
    behaviors = ["active_profile"]
    if fields.get("experience"):
        behaviors.append("has_experience")
    if fields.get("activity"):
        behaviors.append("active_poster")
    behaviors.extend(_topic_tags(text))
    for behavior, patterns in _ROLE_PATTERNS.items():
        if any(pattern in text for pattern in patterns):
            behaviors.append(behavior)
    return behaviors


def _topic_tags(text: str) -> list[str]:
    normalized = _normalize_text(text)
    behaviors: list[str] = []
    for topic, patterns in _TOPIC_KEYWORDS.items():
        if any(pattern in normalized for pattern in patterns):
            behaviors.append(f"topic_{topic}")
    return behaviors


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(_string_values(item))
        return values
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(_string_values(item))
        return values
    return []


def _normalize_text(text: str) -> str:
    return " " + re.sub(r"\s+", " ", text).lower() + " "


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in _STOPWORDS]


def _has_link(text: str) -> bool:
    return "http://" in text or "https://" in text or "www." in text


def _conversation_path(payload: dict[str, Any]) -> str:
    platform = _slug(str(payload.get("platform") or "unknown"))
    community_id = _slug(str(payload.get("community_id") or "global"))
    thread_id = _slug(str(payload.get("thread_id") or payload.get("message_id") or "record"))
    return f"/{platform}/{community_id}/{thread_id}"


def _profile_path(payload: dict[str, Any], fields: dict[str, Any]) -> str:
    platform = _slug(str(payload.get("platform") or "unknown"))
    identifier = _slug(
        str(
            fields.get("public_identifier")
            or payload.get("platform_user_id")
            or payload.get("username")
            or "profile"
        )
    )
    return f"/{platform}/profile/{identifier}"


def _path_from_evidence(payload: dict[str, Any]) -> str | None:
    evidence_pointer = payload.get("evidence_pointer")
    if not isinstance(evidence_pointer, dict):
        return None
    source_url = str(evidence_pointer.get("source_url") or "")
    parsed = urlparse(source_url)
    if not parsed.scheme or not parsed.path or parsed.path == "/":
        return None
    return parsed.path


def _stable_id(payload: dict[str, Any], prefix: str, suffix: str) -> str:
    platform = _slug(str(payload.get("platform") or "unknown"))
    crawl_run_id = _slug(str(payload.get("crawl_run_id") or "run"))
    return f"{platform}:{prefix}:{crawl_run_id}:{_slug(suffix or 'record')}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unknown"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _none_if_blank(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
