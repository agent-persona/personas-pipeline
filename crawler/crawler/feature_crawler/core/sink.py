from __future__ import annotations

import json
from pathlib import Path

from .models import Record, record_to_dict


class JsonlSink:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, records: list[Record]) -> Path:
        if not records:
            raise ValueError("No records to write.")
        payloads = [record_to_dict(record) for record in records]
        first = payloads[0]
        output_dir = self.root.joinpath(*self._output_parts(payloads))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{first['crawl_run_id']}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload, sort_keys=True))
                handle.write("\n")
        return output_path

    def _output_parts(self, payloads: list[dict[str, object]]) -> list[str]:
        first = payloads[0]
        platform = str(first.get("platform") or "unknown")
        date_fragment = str(first.get("observed_at") or "")[:10] or "unknown-date"
        if platform == "discord":
            return self._discord_output_parts(payloads, date_fragment)
        community_id = str(first.get("community_id") or "global")
        return [platform, community_id, date_fragment]

    def _discord_output_parts(
        self,
        payloads: list[dict[str, object]],
        date_fragment: str,
    ) -> list[str]:
        community_records = [
            payload
            for payload in payloads
            if payload.get("record_type") == "community"
        ]
        server_record = next(
            (
                payload
                for payload in community_records
                if payload.get("community_type") == "server"
            ),
            None,
        )
        channel_records = {
            str(payload.get("community_id") or ""): payload
            for payload in community_records
            if payload.get("community_type") in {"channel", "forum"}
        }

        server_id = str(server_record.get("community_id") or "") if server_record else ""
        server_name = str(
            (server_record or {}).get("community_name") or server_id or "discord"
        )
        referenced_channel_ids = {
            str(payload.get("community_id") or "")
            for payload in payloads
            if payload.get("record_type") in {"thread", "message", "interaction"}
            and payload.get("community_id")
        }
        if server_id:
            referenced_channel_ids.discard(server_id)
        if not referenced_channel_ids and len(channel_records) == 1:
            referenced_channel_ids = set(channel_records)

        if len(referenced_channel_ids) == 1:
            channel_id = next(iter(referenced_channel_ids))
            channel_name = str(
                channel_records.get(channel_id, {}).get("community_name") or channel_id
            )
            channel_folder = f"{_slug(channel_name)}_{channel_id}"
        elif len(referenced_channel_ids) > 1:
            channel_folder = "multi-channel"
        else:
            channel_folder = "unknown-channel"

        return [
            "discord",
            _slug(server_name),
            channel_folder,
            *_discord_date_parts(date_fragment),
        ]


def _discord_date_parts(date_fragment: str) -> list[str]:
    parts = date_fragment.split("-")
    if len(parts) != 3:
        return [date_fragment]
    year, month, day = parts
    return [str(int(month)), str(int(day)), year]


def _slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in value]
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "unknown"
