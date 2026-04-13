from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .policy import CollectionBasis, PolicyRegistry
from .records import BronzeRecord


@dataclass(frozen=True)
class CrawlTarget:
    platform: str
    community_id: str
    community_name: str
    collection_basis: CollectionBasis | None = None
    source_url: str | None = None

    @property
    def storage_key(self) -> str:
        safe_name = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in self.community_name)
        return f"{self.community_id}--{safe_name.strip('-') or 'community'}"


@dataclass(frozen=True)
class BronzeWriteResult:
    root: Path
    files_written: dict[str, Path]
    record_counts: dict[str, int]


class BronzeWriter:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, target: CrawlTarget, records: list[BronzeRecord]) -> BronzeWriteResult:
        if not records:
            raise ValueError("records must not be empty")
        run_id = records[0].crawl_run_id
        files_written: dict[str, Path] = {}
        counts: Counter[str] = Counter()
        grouped: dict[str, list[dict[str, object]]] = {}

        for record in records:
            record.validate()
            counts[record.record_type] += 1
            grouped.setdefault(record.record_type, []).append(record.to_dict())

        base_dir = self.root / target.platform / target.storage_key / run_id
        base_dir.mkdir(parents=True, exist_ok=True)

        for record_type, payloads in grouped.items():
            file_path = base_dir / f"{record_type}.jsonl"
            with file_path.open("w", encoding="utf-8") as handle:
                for payload in payloads:
                    handle.write(json.dumps(payload, sort_keys=True))
                    handle.write("\n")
            files_written[record_type] = file_path

        manifest_path = base_dir / "manifest.json"
        manifest = {
            "platform": target.platform,
            "community_id": target.community_id,
            "community_name": target.community_name,
            "collection_basis": (target.collection_basis.value if target.collection_basis else None),
            "record_counts": dict(counts),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        files_written["manifest"] = manifest_path
        return BronzeWriteResult(root=base_dir, files_written=files_written, record_counts=dict(counts))


class CrawlerRunner:
    def __init__(self, policy_registry: PolicyRegistry, writer: BronzeWriter) -> None:
        self.policy_registry = policy_registry
        self.writer = writer

    def run(self, connector: "CommunityConnector", target: CrawlTarget, *, since: str | None = None) -> BronzeWriteResult:
        self.policy_registry.assert_allowed(
            platform=target.platform,
            collection_basis=target.collection_basis,
            use_fallback=connector.uses_fallback,
        )
        records = list(connector.fetch(target=target, since=since))
        return self.writer.write(target, records)


class CommunityConnector:
    platform: str
    uses_fallback = False

    def fetch(self, target: CrawlTarget, since: str | None = None) -> list[BronzeRecord]:
        raise NotImplementedError
