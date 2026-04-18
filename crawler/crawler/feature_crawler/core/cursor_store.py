from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CursorState:
    value: str
    metadata: dict[str, Any]


class JsonCursorStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, key: str) -> CursorState | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = str(payload.get("value") or "").strip()
        if not value:
            return None
        metadata = payload.get("metadata")
        return CursorState(value=value, metadata=metadata if isinstance(metadata, dict) else {})

    def save(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> Path:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "key": key,
                    "value": value,
                    "metadata": metadata or {},
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path

    def _path_for(self, key: str) -> Path:
        safe_parts = [self._slug(part) for part in key.split("/") if part.strip()]
        filename = "__".join(safe_parts) or "cursor"
        return self.root / f"{filename}.json"

    def _slug(self, value: str) -> str:
        chars = [char.lower() if char.isalnum() else "-" for char in value]
        slug = "".join(chars).strip("-")
        while "--" in slug:
            slug = slug.replace("--", "-")
        return slug or "cursor"
