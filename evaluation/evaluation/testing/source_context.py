from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class SourceContext(BaseModel):
    id: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)
    chunks: list[str] = Field(default_factory=list)
    conversation_transcript: list[dict[str, Any]] = Field(default_factory=list)
    extra_data: dict[str, Any] = Field(default_factory=dict)

    def get_chunks(self, max_chunk_size: int = 512) -> list[str]:
        if self.chunks:
            return self.chunks
        words = self.text.split()
        return [
            " ".join(words[i : i + max_chunk_size])
            for i in range(0, len(words), max_chunk_size)
        ]
