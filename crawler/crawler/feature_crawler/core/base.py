from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable

from .models import CrawlTarget, Record
from .policy import SourcePolicy, assert_target_allowed


@dataclass(frozen=True, slots=True)
class CrawlContext:
    crawl_run_id: str
    observed_at: str

    @classmethod
    def create(cls) -> "CrawlContext":
        now = datetime.now(UTC)
        return cls(
            crawl_run_id=now.strftime("run_%Y%m%d_%H%M%S"),
            observed_at=now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        )


class CommunityConnector(ABC):
    platform: str

    def validate_target(self, target: CrawlTarget) -> SourcePolicy:
        return assert_target_allowed(target)

    @abstractmethod
    def fetch(
        self,
        target: CrawlTarget,
        context: CrawlContext,
        since: str | None = None,
    ) -> Iterable[Record]:
        raise NotImplementedError
