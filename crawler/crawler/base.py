from __future__ import annotations

from abc import ABC, abstractmethod

from .models import Record


class Connector(ABC):
    """Base class for all connectors.

    A connector knows how to pull records from one source for one tenant.
    Real connectors handle auth refresh, retries, rate limits; mock
    connectors return fixture data so the rest of the pipeline can run
    end-to-end without external dependencies.
    """

    name: str = "base"

    @abstractmethod
    def fetch(self, tenant_id: str, since: str | None = None) -> list[Record]:
        """Fetch records for the tenant. `since` is an ISO timestamp."""
        ...
