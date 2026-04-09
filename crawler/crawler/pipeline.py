from __future__ import annotations

from .base import Connector
from .connectors import GA4Connector, HubspotConnector, IntercomConnector
from .models import Record

# Default day-1 connector set
DEFAULT_CONNECTORS: list[Connector] = [
    GA4Connector(),
    IntercomConnector(),
    HubspotConnector(),
]


def fetch_all(
    tenant_id: str,
    connectors: list[Connector] | None = None,
    since: str | None = None,
) -> list[Record]:
    """Fetch records from every connector and concatenate the results."""
    if connectors is None:
        connectors = DEFAULT_CONNECTORS
    all_records: list[Record] = []
    for connector in connectors:
        all_records.extend(connector.fetch(tenant_id, since=since))
    return all_records
