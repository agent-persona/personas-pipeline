from __future__ import annotations

from crawler.base import Connector
from crawler.models import Record

# Mock HubSpot contact records — provide firmographic context.
_HUBSPOT_CONTACTS = [
    # Engineers
    ("user_eng_a", "technical_role", "Senior DevOps Engineer", "50-200", "fintech"),
    ("user_eng_b", "technical_role", "Staff Engineer", "200-500", "saas"),
    ("user_eng_c", "technical_role", "Platform Engineer", "100-200", "saas"),
    ("user_eng_d", "technical_role", "Engineering Manager", "100-500", "enterprise_software"),

    # Designers
    ("user_des_a", "creative_role", "Freelance Brand Designer", "1", "design_services"),
    ("user_des_b", "creative_role", "Independent Designer", "1", "design_services"),
    ("user_des_c", "creative_role", "Brand Strategist", "1-5", "design_services"),
    ("user_des_d", "creative_role", "Design Studio Owner", "1-5", "design_services"),
]


class HubspotConnector(Connector):
    name = "hubspot"

    def fetch(self, tenant_id: str, since: str | None = None) -> list[Record]:
        records: list[Record] = []
        for i, (user_id, behavior, title, size, industry) in enumerate(_HUBSPOT_CONTACTS):
            records.append(
                Record(
                    record_id=f"hubspot_{i:03d}",
                    tenant_id=tenant_id,
                    source=self.name,
                    timestamp="2026-03-28T09:00:00Z",
                    user_id=user_id,
                    behaviors=[behavior],
                    pages=[],
                    payload={
                        "contact_title": title,
                        "company_size": size,
                        "industry": industry,
                    },
                )
            )
        return records
