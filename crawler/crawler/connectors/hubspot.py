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

    # Product Managers
    ("user_pm_a", "product_role", "Senior Product Manager", "200-500", "saas"),
    ("user_pm_b", "product_role", "Product Manager", "100-200", "fintech"),
    ("user_pm_c", "product_role", "Director of Product", "500-1000", "saas"),

    # Sales Reps
    ("user_sales_a", "sales_role", "Account Executive", "200-500", "saas"),
    ("user_sales_b", "sales_role", "Sales Development Rep", "100-200", "saas"),
    ("user_sales_c", "sales_role", "Senior AE", "500-1000", "enterprise_software"),

    # Marketing Managers
    ("user_mkt_a", "marketing_role", "Growth Marketing Manager", "100-200", "saas"),
    ("user_mkt_b", "marketing_role", "Content Marketing Lead", "200-500", "saas"),
    ("user_mkt_c", "marketing_role", "Demand Gen Manager", "200-500", "martech"),

    # Customer Success
    ("user_cs_a", "cs_role", "Customer Success Manager", "200-500", "saas"),
    ("user_cs_b", "cs_role", "Onboarding Specialist", "100-200", "saas"),
    ("user_cs_c", "cs_role", "Senior CSM", "500-1000", "enterprise_software"),

    # Executives
    ("user_exec_a", "executive_role", "VP of Engineering", "200-500", "saas"),
    ("user_exec_b", "executive_role", "CTO", "50-200", "fintech"),
    ("user_exec_c", "executive_role", "COO", "100-200", "saas"),

    # Project Coordinators
    ("user_coord_a", "operations_role", "Project Coordinator", "200-500", "professional_services"),
    ("user_coord_b", "operations_role", "PMO Analyst", "500-1000", "enterprise_software"),
    ("user_coord_c", "operations_role", "Program Manager", "200-500", "consulting"),
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
