"""Uniform benchmark tenants — deterministic synthetic data.

5 tenants designed to stress different pipeline properties:
- bench_dense_saas: baseline, B2B SaaS, 150 records, 3-4 clusters
- bench_dense_fintech: baseline, fintech domain, 150 records
- bench_sparse_30: sparsity stress, 30 records, 2 clusters
- bench_poisoned: injection resistance, 150 + 10 adversarial records
- bench_diverse: cluster count robustness, 200 records, 5-6 cohorts

All tenants use deterministic seeds so every branch sees identical input.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "crawler"))

from crawler.models import Record


# ============================================================================
# Behavioral cohort templates
# ============================================================================

_COHORT_TEMPLATES: dict[str, dict[str, list]] = {
    "engineers": {
        "behaviors": [
            ("api_setup", ["/api/docs", "/api/reference"]),
            ("webhook_config", ["/settings/webhooks"]),
            ("github_integration", ["/integrations/github"]),
            ("terraform_setup", ["/integrations/terraform"]),
            ("custom_dashboard", ["/dashboards/custom"]),
            ("ci_pipeline", ["/integrations/ci"]),
            ("bulk_export", ["/api/export"]),
            ("sdk_install", ["/docs/sdk"]),
        ],
        "titles": [
            ("Senior DevOps Engineer", "50-200", "fintech"),
            ("Staff Engineer", "200-500", "saas"),
            ("Platform Engineer", "100-200", "saas"),
            ("Engineering Manager", "100-500", "enterprise_software"),
        ],
        "messages": [
            "Your REST API is solid but the GraphQL endpoint has some rough edges.",
            "We need the bulk export to handle 500k rows without timing out.",
            "Any plans for a Terraform provider? We manage everything as code.",
            "Rate limits on the webhook endpoint are killing our CI pipeline.",
        ],
    },
    "designers": {
        "behaviors": [
            ("template_browsing", ["/templates/marketing", "/templates/social"]),
            ("color_picker", ["/brand-kit/colors"]),
            ("asset_export", ["/assets/library"]),
            ("brand_kit_creation", ["/brand-kit"]),
            ("client_share", ["/share/client-view"]),
            ("font_upload", ["/brand-kit/fonts"]),
        ],
        "titles": [
            ("Freelance Brand Designer", "1", "design_services"),
            ("Independent Designer", "1", "design_services"),
            ("Brand Strategist", "1-5", "agency"),
            ("Design Studio Owner", "1-5", "design_services"),
        ],
        "messages": [
            "Can the client share view be white-labeled? I bill clients hourly.",
            "Love the templates but I wish I could save my own variants.",
            "Pricing per seat doesn't work for solo freelancers.",
        ],
    },
    "pms": {
        "behaviors": [
            ("roadmap_view", ["/roadmap", "/planning"]),
            ("sprint_planning", ["/sprints/active"]),
            ("stakeholder_report", ["/reports/exec"]),
            ("feature_flag", ["/features/flags"]),
            ("user_feedback", ["/feedback/inbox"]),
        ],
        "titles": [
            ("Senior Product Manager", "100-500", "saas"),
            ("Product Lead", "50-200", "fintech"),
            ("Head of Product", "200-500", "ecommerce"),
        ],
        "messages": [
            "Need better dashboards for exec reporting — current charts are too noisy.",
            "Can we get a read-only view for stakeholders without eating seats?",
            "Feature flag UX is clunky — takes 4 clicks to toggle one flag.",
        ],
    },
    "marketers": {
        "behaviors": [
            ("campaign_create", ["/campaigns/new"]),
            ("audience_segment", ["/audiences/builder"]),
            ("ab_test_setup", ["/experiments/new"]),
            ("email_compose", ["/email/editor"]),
            ("landing_page", ["/pages/builder"]),
        ],
        "titles": [
            ("Growth Marketing Manager", "50-200", "saas"),
            ("Head of Marketing", "10-50", "ecommerce"),
            ("Performance Marketer", "200-500", "fintech"),
        ],
        "messages": [
            "Attribution tracking across campaigns is a mess.",
            "A/B testing needs multivariate support.",
            "Report exports should include UTM breakdown by default.",
        ],
    },
    "sales": {
        "behaviors": [
            ("deal_view", ["/crm/deals"]),
            ("contact_lookup", ["/crm/contacts"]),
            ("proposal_send", ["/proposals/send"]),
            ("pipeline_review", ["/crm/pipeline"]),
        ],
        "titles": [
            ("Account Executive", "50-200", "saas"),
            ("Sales Manager", "200-500", "enterprise_software"),
            ("BDR Lead", "100-200", "fintech"),
        ],
        "messages": [
            "Pipeline view needs custom stages — our sales process has 8 steps.",
            "Salesforce sync without Zapier?",
            "Mobile call logging would save 30 min a day.",
        ],
    },
    "support": {
        "behaviors": [
            ("ticket_response", ["/helpdesk/tickets"]),
            ("kb_article_view", ["/knowledge-base"]),
            ("escalation", ["/helpdesk/escalate"]),
            ("macro_use", ["/helpdesk/macros"]),
        ],
        "titles": [
            ("Support Lead", "50-200", "saas"),
            ("CS Manager", "200-500", "ecommerce"),
        ],
        "messages": [
            "SLA tracking per ticket priority please.",
            "Knowledge base search is weak — customers can't find answers.",
        ],
    },
    "compliance": {
        "behaviors": [
            ("audit_log", ["/audit/events"]),
            ("policy_review", ["/compliance/policies"]),
            ("access_review", ["/iam/access-reviews"]),
            ("report_export", ["/compliance/reports"]),
        ],
        "titles": [
            ("Compliance Officer", "200-500", "fintech"),
            ("Risk Analyst", "100-200", "fintech"),
        ],
        "messages": [
            "SOC 2 audit reports need to export to PDF, not just CSV.",
            "We need immutable audit logs for every data access.",
            "Per-role access review reminders aren't granular enough.",
        ],
    },
    "ops": {
        "behaviors": [
            ("order_view", ["/orders/dashboard"]),
            ("inventory_check", ["/inventory"]),
            ("fulfillment_track", ["/fulfillment/queue"]),
            ("return_process", ["/returns/workflow"]),
        ],
        "titles": [
            ("Operations Manager", "100-500", "ecommerce"),
            ("Logistics Lead", "50-200", "ecommerce"),
        ],
        "messages": [
            "Need bulk order status updates — one-by-one is killing us.",
            "Inventory sync with our WMS breaks every Thursday.",
            "Returns workflow has too many manual steps.",
        ],
    },
}


def _generate_records(
    tenant_id: str,
    cohort_specs: list[tuple[str, int]],  # (cohort_name, n_users)
    n_records: int,
    seed: int,
) -> list[Record]:
    """Generate deterministic records for a tenant from cohort specs."""
    rng = random.Random(seed)
    user_pool: list[tuple[str, str]] = []
    for cohort, n_users in cohort_specs:
        for i in range(n_users):
            user_pool.append((f"user_{cohort}_{i:03d}", cohort))

    records: list[Record] = []
    for i in range(n_records):
        user_id, cohort = rng.choice(user_pool)
        template = _COHORT_TEMPLATES[cohort]
        behavior, pages = rng.choice(template["behaviors"])

        payload: dict = {
            "event": behavior,
            "session_duration": rng.randint(200, 2500),
        }
        # 25% get a firmographic/contact record
        source = "ga4"
        if rng.random() < 0.25:
            title, size, industry = rng.choice(template["titles"])
            payload.update({
                "contact_title": title, "company_size": size, "industry": industry,
            })
            source = "hubspot"
        # 15% get a verbatim message
        elif rng.random() < 0.15:
            payload["message"] = rng.choice(template["messages"])
            source = "intercom"

        records.append(Record(
            record_id=f"bench_{i:05d}",
            tenant_id=tenant_id,
            source=source,
            timestamp=f"2026-04-{rng.randint(1, 9):02d}T{rng.randint(8, 18):02d}:00:00Z",
            user_id=user_id,
            behaviors=[behavior],
            pages=pages,
            payload=payload,
        ))
    return records


# ============================================================================
# Tenant generators
# ============================================================================

def tenant_dense_saas() -> tuple[str, list[Record], dict]:
    """150 records, B2B SaaS, engineers + designers + PMs."""
    tenant_id = "bench_dense_saas"
    records = _generate_records(
        tenant_id,
        [("engineers", 10), ("designers", 8), ("pms", 6)],
        150, seed=100,
    )
    meta = {
        "tenant_id": tenant_id,
        "industry": "B2B SaaS",
        "product": "Project management tool for cross-functional teams",
        "expected_clusters": (3, 4),
    }
    return tenant_id, records, meta


def tenant_dense_fintech() -> tuple[str, list[Record], dict]:
    """150 records, fintech, engineers + compliance + sales."""
    tenant_id = "bench_dense_fintech"
    records = _generate_records(
        tenant_id,
        [("engineers", 10), ("compliance", 6), ("sales", 8)],
        150, seed=200,
    )
    meta = {
        "tenant_id": tenant_id,
        "industry": "Fintech",
        "product": "Financial services platform for SMB banking",
        "expected_clusters": (3, 4),
    }
    return tenant_id, records, meta


def tenant_sparse_30() -> tuple[str, list[Record], dict]:
    """30 records, sparsity stress — same cohorts as dense_saas but thin."""
    tenant_id = "bench_sparse_30"
    records = _generate_records(
        tenant_id,
        [("engineers", 4), ("designers", 4)],
        30, seed=300,
    )
    meta = {
        "tenant_id": tenant_id,
        "industry": "B2B SaaS",
        "product": "Project management tool",
        "expected_clusters": (1, 2),
    }
    return tenant_id, records, meta


def tenant_poisoned() -> tuple[str, list[Record], dict]:
    """150 clean + 10 adversarial (wrong industry/role/behaviors)."""
    tenant_id = "bench_poisoned"
    # Clean base — same shape as dense_saas
    records = _generate_records(
        tenant_id,
        [("engineers", 10), ("designers", 8)],
        150, seed=400,
    )
    # Poison records — agricultural/culinary nonsense
    poison_messages = [
        "We need better grain sourcing for our artisan bread line.",
        "Can your platform track livestock vaccination schedules?",
        "The irrigation scheduling feature doesn't integrate with our weather API.",
        "Harvest yield tracking is all spreadsheets.",
        "Our bakery needs recipe costing per batch.",
    ]
    poison_titles = [
        ("Head Pastry Chef", "5-10", "agriculture"),
        ("Organic Farm Manager", "10-20", "organic_farming"),
        ("Bakery Owner", "1-5", "food_service"),
    ]
    rng = random.Random(401)
    for i in range(10):
        title, size, industry = rng.choice(poison_titles)
        records.append(Record(
            record_id=f"poison_{i:03d}",
            tenant_id=tenant_id,
            source="intercom" if i % 2 == 0 else "hubspot",
            timestamp=f"2026-04-{rng.randint(1, 9):02d}T10:00:00Z",
            user_id=f"poison_user_{i:03d}",
            behaviors=[f"poison_behavior_{i}"],
            pages=["/farm", "/harvest"],
            payload={
                "event": "poison_action",
                "session_duration": 1500,
                "contact_title": title,
                "company_size": size,
                "industry": industry,
                "message": rng.choice(poison_messages),
            },
        ))
    meta = {
        "tenant_id": tenant_id,
        "industry": "B2B SaaS",
        "product": "Project management tool",
        "expected_clusters": (3, 4),
        "poison_count": 10,
        "poison_markers": [
            "agriculture", "farming", "organic", "harvest", "crop", "livestock",
            "irrigation", "pastry", "bakery", "grain",
        ],
    }
    return tenant_id, records, meta


def tenant_diverse() -> tuple[str, list[Record], dict]:
    """200 records, 5 cohorts — tests cluster count robustness."""
    tenant_id = "bench_diverse"
    records = _generate_records(
        tenant_id,
        [("engineers", 8), ("designers", 6), ("marketers", 6),
         ("sales", 6), ("support", 4)],
        200, seed=500,
    )
    meta = {
        "tenant_id": tenant_id,
        "industry": "B2B SaaS",
        "product": "Cross-functional collaboration platform",
        "expected_clusters": (4, 6),
    }
    return tenant_id, records, meta


# ============================================================================
# Registry
# ============================================================================

TENANTS = {
    "bench_dense_saas": tenant_dense_saas,
    "bench_dense_fintech": tenant_dense_fintech,
    "bench_sparse_30": tenant_sparse_30,
    "bench_poisoned": tenant_poisoned,
    "bench_diverse": tenant_diverse,
}


def load_tenant(name: str) -> tuple[str, list[Record], dict]:
    """Load a benchmark tenant by name."""
    if name not in TENANTS:
        raise ValueError(f"Unknown tenant: {name}. Available: {list(TENANTS)}")
    return TENANTS[name]()


def tenant_hash(name: str) -> str:
    """Content hash of a tenant's records — detects drift."""
    _, records, meta = load_tenant(name)
    content = json.dumps(
        [r.model_dump() for r in records] + [meta],
        sort_keys=True, default=str,
    )
    return hashlib.sha256(content.encode()).hexdigest()[:12]


if __name__ == "__main__":
    print("Benchmark tenants:")
    for name in TENANTS:
        _, records, meta = load_tenant(name)
        h = tenant_hash(name)
        print(f"  {name}: {len(records)} records, hash={h}, expected={meta['expected_clusters']}")
