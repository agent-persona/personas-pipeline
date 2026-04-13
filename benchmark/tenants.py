"""Uniform benchmark tenants — deterministic synthetic data.

10 tenants designed to produce ~50 clusters (personas) total and stress
different pipeline properties:

  baseline domain coverage (high record volume, diverse cohorts):
    - bench_mega_saas       500 rec, 6 cohorts -> ~6 clusters
    - bench_mega_fintech    500 rec, 5 cohorts -> ~5 clusters
    - bench_mega_ecommerce  500 rec, 5 cohorts -> ~5 clusters
    - bench_dense_devtools  350 rec, 4 cohorts -> ~4 clusters

  stress scenarios:
    - bench_sparse_30        30 rec, 2 cohorts -> ~2 clusters (sparsity)
    - bench_sparse_60        60 rec, 3 cohorts -> ~3 clusters (mid-sparsity)
    - bench_poisoned        350 rec + 15 poison -> ~5 clusters (injection)
    - bench_heavy_tail      300 rec, 80/20 split -> ~4 clusters (imbalance)
    - bench_single_cohort   200 rec, 1 cohort   -> ~1 cluster  (degeneracy)

  wide variety:
    - bench_diverse         500 rec, 8 cohorts  -> ~8 clusters

Target: 43-50 personas total.
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
# Behavioral cohort templates — reusable across tenants
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
            ("call_log", ["/crm/activities"]),
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
            ("csat_review", ["/analytics/csat"]),
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
            ("soc2_checklist", ["/compliance/soc2"]),
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
    "security": {
        "behaviors": [
            ("threat_review", ["/security/threats"]),
            ("incident_response", ["/security/incidents"]),
            ("policy_config", ["/security/policies"]),
            ("vuln_scan_review", ["/security/scans"]),
            ("access_anomaly", ["/security/anomalies"]),
        ],
        "titles": [
            ("Security Analyst", "200-500", "fintech"),
            ("CISO", "500+", "enterprise_software"),
            ("Security Engineer", "100-500", "saas"),
        ],
        "messages": [
            "We need automated threat correlation across log sources.",
            "Incident response runbooks should trigger from alerts automatically.",
            "Per-tenant isolation is non-negotiable for regulated clients.",
        ],
    },
    "data": {
        "behaviors": [
            ("query_builder", ["/analytics/queries"]),
            ("dashboard_create", ["/analytics/dashboards"]),
            ("dataset_import", ["/data/import"]),
            ("model_train", ["/ml/training"]),
            ("pipeline_monitor", ["/data/pipelines"]),
        ],
        "titles": [
            ("Data Engineer", "100-500", "saas"),
            ("Analytics Lead", "50-200", "ecommerce"),
            ("ML Engineer", "200-500", "fintech"),
        ],
        "messages": [
            "SQL query builder hits timeouts above 10M rows.",
            "Need Airflow integration for scheduled pipelines.",
            "Dashboard refresh is too slow — users give up waiting.",
        ],
    },
}


def _generate_records(
    tenant_id: str,
    cohort_specs: list[tuple[str, int]],
    n_records: int,
    seed: int,
    user_behavior_weighting: float = 1.0,
) -> list[Record]:
    """Generate deterministic records for a tenant."""
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
        source = "ga4"
        if rng.random() < 0.25:
            title, size, industry = rng.choice(template["titles"])
            payload.update({
                "contact_title": title, "company_size": size, "industry": industry,
            })
            source = "hubspot"
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

def tenant_mega_saas() -> tuple[str, list[Record], dict]:
    """500 records, 6 cohorts — flagship B2B SaaS scenario."""
    tid = "bench_mega_saas"
    records = _generate_records(
        tid,
        [("engineers", 12), ("designers", 8), ("pms", 8),
         ("marketers", 8), ("sales", 6), ("support", 4)],
        500, seed=100,
    )
    return tid, records, {
        "tenant_id": tid, "industry": "B2B SaaS",
        "product": "Cross-functional project management platform",
        "expected_clusters": (5, 6),
    }


def tenant_mega_fintech() -> tuple[str, list[Record], dict]:
    """500 records, 5 cohorts — fintech stack with compliance focus."""
    tid = "bench_mega_fintech"
    records = _generate_records(
        tid,
        [("engineers", 12), ("compliance", 8), ("sales", 8),
         ("security", 6), ("data", 6)],
        500, seed=200,
    )
    return tid, records, {
        "tenant_id": tid, "industry": "Fintech",
        "product": "Financial services platform with compliance tooling",
        "expected_clusters": (4, 5),
    }


def tenant_mega_ecommerce() -> tuple[str, list[Record], dict]:
    """500 records, 5 cohorts — e-commerce marketplace."""
    tid = "bench_mega_ecommerce"
    records = _generate_records(
        tid,
        [("engineers", 10), ("marketers", 10), ("sales", 8),
         ("ops", 8), ("support", 6)],
        500, seed=300,
    )
    return tid, records, {
        "tenant_id": tid, "industry": "E-commerce",
        "product": "Merchant operations platform for multi-channel retail",
        "expected_clusters": (4, 5),
    }


def tenant_dense_devtools() -> tuple[str, list[Record], dict]:
    """350 records, 4 cohorts — developer tools company."""
    tid = "bench_dense_devtools"
    records = _generate_records(
        tid,
        [("engineers", 12), ("data", 8), ("pms", 6), ("security", 4)],
        350, seed=400,
    )
    return tid, records, {
        "tenant_id": tid, "industry": "Developer Tools",
        "product": "Observability and developer productivity platform",
        "expected_clusters": (3, 4),
    }


def tenant_sparse_30() -> tuple[str, list[Record], dict]:
    """30 records — sparsity stress."""
    tid = "bench_sparse_30"
    records = _generate_records(
        tid, [("engineers", 4), ("designers", 4)],
        30, seed=500,
    )
    return tid, records, {
        "tenant_id": tid, "industry": "B2B SaaS",
        "product": "Project management tool",
        "expected_clusters": (1, 2),
    }


def tenant_sparse_60() -> tuple[str, list[Record], dict]:
    """60 records — mid-sparsity."""
    tid = "bench_sparse_60"
    records = _generate_records(
        tid, [("engineers", 5), ("marketers", 4), ("sales", 4)],
        60, seed=550,
    )
    return tid, records, {
        "tenant_id": tid, "industry": "B2B SaaS",
        "product": "Sales and marketing automation",
        "expected_clusters": (2, 3),
    }


def tenant_poisoned() -> tuple[str, list[Record], dict]:
    """350 clean + 15 adversarial (agriculture/culinary nonsense)."""
    tid = "bench_poisoned"
    records = _generate_records(
        tid,
        [("engineers", 10), ("designers", 6), ("pms", 6), ("sales", 4)],
        350, seed=600,
    )
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
    rng = random.Random(601)
    for i in range(15):
        title, size, industry = rng.choice(poison_titles)
        records.append(Record(
            record_id=f"poison_{i:03d}",
            tenant_id=tid,
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
    return tid, records, {
        "tenant_id": tid, "industry": "B2B SaaS",
        "product": "Project management tool",
        "expected_clusters": (4, 5),
        "poison_count": 15,
        "poison_markers": [
            "agriculture", "farming", "organic", "harvest", "crop", "livestock",
            "irrigation", "pastry", "bakery", "grain",
        ],
    }


def tenant_heavy_tail() -> tuple[str, list[Record], dict]:
    """300 records, heavy-tailed — 70% engineers, 10% each of 3 others."""
    tid = "bench_heavy_tail"
    # Build cohort specs with 70/10/10/10 weighting (by user count)
    records = _generate_records(
        tid,
        [("engineers", 20), ("designers", 3), ("sales", 3), ("support", 3)],
        300, seed=700,
    )
    return tid, records, {
        "tenant_id": tid, "industry": "B2B SaaS",
        "product": "Developer-heavy project management platform",
        "expected_clusters": (3, 4),
    }


def tenant_single_cohort() -> tuple[str, list[Record], dict]:
    """200 records, 1 cohort — degeneracy test."""
    tid = "bench_single_cohort"
    records = _generate_records(
        tid, [("engineers", 15)],
        200, seed=800,
    )
    return tid, records, {
        "tenant_id": tid, "industry": "Developer Tools",
        "product": "API platform",
        "expected_clusters": (1, 1),
    }


def tenant_diverse() -> tuple[str, list[Record], dict]:
    """500 records, 8 cohorts — maximum variety."""
    tid = "bench_diverse"
    records = _generate_records(
        tid,
        [("engineers", 8), ("designers", 6), ("pms", 6), ("marketers", 6),
         ("sales", 6), ("support", 4), ("data", 6), ("security", 4)],
        500, seed=900,
    )
    return tid, records, {
        "tenant_id": tid, "industry": "B2B SaaS",
        "product": "Horizontal platform serving many functions",
        "expected_clusters": (6, 8),
    }


# ============================================================================
# Registry
# ============================================================================

TENANTS = {
    "bench_mega_saas": tenant_mega_saas,
    "bench_mega_fintech": tenant_mega_fintech,
    "bench_mega_ecommerce": tenant_mega_ecommerce,
    "bench_dense_devtools": tenant_dense_devtools,
    "bench_sparse_30": tenant_sparse_30,
    "bench_sparse_60": tenant_sparse_60,
    "bench_poisoned": tenant_poisoned,
    "bench_heavy_tail": tenant_heavy_tail,
    "bench_single_cohort": tenant_single_cohort,
    "bench_diverse": tenant_diverse,
}


def load_tenant(name: str) -> tuple[str, list[Record], dict]:
    if name not in TENANTS:
        raise ValueError(f"Unknown tenant: {name}. Available: {list(TENANTS)}")
    return TENANTS[name]()


def tenant_hash(name: str) -> str:
    _, records, meta = load_tenant(name)
    content = json.dumps(
        [r.model_dump() for r in records] + [meta],
        sort_keys=True, default=str,
    )
    return hashlib.sha256(content.encode()).hexdigest()[:12]


if __name__ == "__main__":
    print(f"{'Tenant':<25} {'Records':>8} {'Hash':>14} {'Expected clusters':<20}")
    print("-" * 75)
    for name in TENANTS:
        _, records, meta = load_tenant(name)
        h = tenant_hash(name)
        ec = meta["expected_clusters"]
        print(f"  {name:<25} {len(records):>6} {h:>14}    {ec}")
