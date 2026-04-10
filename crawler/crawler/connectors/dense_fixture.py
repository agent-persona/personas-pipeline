"""Dense synthetic tenant fixture for experiment 3.06 (sparse-data ablation).

Generates a configurable number of records across multiple user cohorts
and behavioral patterns. Designed to be downsampled to test groundedness
collapse at various data densities.

Default: 200 records across 5 cohorts (engineers, designers, marketers,
sales, support). Each cohort has distinct behavioral signatures so
segmentation produces meaningful clusters at every density tier.
"""

from __future__ import annotations

import random
from typing import Sequence

from crawler.base import Connector
from crawler.models import Record

# Behavioral templates per cohort. Each tuple:
# (behavior, pages, payload_extras, duration_range)
_COHORTS: dict[str, list[tuple[str, list[str], dict, tuple[int, int]]]] = {
    "eng": [
        ("api_setup", ["/api/docs", "/api/reference"], {"topic": "api"}, (800, 2400)),
        ("webhook_config", ["/settings/webhooks"], {"topic": "automation"}, (600, 1800)),
        ("github_integration", ["/integrations/github"], {"topic": "integration"}, (400, 1200)),
        ("terraform_setup", ["/integrations/terraform"], {"topic": "infra"}, (500, 1500)),
        ("custom_dashboard", ["/dashboards/custom"], {"topic": "analytics"}, (900, 2500)),
        ("ci_pipeline", ["/integrations/ci", "/api/webhooks"], {"topic": "ci"}, (700, 2000)),
        ("bulk_export", ["/api/export", "/data/export"], {"topic": "data"}, (300, 900)),
        ("sdk_install", ["/docs/sdk", "/api/quickstart"], {"topic": "sdk"}, (400, 1100)),
    ],
    "des": [
        ("template_browsing", ["/templates/marketing", "/templates/social"], {"topic": "templates"}, (1000, 2500)),
        ("color_picker", ["/brand-kit/colors"], {"topic": "branding"}, (300, 800)),
        ("asset_export", ["/assets/library"], {"topic": "assets"}, (200, 600)),
        ("brand_kit_creation", ["/brand-kit"], {"topic": "branding"}, (800, 2000)),
        ("client_share", ["/share/client-view"], {"topic": "sharing"}, (200, 500)),
        ("font_upload", ["/brand-kit/fonts"], {"topic": "typography"}, (150, 400)),
    ],
    "mkt": [
        ("campaign_create", ["/campaigns/new", "/campaigns/builder"], {"topic": "campaigns"}, (1200, 3000)),
        ("audience_segment", ["/audiences/builder"], {"topic": "targeting"}, (800, 2200)),
        ("ab_test_setup", ["/experiments/new"], {"topic": "testing"}, (600, 1500)),
        ("report_view", ["/analytics/campaigns"], {"topic": "reporting"}, (400, 1000)),
        ("email_compose", ["/email/editor"], {"topic": "email"}, (900, 2400)),
        ("landing_page", ["/pages/builder"], {"topic": "pages"}, (700, 1800)),
    ],
    "sales": [
        ("deal_view", ["/crm/deals"], {"topic": "deals"}, (300, 900)),
        ("contact_lookup", ["/crm/contacts"], {"topic": "contacts"}, (200, 600)),
        ("proposal_send", ["/proposals/send"], {"topic": "proposals"}, (400, 1200)),
        ("pipeline_review", ["/crm/pipeline"], {"topic": "pipeline"}, (500, 1500)),
        ("call_log", ["/crm/activities"], {"topic": "activities"}, (150, 500)),
    ],
    "support": [
        ("ticket_response", ["/helpdesk/tickets"], {"topic": "support"}, (300, 800)),
        ("kb_article_view", ["/knowledge-base"], {"topic": "docs"}, (200, 600)),
        ("escalation", ["/helpdesk/escalate"], {"topic": "escalation"}, (100, 400)),
        ("csat_review", ["/analytics/csat"], {"topic": "metrics"}, (250, 700)),
        ("macro_use", ["/helpdesk/macros"], {"topic": "efficiency"}, (100, 300)),
    ],
}

# Intercom-style messages per cohort (for verbatim quote grounding)
_MESSAGES: dict[str, list[str]] = {
    "eng": [
        "Your REST API is solid but the GraphQL endpoint has some rough edges.",
        "We need the bulk export to handle 500k rows without timing out.",
        "Any plans for a Terraform provider? We manage everything as code.",
        "Rate limits on the webhook endpoint are killing our CI pipeline.",
        "The SDK doesn't support async/await yet — that's a blocker for us.",
    ],
    "des": [
        "Can the client share view be white-labeled? I bill clients hourly.",
        "Love the templates but I wish I could save my own variants for reuse.",
        "Pricing per seat doesn't work for solo freelancers with many clients.",
        "The color picker needs to support Pantone codes for print work.",
    ],
    "mkt": [
        "We need better attribution tracking across campaigns.",
        "A/B testing setup is clunky — can we get multivariate support?",
        "Would love to connect our CDP directly to audience segments.",
        "Report exports should include UTM breakdown by default.",
    ],
    "sales": [
        "Pipeline view needs custom stages — our sales process has 8 steps.",
        "Can we get Salesforce sync that doesn't require Zapier?",
        "Call logging from mobile would save my team 30 minutes a day.",
    ],
    "support": [
        "We need SLA tracking per ticket priority level.",
        "The knowledge base search is too basic — customers can't find answers.",
        "Macro suggestions based on ticket content would be a game-changer.",
    ],
}

# HubSpot-style firmographic data per cohort
_FIRMOGRAPHIC: dict[str, list[tuple[str, str, str]]] = {
    "eng": [
        ("Senior DevOps Engineer", "50-200", "fintech"),
        ("Staff Engineer", "200-500", "saas"),
        ("Platform Engineer", "100-200", "saas"),
        ("Engineering Manager", "100-500", "enterprise_software"),
        ("Backend Developer", "10-50", "healthtech"),
    ],
    "des": [
        ("Freelance Brand Designer", "1", "design_services"),
        ("Independent Designer", "1", "design_services"),
        ("Brand Strategist", "1-5", "agency"),
        ("Design Studio Owner", "1-5", "design_services"),
    ],
    "mkt": [
        ("Growth Marketing Manager", "50-200", "saas"),
        ("Head of Marketing", "10-50", "ecommerce"),
        ("Performance Marketer", "200-500", "fintech"),
        ("Content Strategist", "50-200", "media"),
    ],
    "sales": [
        ("Account Executive", "50-200", "saas"),
        ("Sales Manager", "200-500", "enterprise_software"),
        ("BDR Lead", "100-200", "fintech"),
    ],
    "support": [
        ("Support Lead", "50-200", "saas"),
        ("CS Manager", "200-500", "ecommerce"),
        ("Help Desk Analyst", "10-50", "healthtech"),
    ],
}


class DenseFixtureConnector(Connector):
    """Generates a dense synthetic tenant with configurable record count.

    Records are deterministically seeded so downsampling is reproducible.
    """

    name = "dense_fixture"

    def __init__(self, n_records: int = 200, seed: int = 42) -> None:
        self.n_records = n_records
        self.seed = seed

    def fetch(self, tenant_id: str, since: str | None = None) -> list[Record]:
        rng = random.Random(self.seed)
        cohort_names = list(_COHORTS.keys())
        records: list[Record] = []

        # Distribute users across cohorts
        users_per_cohort = max(3, self.n_records // (len(cohort_names) * 4))

        user_pool: list[tuple[str, str]] = []  # (user_id, cohort)
        for cohort in cohort_names:
            for u in range(users_per_cohort):
                user_pool.append((f"user_{cohort}_{u:03d}", cohort))

        for i in range(self.n_records):
            user_id, cohort = rng.choice(user_pool)
            behaviors = _COHORTS[cohort]
            behavior, pages, extras, (dur_lo, dur_hi) = rng.choice(behaviors)
            duration = rng.randint(dur_lo, dur_hi)

            payload = {
                "event": behavior,
                "session_duration": duration,
                **extras,
            }

            # ~20% of records get a firmographic entry
            if rng.random() < 0.20:
                firmo = rng.choice(_FIRMOGRAPHIC.get(cohort, [("Unknown", "?", "?")]))
                payload["contact_title"] = firmo[0]
                payload["company_size"] = firmo[1]
                payload["industry"] = firmo[2]

            # ~15% get a verbatim message
            if rng.random() < 0.15:
                msgs = _MESSAGES.get(cohort, [])
                if msgs:
                    payload["message"] = rng.choice(msgs)

            records.append(
                Record(
                    record_id=f"dense_{i:05d}",
                    tenant_id=tenant_id,
                    source=self.name,
                    timestamp=f"2026-04-{rng.randint(1,9):02d}T{rng.randint(8,18):02d}:00:00Z",
                    user_id=user_id,
                    behaviors=[behavior],
                    pages=pages,
                    payload=payload,
                )
            )

        return records


def downsample(
    records: Sequence[Record],
    n: int,
    seed: int = 42,
) -> list[Record]:
    """Deterministically downsample records to n items."""
    if n >= len(records):
        return list(records)
    rng = random.Random(seed)
    return rng.sample(list(records), n)
