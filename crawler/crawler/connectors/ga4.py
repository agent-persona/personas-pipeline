from __future__ import annotations

from crawler.base import Connector
from crawler.models import Record

# Mock GA4 events. Eight distinct user types for a B2B SaaS project
# management tool, so segmentation produces 8 behaviorally distinct clusters.
_GA4_EVENTS = [
    # Engineers — API-heavy, integrations, CI/CD
    ("user_eng_a", "api_setup", ["/settings/integrations", "/api/docs"], 2340),
    ("user_eng_a", "webhook_config", ["/settings/webhooks"], 1560),
    ("user_eng_a", "github_integration", ["/integrations/github"], 980),
    ("user_eng_b", "api_setup", ["/api/docs", "/api/reference"], 1820),
    ("user_eng_b", "custom_dashboard", ["/dashboards/custom"], 1980),
    ("user_eng_b", "team_invite", ["/team/manage"], 420),
    ("user_eng_c", "webhook_config", ["/settings/webhooks", "/integrations/slack"], 1560),
    ("user_eng_c", "api_setup", ["/api/docs"], 1240),
    ("user_eng_c", "terraform_setup", ["/integrations/terraform"], 870),
    ("user_eng_d", "github_integration", ["/integrations/github"], 660),
    ("user_eng_d", "api_setup", ["/api/docs"], 920),

    # Designers — templates, branding, client sharing
    ("user_des_a", "template_browsing", ["/templates/marketing"], 1820),
    ("user_des_a", "color_picker", ["/brand-kit/colors"], 540),
    ("user_des_a", "asset_export", ["/assets/library"], 380),
    ("user_des_b", "client_share", ["/share/client-view"], 280),
    ("user_des_b", "template_browsing", ["/templates/marketing"], 1450),
    ("user_des_b", "comment_threading", ["/share/comments"], 920),
    ("user_des_c", "template_browsing", ["/templates/marketing"], 2100),
    ("user_des_c", "color_picker", ["/brand-kit/colors"], 720),
    ("user_des_c", "brand_kit_creation", ["/brand-kit"], 1340),
    ("user_des_d", "asset_export", ["/assets/library"], 220),
    ("user_des_d", "template_browsing", ["/templates/marketing"], 1890),

    # Product Managers — roadmaps, prioritization, analytics
    ("user_pm_a", "roadmap_planning", ["/roadmap/board", "/roadmap/timeline"], 3200),
    ("user_pm_a", "feature_prioritization", ["/roadmap/scoring"], 2100),
    ("user_pm_a", "analytics_review", ["/analytics/funnel"], 1800),
    ("user_pm_b", "roadmap_planning", ["/roadmap/board"], 2800),
    ("user_pm_b", "feature_prioritization", ["/roadmap/scoring", "/roadmap/votes"], 1950),
    ("user_pm_b", "release_notes", ["/releases/changelog"], 1200),
    ("user_pm_c", "roadmap_planning", ["/roadmap/timeline"], 2600),
    ("user_pm_c", "analytics_review", ["/analytics/funnel", "/analytics/retention"], 2400),
    ("user_pm_c", "feature_prioritization", ["/roadmap/scoring"], 1700),

    # Sales Reps — pipeline, demos, CRM
    ("user_sales_a", "pipeline_management", ["/pipeline/deals", "/pipeline/forecast"], 1400),
    ("user_sales_a", "demo_setup", ["/demos/sandbox", "/demos/templates"], 980),
    ("user_sales_a", "crm_sync", ["/integrations/salesforce"], 420),
    ("user_sales_b", "pipeline_management", ["/pipeline/deals"], 1600),
    ("user_sales_b", "demo_setup", ["/demos/sandbox"], 1100),
    ("user_sales_b", "prospect_tracking", ["/pipeline/prospects"], 880),
    ("user_sales_c", "demo_setup", ["/demos/sandbox", "/demos/templates"], 1320),
    ("user_sales_c", "pipeline_management", ["/pipeline/forecast"], 1050),
    ("user_sales_c", "crm_sync", ["/integrations/salesforce", "/integrations/hubspot"], 560),

    # Marketing Managers — campaigns, content, UTM analytics
    ("user_mkt_a", "campaign_builder", ["/campaigns/email", "/campaigns/templates"], 2200),
    ("user_mkt_a", "content_calendar", ["/content/calendar", "/content/drafts"], 1800),
    ("user_mkt_a", "utm_analytics", ["/analytics/utm", "/analytics/attribution"], 1400),
    ("user_mkt_b", "campaign_builder", ["/campaigns/email"], 2050),
    ("user_mkt_b", "content_calendar", ["/content/calendar"], 1650),
    ("user_mkt_b", "landing_page_builder", ["/campaigns/landing-pages"], 2400),
    ("user_mkt_c", "content_calendar", ["/content/calendar", "/content/drafts"], 1900),
    ("user_mkt_c", "campaign_builder", ["/campaigns/email", "/campaigns/social"], 2300),
    ("user_mkt_c", "utm_analytics", ["/analytics/utm"], 1100),

    # Customer Success — onboarding, health scores, ticket triage
    ("user_cs_a", "onboarding_flow", ["/customers/onboarding", "/customers/checklist"], 1600),
    ("user_cs_a", "health_score_review", ["/customers/health", "/customers/risk"], 2100),
    ("user_cs_a", "ticket_triage", ["/support/queue", "/support/escalations"], 980),
    ("user_cs_b", "onboarding_flow", ["/customers/onboarding"], 1450),
    ("user_cs_b", "health_score_review", ["/customers/health"], 1900),
    ("user_cs_b", "renewal_prep", ["/customers/renewals"], 1200),
    ("user_cs_c", "ticket_triage", ["/support/queue"], 1100),
    ("user_cs_c", "onboarding_flow", ["/customers/onboarding", "/customers/checklist"], 1700),
    ("user_cs_c", "health_score_review", ["/customers/risk"], 2000),

    # Executives — dashboards, ROI reports, board decks
    ("user_exec_a", "executive_dashboard", ["/dashboards/executive", "/dashboards/kpi"], 3600),
    ("user_exec_a", "roi_report", ["/reports/roi", "/reports/cost-analysis"], 2800),
    ("user_exec_a", "board_deck_export", ["/reports/export", "/reports/board-deck"], 1200),
    ("user_exec_b", "executive_dashboard", ["/dashboards/executive"], 3100),
    ("user_exec_b", "roi_report", ["/reports/roi"], 2500),
    ("user_exec_b", "team_performance", ["/reports/team-velocity"], 1800),
    ("user_exec_c", "executive_dashboard", ["/dashboards/kpi"], 2900),
    ("user_exec_c", "board_deck_export", ["/reports/board-deck"], 1400),
    ("user_exec_c", "roi_report", ["/reports/cost-analysis"], 2200),

    # Project Coordinators — resource planning, timelines, status
    ("user_coord_a", "resource_planning", ["/projects/resources", "/projects/capacity"], 1800),
    ("user_coord_a", "timeline_management", ["/projects/gantt", "/projects/milestones"], 2400),
    ("user_coord_a", "status_update", ["/projects/status", "/projects/standup"], 600),
    ("user_coord_b", "resource_planning", ["/projects/resources"], 1650),
    ("user_coord_b", "timeline_management", ["/projects/gantt"], 2100),
    ("user_coord_b", "dependency_tracking", ["/projects/dependencies"], 1400),
    ("user_coord_c", "timeline_management", ["/projects/milestones"], 2200),
    ("user_coord_c", "status_update", ["/projects/standup", "/projects/status"], 750),
    ("user_coord_c", "resource_planning", ["/projects/capacity"], 1500),
]


class GA4Connector(Connector):
    name = "ga4"

    def fetch(self, tenant_id: str, since: str | None = None) -> list[Record]:
        records: list[Record] = []
        for i, (user_id, behavior, pages, duration) in enumerate(_GA4_EVENTS):
            records.append(
                Record(
                    record_id=f"ga4_{i:03d}",
                    tenant_id=tenant_id,
                    source=self.name,
                    timestamp="2026-04-01T12:00:00Z",
                    user_id=user_id,
                    behaviors=[behavior],
                    pages=pages,
                    payload={
                        "event": behavior,
                        "session_duration": duration,
                    },
                )
            )
        return records
