from __future__ import annotations

from crawler.base import Connector
from crawler.models import Record

# Mock GA4 events. Two distinct user types: power-user engineers
# and freelance designers, so segmentation produces two clusters.
_GA4_EVENTS = [
    # Engineers
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

    # Designers
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
