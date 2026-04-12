from __future__ import annotations

from crawler.base import Connector
from crawler.models import Record

# Mock Intercom messages — verbatim quotes that synthesis can lift into
# the persona's sample_quotes field.
_INTERCOM_MESSAGES = [
    # Engineers
    ("user_eng_a", "graphql_question", "api_feedback",
     "Your REST API is solid but the GraphQL endpoint has some rough edges. Plans to improve the schema?"),
    ("user_eng_b", "consolidation_pain", "integration_request",
     "We're trying to reduce context-switching. If I could get Jira, GitHub, and your tool in one dashboard that would be huge."),
    ("user_eng_c", "automation_request", "reporting_automation",
     "Is there a way to set up automated reports that go to Slack? We need deployment velocity metrics for standup."),
    ("user_eng_d", "api_question", "api_feedback",
     "What are the rate limits on the bulk export endpoint? We hit them in CI."),

    # Designers
    ("user_des_a", "white_label_request", "white_labeling",
     "I bill clients hourly so anything that saves me 10 minutes per project is worth real money. Can your client share view be white-labeled?"),
    ("user_des_b", "client_experience_concern", "client_experience",
     "My clients are non-technical. They get confused by your edit interface. Is there a presentation mode where they only see the design and can comment?"),
    ("user_des_c", "feature_request", "feature_request",
     "Love the templates. Saves me hours when starting a new client brand. But I wish I could save my own template variants for reuse across clients."),
    ("user_des_d", "pricing_complaint", "pricing",
     "Pricing per seat doesn't work for me. I work alone but I have 15 clients. I'd pay more for unlimited share links instead of a per-seat plan."),

    # Product Managers
    ("user_pm_a", "roadmap_feedback", "product_feedback",
     "We need a way to tie customer requests to roadmap items. Right now I'm copying Intercom links into Notion which is insane."),
    ("user_pm_b", "prioritization_pain", "product_feedback",
     "The scoring model for feature prioritization is too rigid. I need weighted RICE or ICE that I can customize per quarter."),
    ("user_pm_c", "analytics_request", "product_feedback",
     "Can you surface feature adoption rates directly in the roadmap view? I shouldn't have to export to Amplitude to see if anyone used what we shipped."),

    # Sales Reps
    ("user_sales_a", "demo_feedback", "sales_feedback",
     "I need a sandbox environment I can spin up per prospect in under 2 minutes. Right now it takes me 20 minutes to prep a demo."),
    ("user_sales_b", "pipeline_pain", "sales_feedback",
     "Your CRM sync with Salesforce is laggy. Deals I close in your tool don't show in SFDC for hours and my manager thinks I'm behind."),
    ("user_sales_c", "prospect_request", "sales_feedback",
     "Can I get a notification when a prospect views the demo I shared? Knowing they opened it is the difference between a cold and warm follow-up."),

    # Marketing Managers
    ("user_mkt_a", "campaign_pain", "marketing_feedback",
     "The email campaign builder is decent but I can't A/B test subject lines without exporting to Mailchimp. That round-trip kills our velocity."),
    ("user_mkt_b", "content_request", "marketing_feedback",
     "We need better content calendar views. I'm managing 40 pieces a month and the list view doesn't cut it — give me a kanban or a proper calendar."),
    ("user_mkt_c", "attribution_question", "marketing_feedback",
     "UTM attribution is broken for multi-touch. A lead touches 3 campaigns before converting and you only credit the last one. We need multi-touch attribution."),

    # Customer Success
    ("user_cs_a", "onboarding_pain", "cs_feedback",
     "Onboarding checklists reset when a customer switches browsers. I've had three enterprise clients complain this week — it makes us look amateur."),
    ("user_cs_b", "health_score_request", "cs_feedback",
     "The health score algorithm weights login frequency too heavily. A customer can log in daily and still churn if they're not hitting value milestones."),
    ("user_cs_c", "escalation_pain", "cs_feedback",
     "When a ticket gets escalated to engineering, I lose visibility. I need a shared view so I can tell the customer what's happening without pinging Slack."),

    # Executives
    ("user_exec_a", "reporting_pain", "executive_feedback",
     "I need a single dashboard that shows engineering velocity, cost per feature, and customer satisfaction side by side. Not three separate reports I have to stitch together."),
    ("user_exec_b", "roi_question", "executive_feedback",
     "How do I justify renewing at this price point? Give me an ROI report I can show the board — time saved, defects avoided, something concrete."),
    ("user_exec_c", "visibility_request", "executive_feedback",
     "I want to see which teams are blocked and why without attending their standups. A cross-team dependency view would save me 5 hours a week."),

    # Project Coordinators
    ("user_coord_a", "resource_pain", "operations_feedback",
     "Resource allocation across 6 projects is a nightmare in spreadsheets. I need to see who's overbooked and drag-and-drop to rebalance."),
    ("user_coord_b", "timeline_request", "operations_feedback",
     "The Gantt chart doesn't auto-shift downstream tasks when an upstream milestone slips. I end up manually updating 30 dates every time something moves."),
    ("user_coord_c", "status_pain", "operations_feedback",
     "Collecting weekly status updates from 8 project leads by Friday is like herding cats. Can you auto-generate a status digest from task completions?"),
]


class IntercomConnector(Connector):
    name = "intercom"

    def fetch(self, tenant_id: str, since: str | None = None) -> list[Record]:
        records: list[Record] = []
        for i, (user_id, behavior, topic, message) in enumerate(_INTERCOM_MESSAGES):
            records.append(
                Record(
                    record_id=f"intercom_{i:03d}",
                    tenant_id=tenant_id,
                    source=self.name,
                    timestamp="2026-04-02T10:00:00Z",
                    user_id=user_id,
                    behaviors=[behavior],
                    pages=[],
                    payload={
                        "message": message,
                        "topic": topic,
                    },
                )
            )
        return records
