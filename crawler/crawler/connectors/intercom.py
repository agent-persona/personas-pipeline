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
