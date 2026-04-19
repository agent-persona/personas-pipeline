"""Slack alerting for eval regressions."""

from __future__ import annotations

import os


class SlackAlerter:
    """Sends alerts to Slack on eval regressions."""

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or os.environ.get("SLACK_EVAL_WEBHOOK_URL")

    def alert_regression(self, persona_id: str, regressions: dict[str, float], suite: str = "persona") -> None:
        if not self.webhook_url:
            return

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Persona Eval Regression Detected"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Persona:* `{persona_id}` | *Suite:* `{suite}`",
                },
            },
        ]

        regression_lines = [f"  {dim_id}: dropped by {drop:.2f}" for dim_id, drop in sorted(regressions.items())]
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Regressions:*\n```\n" + "\n".join(regression_lines) + "\n```",
            },
        })

        self._send({"blocks": blocks})

    def _send(self, payload: dict) -> None:
        import httpx
        if self.webhook_url:
            try:
                httpx.post(self.webhook_url, json=payload, timeout=10)
            except Exception:
                pass  # Alerting should never crash the monitoring pipeline
