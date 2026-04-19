"""Tests for Slack alerting."""

import pytest
from unittest.mock import patch, MagicMock


def test_alerter_importable():
    from persona_eval.alerting import SlackAlerter
    assert SlackAlerter is not None


def test_no_webhook_does_nothing():
    from persona_eval.alerting import SlackAlerter
    alerter = SlackAlerter(webhook_url=None)
    # Should not raise
    alerter.alert_regression("p1", {"D1": 0.5})


def test_alert_sends_payload():
    from persona_eval.alerting import SlackAlerter
    alerter = SlackAlerter(webhook_url="https://hooks.slack.com/test")
    with patch("httpx.post") as mock_post:
        alerter.alert_regression("p1", {"D1": 0.5, "D3": 0.3})
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        assert "blocks" in payload
        assert len(payload["blocks"]) >= 2


def test_alert_includes_persona_id():
    from persona_eval.alerting import SlackAlerter
    alerter = SlackAlerter(webhook_url="https://hooks.slack.com/test")
    with patch("httpx.post") as mock_post:
        alerter.alert_regression("test-persona-42", {"D1": 0.5})
        payload = mock_post.call_args[1]["json"]
        # Check persona ID appears somewhere in blocks
        blocks_text = str(payload["blocks"])
        assert "test-persona-42" in blocks_text


def test_webhook_from_env():
    from persona_eval.alerting import SlackAlerter
    with patch.dict("os.environ", {"SLACK_EVAL_WEBHOOK_URL": "https://env-webhook.com"}):
        alerter = SlackAlerter()
        assert alerter.webhook_url == "https://env-webhook.com"
