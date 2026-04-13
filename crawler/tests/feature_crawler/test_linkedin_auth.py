from __future__ import annotations

import json
import os
import threading
import unittest
from urllib.request import urlopen

from crawler.feature_crawler.platforms.linkedin.connector_auth import (
    LinkedInAuthClient,
    build_authorization_url,
    wait_for_callback_once,
)


class LinkedInAuthTest(unittest.TestCase):
    def test_build_authorization_url_contains_expected_params(self) -> None:
        url = build_authorization_url(
            client_id="client-123",
            redirect_uri="http://127.0.0.1:8080/callback",
            state="state-xyz",
        )
        self.assertIn("https://www.linkedin.com/oauth/v2/authorization?", url)
        self.assertIn("client_id=client-123", url)
        self.assertIn("state=state-xyz", url)
        self.assertIn("scope=openid+profile+email", url)

    def test_exchange_code_uses_form_payload(self) -> None:
        seen: dict[str, object] = {}

        def fake_fetch(url: str, payload: dict[str, object]):
            seen["url"] = url
            seen["payload"] = payload
            return {"access_token": "token-123", "expires_in": 3600}

        client = LinkedInAuthClient(
            client_id="client-123",
            client_secret="secret-456",
            redirect_uri="http://127.0.0.1:8080/callback",
            fetch_json=fake_fetch,
        )
        result = client.exchange_code(code="auth-code-1")
        self.assertEqual(result["access_token"], "token-123")
        self.assertEqual(seen["payload"]["grant_type"], "authorization_code")
        self.assertEqual(seen["payload"]["client_secret"], "secret-456")

    def test_from_env_requires_values(self) -> None:
        original = dict(os.environ)
        try:
            os.environ.pop("LINKEDIN_CLIENT_ID", None)
            os.environ.pop("LINKEDIN_CLIENT_SECRET", None)
            os.environ.pop("LINKEDIN_REDIRECT_URI", None)
            with self.assertRaises(ValueError):
                LinkedInAuthClient.from_env()
        finally:
            os.environ.clear()
            os.environ.update(original)

    def test_wait_for_callback_once_captures_query_params(self) -> None:
        result_holder: dict[str, object] = {}

        def run_server() -> None:
            result_holder["payload"] = wait_for_callback_once(port=8765, timeout_seconds=3.0)

        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        response = urlopen("http://127.0.0.1:8765/callback?code=abc123&state=state-xyz", timeout=3)
        self.assertEqual(response.status, 200)
        thread.join(timeout=3)
        payload = result_holder["payload"]
        self.assertEqual(payload.code, "abc123")
        self.assertEqual(payload.state, "state-xyz")
