from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen


def _clean_space(value: str | None) -> str | None:
    if value is None:
        return None
    clean = " ".join(value.split()).strip()
    return clean or None


def build_authorization_url(
    *,
    client_id: str,
    redirect_uri: str,
    scope: str = "openid profile email",
    state: str,
    code_challenge: str | None = None,
    code_challenge_method: str = "S256",
) -> str:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state,
    }
    if code_challenge:
        params["code_challenge"] = code_challenge
        params["code_challenge_method"] = code_challenge_method
    return f"https://www.linkedin.com/oauth/v2/authorization?{urlencode(params)}"


@dataclass(slots=True)
class LinkedInAuthClient:
    client_id: str
    client_secret: str
    redirect_uri: str
    token_endpoint: str = "https://www.linkedin.com/oauth/v2/accessToken"
    fetch_json: Any | None = None

    @classmethod
    def from_env(
        cls,
        *,
        client_id_env: str = "LINKEDIN_CLIENT_ID",
        client_secret_env: str = "LINKEDIN_CLIENT_SECRET",
        redirect_uri_env: str = "LINKEDIN_REDIRECT_URI",
        redirect_uri: str | None = None,
    ) -> "LinkedInAuthClient":
        client_id = _clean_space(os.environ.get(client_id_env))
        client_secret = _clean_space(os.environ.get(client_secret_env))
        resolved_redirect_uri = redirect_uri or _clean_space(os.environ.get(redirect_uri_env))
        if not client_id or not client_secret or not resolved_redirect_uri:
            raise ValueError(
                f"missing LinkedIn OAuth config; expected ${client_id_env}, ${client_secret_env}, and redirect URI"
            )
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=resolved_redirect_uri,
        )

    def exchange_code(self, *, code: str, code_verifier: str | None = None) -> dict[str, Any]:
        payload = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
        }
        if code_verifier:
            payload["code_verifier"] = code_verifier
        else:
            payload["client_secret"] = self.client_secret

        if self.fetch_json is not None:
            return self.fetch_json(self.token_endpoint, payload)

        request = Request(
            self.token_endpoint,
            method="POST",
            data=urlencode(payload).encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        )
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))


@dataclass(slots=True)
class LinkedInCallbackPayload:
    code: str | None
    state: str | None
    error: str | None
    raw_params: dict[str, list[str]]


def wait_for_callback_once(
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    path: str = "/callback",
    timeout_seconds: float = 180.0,
) -> LinkedInCallbackPayload:
    captured: dict[str, LinkedInCallbackPayload] = {}
    done = threading.Event()

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != path:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"not found")
                return
            params = parse_qs(parsed.query, keep_blank_values=True)
            payload = LinkedInCallbackPayload(
                code=_first(params, "code"),
                state=_first(params, "state"),
                error=_first(params, "error"),
                raw_params=params,
            )
            captured["payload"] = payload
            body = b"LinkedIn auth received. You can return to the terminal."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            done.set()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    server = HTTPServer((host, port), CallbackHandler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    finished = done.wait(timeout_seconds)
    server.server_close()
    thread.join(timeout=1.0)
    if not finished:
        raise TimeoutError(f"timed out waiting for LinkedIn callback on http://{host}:{port}{path}")
    return captured["payload"]


def _first(params: dict[str, list[str]], key: str) -> str | None:
    values = params.get(key) or []
    return values[0] if values else None
