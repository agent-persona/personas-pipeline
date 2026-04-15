from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
from urllib.parse import urlparse

from ...core.base import CrawlContext
from ...core.cursor_store import JsonCursorStore
from ...core.models import CrawlTarget
from ...core.sink import JsonlSink
from .connector_auth import LinkedInAuthClient, build_authorization_url, wait_for_callback_once
from .connector_browser import LinkedInBrowserConnector
from .connector_official import LinkedInOfficialConnector
from .connector_profile import LinkedInProfileConnector
from .connector_vendor import LinkedInVendorConnector


def register_cli(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> dict[str, callable]:
    crawl_linkedin = subparsers.add_parser(
        "crawl-linkedin",
        help="Crawl a LinkedIn profile page and write canonical JSONL.",
    )
    crawl_linkedin.add_argument("--url", required=True)
    crawl_linkedin.add_argument("--output-dir", required=True)
    crawl_linkedin.add_argument(
        "--mode",
        choices=["official-oidc", "public-html", "session-html", "session-browser", "apify", "brightdata", "linkdapi"],
        default="public-html",
    )
    crawl_linkedin.add_argument(
        "--collection-basis",
        default="consented",
        choices=["owned", "consented", "public-permitted", "blocked"],
    )
    crawl_linkedin.add_argument("--community-name", default=None)
    crawl_linkedin.add_argument("--target-id", default=None)
    crawl_linkedin.add_argument("--allow-persona-inference", action="store_true")
    crawl_linkedin.add_argument("--cookie-env", default="LINKEDIN_COOKIE")
    crawl_linkedin.add_argument("--li-at-env", default="LINKEDIN_SESSION_COOKIE_LI_AT")
    crawl_linkedin.add_argument("--jsessionid-env", default="LINKEDIN_SESSION_COOKIE_JSESSIONID")
    crawl_linkedin.add_argument("--access-token-env", default="LINKEDIN_ACCESS_TOKEN")
    crawl_linkedin.add_argument("--since", default=None)
    crawl_linkedin.add_argument("--until", default=None)
    crawl_linkedin.add_argument("--scope", default="profile")
    crawl_linkedin.add_argument("--activity-limit", type=int, default=25)
    crawl_linkedin.add_argument("--comment-limit", type=int, default=64)
    crawl_linkedin.add_argument("--network-limit", type=int, default=50)
    crawl_linkedin.add_argument("--max-pages", type=int, default=3)
    crawl_linkedin.add_argument("--cursor-store", default=None)
    crawl_linkedin.add_argument("--cursor-key", default=None)
    auth_url = subparsers.add_parser(
        "linkedin-auth-url",
        help="Print a LinkedIn OIDC authorization URL for member consent.",
    )
    auth_url.add_argument("--client-id", required=True)
    auth_url.add_argument("--redirect-uri", required=True)
    auth_url.add_argument("--scope", default="openid profile email")
    auth_url.add_argument("--state", default=None)
    exchange = subparsers.add_parser(
        "linkedin-exchange-code",
        help="Exchange a LinkedIn authorization code for tokens.",
    )
    exchange.add_argument("--code", required=True)
    exchange.add_argument("--redirect-uri", required=True)
    exchange.add_argument("--client-id-env", default="LINKEDIN_CLIENT_ID")
    exchange.add_argument("--client-secret-env", default="LINKEDIN_CLIENT_SECRET")
    exchange.add_argument("--redirect-uri-env", default="LINKEDIN_REDIRECT_URI")
    exchange.add_argument("--code-verifier", default=None)

    callback = subparsers.add_parser(
        "linkedin-wait-for-callback",
        help="Wait for one LinkedIn OAuth callback on a local HTTP endpoint.",
    )
    callback.add_argument("--host", default="127.0.0.1")
    callback.add_argument("--port", type=int, default=8080)
    callback.add_argument("--path", default="/callback")
    callback.add_argument("--timeout-seconds", type=float, default=180.0)
    callback.add_argument("--exchange", action="store_true")
    callback.add_argument("--client-id-env", default="LINKEDIN_CLIENT_ID")
    callback.add_argument("--client-secret-env", default="LINKEDIN_CLIENT_SECRET")
    callback.add_argument("--redirect-uri-env", default="LINKEDIN_REDIRECT_URI")
    callback.add_argument("--code-verifier", default=None)
    return {
        "crawl-linkedin": run_cli,
        "linkedin-auth-url": run_auth_url_cli,
        "linkedin-exchange-code": run_exchange_code_cli,
        "linkedin-wait-for-callback": run_wait_for_callback_cli,
    }


def _normalize_url(raw_value: str) -> str:
    parsed = urlparse(raw_value)
    if parsed.scheme:
        return raw_value
    return Path(raw_value).expanduser().resolve().as_uri()


def run_cli(args: argparse.Namespace) -> int:
    url = _normalize_url(args.url)
    parsed = urlparse(url)
    fallback_name = Path(parsed.path).stem if parsed.scheme == "file" else (parsed.path.rstrip("/").split("/")[-1] or "linkedin-profile")
    target_id = args.target_id or fallback_name
    effective_since = args.since
    cursor_store = JsonCursorStore(Path(args.cursor_store)) if args.cursor_store else None
    cursor_key = args.cursor_key or f"linkedin/{target_id}/{args.mode}/{args.scope}"
    if cursor_store and not effective_since:
        existing = cursor_store.load(cursor_key)
        if existing is not None:
            effective_since = existing.value
    scopes = {item.strip().lower() for item in str(args.scope).split(",") if item.strip()}
    target = CrawlTarget(
        platform="linkedin",
        target_id=target_id,
        url=url,
        community_name=args.community_name or target_id,
        collection_basis=args.collection_basis,
        allow_persona_inference=args.allow_persona_inference,
        metadata={
            "since": effective_since,
            "until": args.until,
            "include_posts": "activity" in scopes or "posts" in scopes,
            "include_network": "network" in scopes or "connections" in scopes,
            "post_limit": args.activity_limit,
            "comment_limit": args.comment_limit,
            "network_limit": args.network_limit,
            "page_limit": args.max_pages,
            "scope": sorted(scopes),
        },
    )
    connector = _build_connector(args)
    context = CrawlContext.create()
    records = list(connector.fetch(target=target, context=context, since=effective_since))
    sink = JsonlSink(Path(args.output_dir))
    output_path = sink.write(records)
    print(f"wrote {len(records)} records to {output_path}")
    if cursor_store:
        cursor_values = [
            value
            for value in (
                getattr(record, "created_at", None) or getattr(record, "observed_at", None)
                for record in records
            )
            if value is not None
        ]
        latest = max(cursor_values, default=context.observed_at)
        cursor_path = cursor_store.save(
            cursor_key,
            latest,
            metadata={"platform": "linkedin", "target_id": target_id, "mode": args.mode, "scope": sorted(scopes)},
        )
        print(f"updated cursor {cursor_key} at {cursor_path}")
    return 0


def _build_connector(args: argparse.Namespace):
    if args.mode == "official-oidc":
        return LinkedInOfficialConnector.from_env(access_token_env=args.access_token_env)
    if args.mode in {"apify", "brightdata", "linkdapi"}:
        return LinkedInVendorConnector.from_env(vendor=args.mode)
    if args.mode == "session-browser":
        return LinkedInBrowserConnector.from_env(
            cookie_env=args.cookie_env,
            li_at_env=args.li_at_env,
            jsessionid_env=args.jsessionid_env,
        )
    return LinkedInProfileConnector.from_env(
        mode=args.mode,
        cookie_env=args.cookie_env,
        li_at_env=args.li_at_env,
        jsessionid_env=args.jsessionid_env,
    )


def run_auth_url_cli(args: argparse.Namespace) -> int:
    state = args.state or secrets.token_urlsafe(24)
    print(build_authorization_url(client_id=args.client_id, redirect_uri=args.redirect_uri, scope=args.scope, state=state))
    return 0


def run_exchange_code_cli(args: argparse.Namespace) -> int:
    client = LinkedInAuthClient.from_env(
        client_id_env=args.client_id_env,
        client_secret_env=args.client_secret_env,
        redirect_uri_env=args.redirect_uri_env,
        redirect_uri=args.redirect_uri,
    )
    payload = client.exchange_code(code=args.code, code_verifier=args.code_verifier)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_wait_for_callback_cli(args: argparse.Namespace) -> int:
    payload = wait_for_callback_once(
        host=args.host,
        port=args.port,
        path=args.path,
        timeout_seconds=args.timeout_seconds,
    )
    if args.exchange:
        if not payload.code:
            raise RuntimeError("callback did not include an authorization code")
        client = LinkedInAuthClient.from_env(
            client_id_env=args.client_id_env,
            client_secret_env=args.client_secret_env,
            redirect_uri_env=args.redirect_uri_env,
            redirect_uri=f"http://{args.host}:{args.port}{args.path}",
        )
        token_payload = client.exchange_code(code=payload.code, code_verifier=args.code_verifier)
        print(json.dumps({"callback": payload.raw_params, "token": token_payload}, indent=2, sort_keys=True))
        return 0
    print(
        json.dumps(
            {
                "code": payload.code,
                "state": payload.state,
                "error": payload.error,
                "raw_params": payload.raw_params,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
