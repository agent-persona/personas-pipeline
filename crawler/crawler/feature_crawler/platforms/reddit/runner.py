from __future__ import annotations

import argparse
import os
from pathlib import Path

from ...core.base import CrawlContext
from ...core.cursor_store import JsonCursorStore
from ...core.models import CrawlTarget
from ...core.sink import JsonlSink
from .connector_api import RedditApiConnector


def register_cli(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> dict[str, callable]:
    crawl_reddit = subparsers.add_parser(
        "crawl-reddit",
        help="Crawl a subreddit via Reddit OAuth or public .json pages and write canonical JSONL.",
    )
    crawl_reddit.add_argument("--subreddit", required=True)
    crawl_reddit.add_argument("--output-dir", required=True)
    crawl_reddit.add_argument("--sort", choices=["new", "hot", "top", "rising"], default="new")
    crawl_reddit.add_argument("--limit", type=int, default=25)
    crawl_reddit.add_argument("--comment-limit", type=int, default=128)
    crawl_reddit.add_argument("--since", default=None)
    crawl_reddit.add_argument("--until", default=None)
    crawl_reddit.add_argument(
        "--collection-basis",
        default="public-permitted",
        choices=["owned", "consented", "public-permitted", "blocked"],
    )
    crawl_reddit.add_argument("--community-name", default=None)
    crawl_reddit.add_argument("--auth-mode", choices=["auto", "oauth", "public-json"], default="auto")
    crawl_reddit.add_argument("--client-id-env", default="REDDIT_CLIENT_ID")
    crawl_reddit.add_argument("--client-secret-env", default="REDDIT_CLIENT_SECRET")
    crawl_reddit.add_argument("--user-agent-env", default="REDDIT_USER_AGENT")
    crawl_reddit.add_argument("--cursor-store", default=None)
    crawl_reddit.add_argument("--cursor-key", default=None)
    return {"crawl-reddit": run_cli}


def run_cli(args: argparse.Namespace) -> int:
    effective_since = args.since
    cursor_store = JsonCursorStore(Path(args.cursor_store)) if args.cursor_store else None
    cursor_key = args.cursor_key or f"reddit/{args.subreddit}/{args.sort}"
    if cursor_store and not effective_since:
        existing = cursor_store.load(cursor_key)
        if existing is not None:
            effective_since = existing.value

    target = CrawlTarget(
        platform="reddit",
        target_id=args.subreddit,
        url=f"https://www.reddit.com/r/{args.subreddit}/",
        community_name=args.community_name or args.subreddit,
        collection_basis=args.collection_basis,
        metadata={
            "subreddit": args.subreddit,
            "sort": args.sort,
            "limit": args.limit,
            "comment_limit": args.comment_limit,
            "until": args.until,
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
            metadata={"platform": "reddit", "subreddit": args.subreddit, "sort": args.sort},
        )
        print(f"updated cursor {cursor_key} at {cursor_path}")
    return 0


def _build_connector(args: argparse.Namespace) -> RedditApiConnector:
    if args.auth_mode == "oauth":
        return RedditApiConnector.from_env(
            client_id_env=args.client_id_env,
            client_secret_env=args.client_secret_env,
            user_agent_env=args.user_agent_env,
        )
    if args.auth_mode == "public-json":
        return RedditApiConnector.from_public_json(
            user_agent_env=args.user_agent_env,
        )

    has_oauth_creds = bool(os.environ.get(args.client_id_env) and os.environ.get(args.client_secret_env))
    if has_oauth_creds:
        return RedditApiConnector.from_env(
            client_id_env=args.client_id_env,
            client_secret_env=args.client_secret_env,
            user_agent_env=args.user_agent_env,
        )
    return RedditApiConnector.from_public_json(
        user_agent_env=args.user_agent_env,
    )
