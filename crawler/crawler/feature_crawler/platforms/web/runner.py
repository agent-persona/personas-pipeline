from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

from ...core.base import CrawlContext
from ...core.models import CrawlTarget
from ...core.sink import JsonlSink
from .connector_approved import ApprovedWebConnector
from .connector_threaded import ThreadAwareWebConnector


def register_cli(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> dict[str, callable]:
    crawl_web = subparsers.add_parser(
        "crawl-web",
        help="Crawl an approved web page and write records to local JSONL.",
    )
    crawl_web.add_argument("--url", required=True)
    crawl_web.add_argument("--output-dir", required=True)
    crawl_web.add_argument(
        "--collection-basis",
        default="public-permitted",
        choices=["owned", "consented", "public-permitted", "blocked"],
    )
    crawl_web.add_argument("--community-name", default=None)
    crawl_web.add_argument("--target-id", default=None)
    crawl_web.add_argument("--thread-aware", action="store_true")
    crawl_web.add_argument(
        "--render-mode",
        choices=["http", "playwright", "auto"],
        default="http",
    )
    crawl_web.add_argument("--since", default=None)
    crawl_web.add_argument("--until", default=None)
    crawl_web.add_argument("--allow-persona-inference", action="store_true")
    return {"crawl-web": run_cli}


def _normalize_url(raw_value: str) -> str:
    parsed = urlparse(raw_value)
    if parsed.scheme:
        return raw_value
    return Path(raw_value).expanduser().resolve().as_uri()


def run_cli(args: argparse.Namespace) -> int:
    parsed = urlparse(args.url)
    community_name = args.community_name or parsed.netloc or "local-web-source"
    target_id = args.target_id or community_name
    target = CrawlTarget(
        platform="web",
        target_id=target_id,
        url=_normalize_url(args.url),
        community_name=community_name,
        collection_basis=args.collection_basis,
        allow_persona_inference=args.allow_persona_inference,
        metadata={
            "render_mode": args.render_mode,
            "since": args.since,
            "until": args.until,
        },
    )
    connector = ThreadAwareWebConnector() if args.thread_aware else ApprovedWebConnector()
    context = CrawlContext.create()
    records = list(connector.fetch(target=target, context=context))
    sink = JsonlSink(Path(args.output_dir))
    output_path = sink.write(records)
    print(f"wrote {len(records)} records to {output_path}")
    return 0
