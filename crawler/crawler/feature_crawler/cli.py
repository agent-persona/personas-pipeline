from __future__ import annotations

import argparse

from .platforms.discord.runner import register_cli as register_discord_cli
from .platforms.linkedin.runner import register_cli as register_linkedin_cli
from .platforms.reddit.runner import register_cli as register_reddit_cli
from .platforms.web.runner import register_cli as register_web_cli


def build_parser() -> tuple[argparse.ArgumentParser, dict[str, callable]]:
    parser = argparse.ArgumentParser(description="Crawler feature CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    handlers: dict[str, callable] = {}
    for register in (register_web_cli, register_discord_cli, register_reddit_cli, register_linkedin_cli):
        handlers.update(register(subparsers))
    return parser, handlers


def main(argv: list[str] | None = None) -> int:
    parser, handlers = build_parser()
    args = parser.parse_args(argv)
    handler = handlers.get(args.command)
    if handler is None:
        parser.error(f"unknown command: {args.command}")
        return 2
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
