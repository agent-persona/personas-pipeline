from __future__ import annotations

import argparse
import json
from pathlib import Path

from crawler.feature_crawler.platforms.linkedin.headless_visible_profiles import (
    build_storage_state_from_brave,
    run_visible_profile_harvest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Headless LinkedIn visible-profile harvest")
    parser.add_argument("--connections-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--refresh-auth-state", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    connections_path = Path(args.connections_file).expanduser().resolve()
    output_path = Path(args.output_file).expanduser().resolve()
    state_path = Path(args.state_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    if args.refresh_auth_state or not state_path.exists():
        state_path.write_text(json.dumps(build_storage_state_from_brave()))
        print(f"wrote auth state to {state_path}", flush=True)

    total = run_visible_profile_harvest(
        connections_path=connections_path,
        output_path=output_path,
        state_path=state_path,
        max_profiles=args.max_profiles,
        checkpoint_every=args.checkpoint_every,
    )
    print(f"wrote {total} profiles to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
