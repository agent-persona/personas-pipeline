from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "generate_mock_conversations.py"
SPEC = importlib.util.spec_from_file_location("mockgen", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {MODULE_PATH}")
mockgen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mockgen
SPEC.loader.exec_module(mockgen)


DISCORD_SERVERS = [
    "creative-lab",
    "maker-circle",
    "design-garden",
    "story-engine",
    "pixel-forge",
    "studio-commons",
    "worldbuilders",
    "sound-camp",
    "craft-club",
    "future-fans",
]

REDDIT_COMMUNITIES = [
    "creativepractice",
    "makersessions",
    "designcritiqueclub",
    "storybuildinglab",
    "pixelcraft",
    "studioworkflows",
    "worldbuildingnotes",
    "musicproductionlogs",
    "gardenprojects",
    "scifireadsclub",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate large mock Discord and Reddit networks.")
    parser.add_argument("--output-dir", default="feature_crawler/data")
    parser.add_argument("--date", default="2026-04-09")
    parser.add_argument("--runs-per-channel", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--platform",
        action="append",
        dest="platforms",
        help="Repeat to limit to discord and/or reddit.",
    )
    return parser.parse_args()


def build_args(
    *,
    output_dir: str,
    date: str,
    seed: int,
    platform: str,
    community_name: str,
    channel_names: list[str],
    observed_at: str,
    runs_per_channel: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=output_dir,
        date=date,
        pages_per_channel=runs_per_channel,
        channels_per_platform=len(channel_names),
        seed=seed,
        channel_names=channel_names,
        crawl_run_id=None,
        observed_at=observed_at,
        discord_server_name=community_name,
        discord_server_id="mock",
        discord_channel_id=None,
        platforms=[platform],
    )


def generate_platform(
    *,
    output_dir: str,
    date: str,
    seed: int,
    platform: str,
    communities: list[str],
    channels: list[str],
    runs_per_channel: int,
) -> int:
    count = 0
    for index, community in enumerate(communities, start=1):
        hour = 8 + index
        observed_at = f"{date}T{hour:02d}:00:00Z"
        args = build_args(
            output_dir=output_dir,
            date=date,
            seed=seed + index,
            platform=platform,
            community_name=community,
            channel_names=channels,
            observed_at=observed_at,
            runs_per_channel=runs_per_channel,
        )
        count += len(mockgen.run_generation(args))
    return count


def main() -> int:
    args = parse_args()
    platforms = args.platforms or ["discord", "reddit"]
    total = 0
    if "discord" in platforms:
        total += generate_platform(
            output_dir=args.output_dir,
            date=args.date,
            seed=args.seed,
            platform="discord",
            communities=DISCORD_SERVERS,
            channels=mockgen.TOPIC_CHANNELS["discord"],
            runs_per_channel=args.runs_per_channel,
        )
    if "reddit" in platforms:
        total += generate_platform(
            output_dir=args.output_dir,
            date=args.date,
            seed=args.seed + 1000,
            platform="reddit",
            communities=REDDIT_COMMUNITIES,
            channels=mockgen.TOPIC_CHANNELS["reddit"],
            runs_per_channel=args.runs_per_channel,
        )
    print(f"generated {total} files")
    print(Path(args.output_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
