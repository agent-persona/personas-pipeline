from __future__ import annotations

import argparse
import hashlib
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crawler.feature_crawler.core.models import (
    AccountRecord,
    CommunityRecord,
    EvidencePointer,
    InteractionRecord,
    MessageRecord,
    ProfileSnapshotRecord,
    Record,
    ThreadRecord,
)
from crawler.feature_crawler.core.sink import JsonlSink


TOPIC_CHANNELS = {
    "discord": [
        "art",
        "music-production",
        "cooking",
        "gardening",
        "board-games",
        "travel-hacks",
        "photography",
        "3d-printing",
        "indie-games",
        "sci-fi-books",
    ],
    "reddit": [
        "dancing",
        "running",
        "coffee",
        "pottery",
        "home-automation",
        "analog-photography",
        "screenwriting",
        "urban-gardening",
        "boardgames",
        "camping",
    ],
}

PEOPLE_CHANNELS = [
    "maya-product-designer",
    "leo-indie-founder",
    "nina-researcher",
    "omar-data-engineer",
    "sofia-community-manager",
    "ezra-game-designer",
    "iris-illustrator",
    "jonah-creator",
    "priya-architect",
    "lucas-operator",
]

PEOPLE_PLATFORMS = (
    "x",
    "linkedin",
    "slack",
    "telegram",
    "whatsapp",
    "twitch",
    "youtube",
    "hackernews",
)

USER_NAMES = (
    "Ari",
    "Bea",
    "Cole",
    "Dina",
    "Evan",
    "Faye",
    "Gray",
    "Hana",
    "Ivo",
    "Jules",
    "Kian",
    "Lena",
    "Milo",
    "Nora",
    "Otis",
    "Pia",
    "Quinn",
    "Rae",
    "Seth",
    "Tala",
)

TOPIC_HOOKS = (
    "warm-up habits",
    "small workflow fixes",
    "what changed this week",
    "mistakes worth avoiding",
    "favorite repeatable drills",
)

TOPIC_ANGLES = (
    "limit variables before chasing polish",
    "change timing before buying tools",
    "write down one rule and test it for a week",
    "start simple and add detail only if the core holds",
    "measure one thing instead of guessing from vibes",
)

PEOPLE_HOOKS = (
    "how this person makes tradeoffs",
    "what they optimize for under pressure",
    "which habits keep showing up in their work",
    "where their judgment differs from the room",
    "how they explain decisions to others",
)

PEOPLE_ANGLES = (
    "clarity before polish",
    "observable systems beat heroic effort",
    "narrow scope first, then add ambition",
    "trust grows from consistent small moves",
    "handoffs fail when assumptions stay implicit",
)


@dataclass(frozen=True, slots=True)
class MockPageSpec:
    platform: str
    channel_name: str
    page_number: int
    observed_at: str
    crawl_run_id: str
    server_name: str | None = None
    server_id: str | None = None
    channel_id: str | None = None


class MockJsonlSink(JsonlSink):
    def _output_parts(self, payloads: list[dict[str, object]]) -> list[str]:
        if not payloads:
            return ["mock"]
        first = payloads[0]
        platform = str(first.get("platform") or "unknown")
        date_fragment = str(first.get("observed_at") or "")[:10] or "unknown-date"
        if platform == "discord":
            community_records = [
                payload
                for payload in payloads
                if payload.get("record_type") == "community"
            ]
            server_record = next(
                (
                    payload
                    for payload in community_records
                    if payload.get("community_type") == "server"
                ),
                None,
            )
            channel_record = next(
                (
                    payload
                    for payload in community_records
                    if payload.get("community_type") in {"channel", "forum"}
                ),
                None,
            )
            parts = ["discord", "mock"]
            server_name = str((server_record or {}).get("community_name") or "")
            if server_name and server_name != "mock":
                parts.append(slugify(server_name))
            if channel_record:
                channel_name = str(channel_record.get("community_name") or "unknown")
                channel_id = str(channel_record.get("community_id") or "unknown")
                parts.append(f"{slugify(channel_name)}_{channel_id}")
            else:
                parts.append("unknown-channel")
            parts.append(flat_date_folder(date_fragment))
            return parts
        if platform == "reddit":
            thread_record = next(
                (
                    payload
                    for payload in payloads
                    if payload.get("record_type") == "thread"
                ),
                None,
            )
            topic_name = str(
                ((thread_record or {}).get("metadata") or {}).get("channel_name")
                or "general"
            )
            community_id = str(first.get("community_id") or "global")
            return [platform, "mock", community_id, slugify(topic_name), date_fragment]

        community_id = str(first.get("community_id") or "global")
        return [platform, "mock", community_id, date_fragment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mock conversation JSONL pages.")
    parser.add_argument("--output-dir", default="feature_crawler/data")
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--pages-per-channel", type=int, default=100)
    parser.add_argument("--channels-per-platform", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--channel-name", action="append", dest="channel_names")
    parser.add_argument("--crawl-run-id")
    parser.add_argument("--observed-at")
    parser.add_argument("--discord-server-name", default="mock")
    parser.add_argument("--discord-server-id", default="mock")
    parser.add_argument("--discord-channel-id")
    parser.add_argument(
        "--platform",
        action="append",
        dest="platforms",
        help="Repeat to limit generation to chosen platforms.",
    )
    return parser.parse_args()


def build_platform_channels(args: argparse.Namespace) -> dict[str, list[str]]:
    supported = dict(TOPIC_CHANNELS)
    for platform in PEOPLE_PLATFORMS:
        supported[platform] = list(PEOPLE_CHANNELS)
    if not args.platforms:
        return supported
    selected: dict[str, list[str]] = {}
    for platform in args.platforms:
        if platform not in supported:
            raise SystemExit(f"Unsupported platform: {platform}")
        selected[platform] = list(args.channel_names or supported[platform])
    return selected


def run_generation(args: argparse.Namespace) -> list[Path]:
    sink = MockJsonlSink(Path(args.output_dir))
    outputs: list[Path] = []
    start = datetime.fromisoformat(
        args.observed_at.replace("Z", "+00:00")
        if args.observed_at
        else f"{args.date}T09:00:00+00:00"
    )
    for platform, channels in build_platform_channels(args).items():
        for channel_name in channels[: args.channels_per_platform]:
            for page_number in range(1, args.pages_per_channel + 1):
                observed_at = (start + timedelta(minutes=page_number - 1)).isoformat()
                observed_at = observed_at.replace("+00:00", "Z")
                spec = MockPageSpec(
                    platform=platform,
                    channel_name=channel_name,
                    page_number=page_number,
                    observed_at=observed_at,
                    crawl_run_id=args.crawl_run_id
                    or f"run_mock_{platform}_{slugify(channel_name)}_page_{page_number:03d}",
                    server_name=(
                        args.discord_server_name
                        if platform in {"discord", "reddit"}
                        else None
                    ),
                    server_id=args.discord_server_id if platform == "discord" else None,
                    channel_id=args.discord_channel_id if platform == "discord" else None,
                )
                outputs.append(sink.write(build_page_records(spec, args.seed)))
    return outputs


def build_page_records(spec: MockPageSpec, seed: int) -> list[Record]:
    rng = random.Random(seed_for(spec, seed))
    thread_id = f"{spec.platform}-thread-{slugify(spec.channel_name)}-{spec.page_number:03d}"
    channel_id = channel_identifier(spec)
    source_url = source_url_for(spec)
    hook, angle = pick_prompt_bits(spec, rng)
    participants = build_participants(spec, rng)

    records: list[Record] = []
    records.extend(build_community_records(spec, channel_id, source_url))
    for participant in participants:
        records.extend(build_identity_records(spec, participant, source_url))

    records.append(
        ThreadRecord(
            record_type="thread",
            platform=spec.platform,
            thread_id=thread_id,
            community_id=channel_id,
            title=thread_title(spec, hook),
            author_platform_user_id=participants[0]["user_id"],
            created_at=spec.observed_at,
            observed_at=spec.observed_at,
            crawl_run_id=spec.crawl_run_id,
            metadata={
                "mock_data": True,
                "page_number": spec.page_number,
                "channel_name": spec.channel_name,
            },
            evidence_pointer=EvidencePointer(source_url=source_url, fetched_at=spec.observed_at),
        )
    )

    previous_message_id: str | None = None
    previous_user_id: str | None = None
    turns = rng.randint(46, 120)
    base_time = datetime.fromisoformat(spec.observed_at.replace("Z", "+00:00"))
    for turn_index in range(turns):
        participant = participants[turn_index % len(participants)]
        created_at = (base_time + timedelta(minutes=turn_index * 3)).isoformat()
        created_at = created_at.replace("+00:00", "Z")
        message_id = (
            f"{spec.platform}-message-{slugify(spec.channel_name)}-{spec.page_number:03d}-{turn_index + 1:02d}"
        )
        records.append(
            MessageRecord(
                record_type="message",
                platform=spec.platform,
                message_id=message_id,
                thread_id=thread_id,
                community_id=channel_id,
                author_platform_user_id=participant["user_id"],
                body=compose_turn(spec, participant, turn_index, hook, angle),
                created_at=created_at,
                observed_at=spec.observed_at,
                crawl_run_id=spec.crawl_run_id,
                reply_to_message_id=previous_message_id,
                reply_to_user_id=previous_user_id,
                metadata={"mock_data": True, "turn_index": turn_index + 1},
                evidence_pointer=EvidencePointer(
                    source_url=f"{source_url}#turn-{turn_index + 1}",
                    fetched_at=spec.observed_at,
                ),
            )
        )
        if previous_user_id and previous_user_id != participant["user_id"]:
            records.append(
                InteractionRecord(
                    record_type="interaction",
                    platform=spec.platform,
                    interaction_type="reply",
                    source_user_id=participant["user_id"],
                    target_user_id=previous_user_id,
                    message_id=message_id,
                    thread_id=thread_id,
                    community_id=channel_id,
                    created_at=created_at,
                    crawl_run_id=spec.crawl_run_id,
                    evidence_pointer=EvidencePointer(
                        source_url=f"{source_url}#turn-{turn_index + 1}",
                        fetched_at=spec.observed_at,
                        derived_from_message_id=message_id,
                    ),
                )
            )
        previous_message_id = message_id
        previous_user_id = participant["user_id"]
    return records


def build_community_records(spec: MockPageSpec, channel_id: str, source_url: str) -> list[Record]:
    pointer = EvidencePointer(source_url=source_url, fetched_at=spec.observed_at)
    if spec.platform == "discord":
        return [
            CommunityRecord(
                record_type="community",
                platform="discord",
                community_id=spec.server_id or "mock",
                community_name=spec.server_name or "mock",
                community_type="server",
                parent_community_id=None,
                description="Synthetic Discord mock corpus.",
                member_count=1200,
                rules_summary="Mock data only.",
                observed_at=spec.observed_at,
                crawl_run_id=spec.crawl_run_id,
                evidence_pointer=pointer,
            ),
            CommunityRecord(
                record_type="community",
                platform="discord",
                community_id=channel_id,
                community_name=spec.channel_name,
                community_type="channel",
                parent_community_id=spec.server_id or "mock",
                description=f"Topic channel for {spec.channel_name}.",
                member_count=180,
                rules_summary="Share process and examples.",
                observed_at=spec.observed_at,
                crawl_run_id=spec.crawl_run_id,
                evidence_pointer=pointer,
            ),
        ]
    return [
        CommunityRecord(
            record_type="community",
            platform=spec.platform,
            community_id=channel_id,
            community_name=community_name(spec),
            community_type="subreddit" if spec.platform == "reddit" else "people-channel",
            parent_community_id=None,
            description=f"Synthetic mock channel for {spec.channel_name}.",
            member_count=240,
            rules_summary="Mock conversations for offline testing.",
            observed_at=spec.observed_at,
            crawl_run_id=spec.crawl_run_id,
            evidence_pointer=pointer,
        )
    ]


def build_identity_records(spec: MockPageSpec, participant: dict[str, str], source_url: str) -> list[Record]:
    pointer = EvidencePointer(source_url=source_url, fetched_at=spec.observed_at)
    return [
        AccountRecord(
            record_type="account",
            platform=spec.platform,
            platform_user_id=participant["user_id"],
            username=participant["username"],
            account_created_at=None,
            first_observed_at=spec.observed_at,
            crawl_run_id=spec.crawl_run_id,
            evidence_pointer=pointer,
        ),
        ProfileSnapshotRecord(
            record_type="profile_snapshot",
            platform=spec.platform,
            platform_user_id=participant["user_id"],
            snapshot_at=spec.observed_at,
            crawl_run_id=spec.crawl_run_id,
            fields={"display_name": participant["username"], "role": participant["role"]},
            evidence_pointer=pointer,
        ),
    ]


def build_participants(spec: MockPageSpec, rng: random.Random) -> list[dict[str, str]]:
    roles = (
        ["host", "regular", "newcomer", "specialist"]
        if spec.platform in TOPIC_CHANNELS
        else ["host", "peer", "skeptic", "operator"]
    )
    picks = rng.sample(range(len(USER_NAMES)), 4)
    participants = []
    for slot, (pick, role) in enumerate(zip(picks, roles, strict=True), start=1):
        participants.append(
            {
                "user_id": f"{spec.platform}-user-{slugify(spec.channel_name)}-{slot}",
                "username": f"{USER_NAMES[pick]}{slot}",
                "role": role,
            }
        )
    return participants


def compose_turn(
    spec: MockPageSpec,
    participant: dict[str, str],
    turn_index: int,
    hook: str,
    angle: str,
) -> str:
    topic_label = spec.channel_name.replace("-", " ")
    prefix = {
        "host": "Kicking this off. ",
        "newcomer": "New here, but ",
        "specialist": "From the technical side, ",
        "skeptic": "Pushing back a little. ",
        "operator": "Operationally, ",
    }.get(participant["role"], "")
    lines = [
        f"I keep thinking about {topic_label} and {hook}. My default is {angle}.",
        f"Small win this week: {angle}. It made {topic_label} feel easier to reason about.",
        f"Question for the room: when {hook} starts going sideways, what do you change first?",
        f"My shortcut is boring but reliable: {angle}. Then I only tweak one variable.",
        f"I would document one principle here: {angle}. That is what scales across sessions.",
    ]
    return prefix + lines[turn_index % len(lines)]


def pick_prompt_bits(spec: MockPageSpec, rng: random.Random) -> tuple[str, str]:
    if spec.channel_name in PEOPLE_CHANNELS:
        return rng.choice(PEOPLE_HOOKS), rng.choice(PEOPLE_ANGLES)
    return rng.choice(TOPIC_HOOKS), rng.choice(TOPIC_ANGLES)


def thread_title(spec: MockPageSpec, hook: str) -> str:
    return f"{spec.channel_name} page {spec.page_number}: {hook}"


def community_name(spec: MockPageSpec) -> str:
    if spec.platform == "reddit":
        return f"r/{slugify(spec.server_name or spec.channel_name)}"
    return spec.channel_name


def channel_identifier(spec: MockPageSpec) -> str:
    if spec.platform == "discord":
        return spec.channel_id or f"mock-{slugify(spec.channel_name)}"
    if spec.platform == "reddit":
        return slugify(spec.server_name or spec.channel_name)
    return f"people/{slugify(spec.channel_name)}"


def source_url_for(spec: MockPageSpec) -> str:
    if spec.platform == "discord":
        return (
            f"mock://discord/{slugify(spec.server_name or 'mock')}/"
            f"{slugify(spec.channel_name)}/page-{spec.page_number:03d}"
        )
    if spec.platform == "reddit":
        return (
            f"mock://reddit/{slugify(spec.server_name or spec.channel_name)}/"
            f"{slugify(spec.channel_name)}/page-{spec.page_number:03d}"
        )
    return f"mock://{spec.platform}/{slugify(spec.channel_name)}/page-{spec.page_number:03d}"


def slugify(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in value]
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "unknown"


def flat_date_folder(date_fragment: str) -> str:
    parts = date_fragment.split("-")
    if len(parts) != 3:
        return date_fragment
    year, month, day = parts
    return f"{int(month)}-{int(day)}-{year}"


def seed_for(spec: MockPageSpec, seed: int) -> int:
    raw = f"{seed}:{spec.platform}:{spec.channel_name}:{spec.page_number}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def main() -> int:
    args = parse_args()
    outputs = run_generation(args)
    print(f"generated {len(outputs)} files")
    if outputs:
        print(outputs[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
