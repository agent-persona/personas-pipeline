# Mock Data

## Purpose

Generate synthetic JSONL conversation pages that match the crawler bronze contract closely enough for local demos, persona experiments, and ingestion tests.

## Output shape

- topic platforms:
  - `discord` -> `feature_crawler/data/discord/mock/<server-optional>/<channel>_<channel-id>/<M-D-YYYY>/<run>.jsonl`
  - `reddit` -> `feature_crawler/data/reddit/mock/<channel>/<date>/<run>.jsonl`
- people platforms:
  - `x`, `linkedin`, `slack`, `telegram`, `whatsapp`, `twitch`, `youtube`, `hackernews`
  - path shape: `feature_crawler/data/<platform>/mock/people/<persona>/<date>/<run>.jsonl`

Each file is one page. Each line is one canonical record.

Records emitted:

- `community`
- `thread`
- `account`
- `profile_snapshot`
- `message`
- `interaction`

## Commands

Sample page only:

```bash
python3 feature_crawler/scripts/generate_mock_conversations.py \
  --platform discord \
  --pages-per-channel 1 \
  --channels-per-platform 1
```

Full default corpus:

```bash
python3 feature_crawler/scripts/generate_mock_conversations.py
```

Large network corpus:

```bash
python3 feature_crawler/scripts/generate_mock_network.py
```

Default behavior:

- 10 channels per platform
- 100 pages per channel
- deterministic content from `--seed`
- bulk network script:
  - Discord: 10 servers x 10 channels x 100 runs
  - Reddit: 10 subreddits x 10 topic threads x 100 runs
  - each file lands in the 100-500 line range
