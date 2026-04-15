#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$SCRIPT_DIR/cron_runner.py"
LOG_DIR="$PROJECT_ROOT/feature_crawler/logs"
CRON_TAG_BEGIN="# feature_crawler:start"
CRON_TAG_END="# feature_crawler:end"
SCHEDULE="${1:-0 6 * * *}"
CRON_CMD="cd \"$PROJECT_ROOT\" && /usr/bin/python3 \"$RUNNER\" >> \"$LOG_DIR/cron_runner.log\" 2>&1"

mkdir -p "$LOG_DIR"

CURRENT_CRONTAB="$(mktemp)"
NEW_CRONTAB="$(mktemp)"
trap 'rm -f "$CURRENT_CRONTAB" "$NEW_CRONTAB"' EXIT

if crontab -l > "$CURRENT_CRONTAB" 2>/dev/null; then
  :
else
  : > "$CURRENT_CRONTAB"
fi

awk -v begin="$CRON_TAG_BEGIN" -v end="$CRON_TAG_END" '
  $0 == begin {skip=1; next}
  $0 == end {skip=0; next}
  skip != 1 {print}
' "$CURRENT_CRONTAB" > "$NEW_CRONTAB"

{
  cat "$NEW_CRONTAB"
  echo "$CRON_TAG_BEGIN"
  echo "$SCHEDULE $CRON_CMD"
  echo "$CRON_TAG_END"
} > "${NEW_CRONTAB}.tmp"
mv "${NEW_CRONTAB}.tmp" "$NEW_CRONTAB"

crontab "$NEW_CRONTAB"

echo "Installed cron entry:"
echo "$SCHEDULE $CRON_CMD"
echo
echo "Current feature_crawler crontab block:"
crontab -l | awk -v begin="$CRON_TAG_BEGIN" -v end="$CRON_TAG_END" '
  $0 == begin {show=1}
  show == 1 {print}
  $0 == end {show=0}
'
