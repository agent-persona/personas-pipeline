#!/bin/bash
# Setup script for automated Discord crawling on macOS.
# Creates a launchd plist that runs the crawl daily at 6am.
#
# Usage:
#   chmod +x feature_crawler/scripts/setup_automation.sh
#   ./feature_crawler/scripts/setup_automation.sh
#
# To uninstall:
#   launchctl unload ~/Library/LaunchAgents/com.agentpersonas.discord-crawl.plist
#   rm ~/Library/LaunchAgents/com.agentpersonas.discord-crawl.plist

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PLIST_NAME="com.agentpersonas.discord-crawl"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"
LOG_DIR="$PROJECT_ROOT/feature_crawler/logs"

# Check token
if [ -z "${DISCORD_USER_TOKEN:-}" ]; then
    if grep -q "DISCORD_USER_TOKEN" "$PROJECT_ROOT/.env" 2>/dev/null; then
        echo "✓ Token found in .env"
    else
        echo "✗ No DISCORD_USER_TOKEN found."
        echo "  Add it to $PROJECT_ROOT/.env:"
        echo "  echo 'DISCORD_USER_TOKEN=your_token_here' >> $PROJECT_ROOT/.env"
        exit 1
    fi
else
    echo "✓ Token found in environment"
fi

mkdir -p "$LOG_DIR"
mkdir -p "$HOME/Library/LaunchAgents"

# Write the plist
cat > "$PLIST_PATH" << PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>${SCRIPT_DIR}/discord_crawl_runner.py</string>
        <string>--guild</string>
        <string>662267976984297473</string>
        <string>--channel</string>
        <string>999550150705954856</string>
        <string>--channel</string>
        <string>952771221915840552</string>
        <string>--community-name</string>
        <string>target-public-server</string>
        <string>--message-limit</string>
        <string>200</string>
        <string>--min-delay</string>
        <string>2.5</string>
        <string>--max-delay</string>
        <string>5.0</string>
        <string>-v</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${PROJECT_ROOT}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
        <key>PYTHONPATH</key>
        <string>${PROJECT_ROOT}</string>
    </dict>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>6</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/crawl_stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/crawl_stderr.log</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
PLISTEOF

echo "✓ Plist written to $PLIST_PATH"

# Load it
launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"
echo "✓ Loaded into launchd — runs daily at 6:00 AM"

# Manual trigger instructions
echo ""
echo "To run immediately:"
echo "  launchctl start ${PLIST_NAME}"
echo ""
echo "To check logs:"
echo "  tail -f ${LOG_DIR}/crawl_stdout.log"
echo ""
echo "To stop/uninstall:"
echo "  launchctl unload ${PLIST_PATH}"
