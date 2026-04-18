"""
Discord Web Browser Connector

Automates the Discord web client to crawl public server channels by scrolling
through message history. Uses Playwright to control a headless or visible Chromium
instance.

WARNING: This connector interacts with Discord's web interface programmatically.
Ensure compliance with Discord's Terms of Service and acceptable use policies
before deploying to production. Excessive crawling may trigger rate limits or
account restrictions.

Browser automation implies certain risks:
- Discord may detect and block automated access
- Account suspension is possible if ToS violations are detected
- CAPTCHA challenges may interrupt crawling sessions
- Message extraction accuracy depends on DOM structure stability
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..base import CommunityConnector, CrawlContext
from ..models import (
    AccountRecord,
    CommunityRecord,
    CrawlTarget,
    EvidencePointer,
    InteractionRecord,
    MessageRecord,
    ProfileSnapshotRecord,
    Record,
    ThreadRecord,
)

logger = logging.getLogger("discord_browser")


class DiscordBrowserConnector(CommunityConnector):
    """
    Playwright-based Discord web client connector for crawling public channels.

    This connector navigates the Discord web interface directly, scrolling through
    message history to extract message data, author information, and thread context.
    """

    platform = "discord"

    def __init__(
        self,
        headless: bool = True,
        slow_mo: int = 500,
        scroll_pause: float = 2.0,
        max_scrolls: int = 100,
        storage_state_path: Optional[str] = None,
    ):
        """
        Initialize the Discord browser connector.

        Args:
            headless: If True, run browser headless. If False, show browser window.
            slow_mo: Milliseconds delay between Playwright actions for stability.
            scroll_pause: Seconds to wait after each scroll-up before re-checking messages.
            max_scrolls: Maximum number of scroll iterations per channel.
            storage_state_path: Path to saved browser state (cookies/localStorage).
                If provided and exists, loads saved login state automatically.
        """
        self.headless = headless
        self.slow_mo = slow_mo
        self.scroll_pause = scroll_pause
        self.max_scrolls = max_scrolls
        self.storage_state_path = storage_state_path

    def fetch(
        self,
        target: CrawlTarget,
        context: CrawlContext,
        since: Optional[datetime] = None,
    ) -> list[Record]:
        """
        Fetch records from a Discord channel by browser automation.

        Core flow:
        1. Launch Playwright chromium
        2. Load storage state if available (persistent login)
        3. Navigate to channel URL
        4. Wait for messages to load
        5. Scroll up repeatedly to load message history
        6. Extract all visible messages from DOM
        7. Convert to Record objects
        8. Close browser

        Args:
            target: CrawlTarget with platform_id in format "{guild_id}/{channel_id}"
            context: CrawlContext with parent community info
            since: Optional timestamp to filter messages (client-side filtering)

        Returns:
            List of Record objects including community, channel, messages, and authors.

        Raises:
            ValueError: If target.platform_id is malformed
            TimeoutError: If Discord fails to load or navigates to blocked page
            Exception: If browser automation fails
        """
        from playwright.sync_api import sync_playwright

        logger.info(f"Starting Discord fetch for target: {target.platform_id}")

        # Parse target
        parts = target.platform_id.split("/")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid Discord target format. Expected '{{guild_id}}/{{channel_id}}', "
                f"got '{target.platform_id}'"
            )

        guild_id, channel_id = parts
        channel_url = f"https://discord.com/channels/{guild_id}/{channel_id}"

        records: list[Record] = []
        browser = None
        page = None

        try:
            # Launch browser
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=self.headless,
                    slow_mo=self.slow_mo,
                )

                # Create context with storage state if available
                context_kwargs = {}
                if (
                    self.storage_state_path
                    and os.path.exists(self.storage_state_path)
                ):
                    logger.info(
                        f"Loading storage state from {self.storage_state_path}"
                    )
                    context_kwargs["storage_state"] = self.storage_state_path

                browser_context = browser.new_context(**context_kwargs)
                page = browser_context.new_page()

                # Navigate to channel
                logger.info(f"Navigating to {channel_url}")
                page.goto(channel_url, wait_until="networkidle")

                # Check for blocked states (CAPTCHA, verification, etc.)
                self._check_blocked_state(page)

                # Wait for messages to load
                try:
                    page.wait_for_selector(
                        "[class*='message_']",
                        timeout=5000,
                    )
                    logger.info("Messages loaded, beginning scroll")
                except Exception as e:
                    logger.warning(
                        f"Timeout waiting for messages: {e}. "
                        "Channel may be empty or Discord loaded differently."
                    )

                # Extract guild and channel info from page
                guild_info = self._extract_guild_info(page)
                logger.info(f"Guild info: {guild_info}")

                # Create community records
                server_community = CommunityRecord(
                    id=guild_info["guild_id"],
                    platform=self.platform,
                    name=guild_info["guild_name"],
                    url=f"https://discord.com/channels/{guild_id}",
                    description="",
                    member_count=None,
                )
                records.append(server_community)

                channel_community = CommunityRecord(
                    id=guild_info["channel_id"],
                    platform=self.platform,
                    name=guild_info["channel_name"],
                    url=channel_url,
                    description="",
                    parent_id=guild_info["guild_id"],
                )
                records.append(channel_community)

                # Create thread record for the channel history window
                thread = ThreadRecord(
                    id=guild_info["channel_id"],
                    platform=self.platform,
                    name=guild_info["channel_name"],
                    community_id=guild_info["channel_id"],
                    parent_id=guild_info["guild_id"],
                    url=channel_url,
                    created_at=None,
                    updated_at=datetime.now(),
                )
                records.append(thread)

                # Scroll to load history
                messages_loaded = self._scroll_to_load_history(
                    page,
                    self.max_scrolls,
                    self.scroll_pause,
                )
                logger.info(f"Scroll complete: {messages_loaded} messages in DOM")

                # Extract all visible messages
                message_dicts = self._extract_messages_from_page(page)
                logger.info(f"Extracted {len(message_dicts)} messages from page")

                # Track seen authors to avoid duplicates
                seen_authors: dict[str, AccountRecord] = {}

                # Convert message dicts to records
                for msg_data in message_dicts:
                    # Filter by since if provided
                    if since and msg_data.get("timestamp"):
                        try:
                            msg_time = datetime.fromisoformat(
                                msg_data["timestamp"].replace("Z", "+00:00")
                            )
                            if msg_time < since:
                                continue
                        except (ValueError, TypeError):
                            pass  # Skip timestamp parsing errors

                    # Create or reuse author records
                    author_id = msg_data.get("author_id") or msg_data[
                        "author_name"
                    ]
                    if author_id not in seen_authors:
                        account = AccountRecord(
                            id=author_id,
                            platform=self.platform,
                            username=msg_data["author_name"],
                            email=None,
                        )
                        records.append(account)
                        seen_authors[author_id] = account

                        # Add profile snapshot
                        profile = ProfileSnapshotRecord(
                            account_id=author_id,
                            platform=self.platform,
                            display_name=msg_data.get("author_display_name")
                            or msg_data["author_name"],
                            avatar_url=msg_data.get("author_avatar_url"),
                            bio=None,
                            url=None,
                            captured_at=datetime.now(),
                        )
                        records.append(profile)

                    # Create message record
                    message = MessageRecord(
                        id=msg_data["id"],
                        platform=self.platform,
                        account_id=author_id,
                        community_id=guild_info["channel_id"],
                        thread_id=guild_info["channel_id"],
                        body=msg_data["body"],
                        body_type="text/plain",
                        url=f"{channel_url}/{msg_data['id']}",
                        created_at=msg_data.get("timestamp"),
                        updated_at=None,
                    )
                    records.append(message)

                    # Create interaction record for replies
                    if msg_data.get("reply_to_id"):
                        interaction = InteractionRecord(
                            id=f"{msg_data['id']}_reply",
                            platform=self.platform,
                            account_id=author_id,
                            target_type="message",
                            target_id=msg_data["reply_to_id"],
                            interaction_type="reply",
                            timestamp=msg_data.get("timestamp"),
                        )
                        records.append(interaction)

                logger.info(
                    f"Fetch complete: {len(records)} total records "
                    f"({len(seen_authors)} authors, "
                    f"{len(message_dicts)} messages)"
                )

                browser_context.close()

        except Exception as e:
            logger.error(f"Error during fetch: {e}", exc_info=True)
            raise

        return records

    def _check_blocked_state(self, page: Any) -> None:
        """
        Check if Discord is displaying a blocked state (CAPTCHA, verification, etc.).

        Raises:
            TimeoutError: If a blocking page is detected.
        """
        blocked_indicators = [
            "text=Checking if the site connection is secure",
            "text=Please verify",
            "h1:has-text('Human verification')",
        ]

        for selector in blocked_indicators:
            try:
                if page.query_selector(selector):
                    raise TimeoutError(
                        f"Discord is showing a blocking page: {selector}"
                    )
            except Exception:
                pass  # Selector not found is expected

    def _extract_guild_info(self, page: Any) -> dict[str, Any]:
        """
        Extract server and channel information from the page header.

        Attempts to extract guild name, channel name, and IDs from DOM.
        Falls back to parsing URL if extraction fails.

        Returns:
            Dict with keys: guild_id, guild_name, channel_id, channel_name
        """
        try:
            # Use JavaScript to extract header info
            info = page.evaluate(
                """() => {
                const guildEl = document.querySelector('[class*="headerText_"]');
                const channelEl = document.querySelector('h1');
                return {
                    guildName: guildEl ? guildEl.textContent : 'Unknown Guild',
                    channelName: channelEl ? channelEl.textContent : 'Unknown Channel'
                };
            }"""
            )

            # Parse guild_id and channel_id from URL
            url = page.url
            parts = url.split("/channels/")
            if len(parts) == 2:
                ids = parts[1].split("/")
                guild_id = ids[0] if len(ids) > 0 else "unknown"
                channel_id = ids[1] if len(ids) > 1 else "unknown"
            else:
                guild_id = "unknown"
                channel_id = "unknown"

            return {
                "guild_id": guild_id,
                "guild_name": info.get("guildName", "Unknown Guild"),
                "channel_id": channel_id,
                "channel_name": info.get("channelName", "Unknown Channel"),
            }
        except Exception as e:
            logger.warning(f"Error extracting guild info: {e}")
            return {
                "guild_id": "unknown",
                "guild_name": "Unknown Guild",
                "channel_id": "unknown",
                "channel_name": "Unknown Channel",
            }

    def _scroll_to_load_history(
        self,
        page: Any,
        max_scrolls: int,
        scroll_pause: float,
    ) -> int:
        """
        Scroll to the top of the message history to load older messages.

        Discord loads messages dynamically as you scroll. This method repeatedly
        scrolls to the top of the message container, waits for new messages to load,
        and repeats until no new messages appear or max_scrolls is reached.

        Args:
            page: Playwright page object
            max_scrolls: Maximum number of scroll iterations
            scroll_pause: Seconds to wait after each scroll

        Returns:
            Total number of message elements found in DOM
        """
        import time

        scroll_count = 0
        prev_count = 0

        for i in range(max_scrolls):
            try:
                # Get current message count
                current_count = page.evaluate(
                    "document.querySelectorAll('[class*=\"message_\"]').length"
                )

                logger.debug(
                    f"Scroll {i + 1}/{max_scrolls}: "
                    f"{current_count} messages (prev: {prev_count})"
                )

                # If no new messages loaded, stop scrolling
                if current_count == prev_count and i > 0:
                    logger.debug("No new messages loaded, stopping scroll")
                    break

                prev_count = current_count

                # Scroll to top of message container
                page.evaluate(
                    """() => {
                    const scroller = document.querySelector('[class*="scroller_"]');
                    if (scroller) {
                        scroller.scrollTop = 0;
                    }
                }"""
                )

                scroll_count += 1
                time.sleep(scroll_pause)

            except Exception as e:
                logger.warning(f"Error during scroll iteration {i + 1}: {e}")
                break

        final_count = page.evaluate(
            "document.querySelectorAll('[class*=\"message_\"]').length"
        )
        logger.info(
            f"Scrolled {scroll_count} times, {final_count} messages loaded"
        )

        return final_count

    def _extract_messages_from_page(self, page: Any) -> list[dict[str, Any]]:
        """
        Extract all visible messages from the Discord page DOM.

        Uses JavaScript evaluation to traverse the DOM and extract message data
        from Discord's obfuscated class names. Handles reply context and author info.

        Returns:
            List of message dicts with keys:
            - id: message ID
            - author_name: display name or username
            - author_id: numeric ID if available
            - author_display_name: full display name if different from username
            - author_avatar_url: profile picture URL if available
            - body: message content text
            - timestamp: ISO 8601 timestamp
            - reply_to_id: ID of replied-to message if present
            - reply_to_author: username of replied-to author if present
        """
        try:
            messages = page.evaluate(
                """() => {
                const msgs = [];
                const messageEls = document.querySelectorAll('[class*="message_"]');

                for (const msgEl of messageEls) {
                    try {
                        // Extract message ID from aria attributes or data attributes
                        let msgId = msgEl.getAttribute('data-message-id');
                        if (!msgId) {
                            const idAttr = Object.keys(msgEl).find(k =>
                                k.startsWith('__reactProps')
                            );
                            if (idAttr) {
                                msgId = msgEl[idAttr]?.memoizedProps?.id;
                            }
                        }
                        if (!msgId) {
                            msgId = msgEl.id || `msg_${Math.random()}`;
                        }

                        // Extract author info
                        const authorEl = msgEl.querySelector(
                            '[class*="username_"]'
                        ) || msgEl.querySelector('h3 span');
                        const authorName = authorEl
                            ? authorEl.textContent.trim()
                            : 'Unknown';

                        // Extract message content
                        const contentEl = msgEl.querySelector(
                            '[id^="message-content-"]'
                        ) || msgEl.querySelector('[class*="message_"]');
                        const content = contentEl
                            ? contentEl.textContent.trim()
                            : '';

                        // Extract timestamp
                        const timeEl = msgEl.querySelector('time');
                        const timestamp = timeEl
                            ? timeEl.getAttribute('datetime')
                            : new Date().toISOString();

                        // Extract reply context if present
                        let replyToId = null;
                        let replyToAuthor = null;
                        const repliedEl = msgEl.querySelector(
                            '[class*="repliedMessage"]'
                        );
                        if (repliedEl) {
                            const replyAuthorEl = repliedEl.querySelector(
                                '[class*="username_"]'
                            );
                            replyToAuthor = replyAuthorEl
                                ? replyAuthorEl.textContent.trim()
                                : null;
                            // Try to extract replied message ID from link
                            const replyLink = repliedEl.querySelector('a');
                            if (replyLink && replyLink.href) {
                                const parts = replyLink.href.split('/');
                                replyToId = parts[parts.length - 1];
                            }
                        }

                        // Extract avatar URL
                        const avatarImg = msgEl.querySelector('img[alt*="' +
                            authorName + '"]') ||
                            msgEl.querySelector('[role="img"] img');
                        const avatarUrl = avatarImg
                            ? avatarImg.src
                            : null;

                        msgs.push({
                            id: msgId,
                            author_name: authorName,
                            author_id: null,
                            author_display_name: authorName,
                            author_avatar_url: avatarUrl,
                            body: content,
                            timestamp: timestamp,
                            reply_to_id: replyToId,
                            reply_to_author: replyToAuthor
                        });
                    } catch (err) {
                        console.error('Error extracting message:', err);
                    }
                }

                return msgs;
            }"""
            )

            logger.info(f"Extracted {len(messages)} messages via JavaScript")
            return messages

        except Exception as e:
            logger.error(f"Error extracting messages from page: {e}")
            return []

    @classmethod
    def save_login_state(cls, storage_state_path: str) -> None:
        """
        Save browser login state for future authenticated access.

        Launches a visible browser window, asks user to manually log into Discord,
        then saves the session state (cookies, localStorage) to a file.

        Args:
            storage_state_path: Path where to save the session state JSON file.

        Example:
            >>> DiscordBrowserConnector.save_login_state("./discord_state.json")
            # Browser opens, user logs in, press Enter in terminal
            # Session saved to ./discord_state.json
        """
        from playwright.sync_api import sync_playwright

        logger.info(f"Starting login flow, will save to {storage_state_path}")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False, slow_mo=500)
            context = browser.new_context()
            page = context.new_page()

            # Navigate to login
            page.goto("https://discord.com/login")

            # Wait for user to complete login
            print("\n" + "=" * 60)
            print("Discord browser window opened.")
            print("Please log in to Discord manually in the browser window.")
            print("Once logged in, press Enter here to save the session...")
            print("=" * 60 + "\n")

            input()

            # Save storage state
            state = context.storage_state()
            Path(storage_state_path).write_text(
                __import__("json").dumps(state, indent=2)
            )
            logger.info(f"Session state saved to {storage_state_path}")

            context.close()
            browser.close()

            print(f"\nSession saved to {storage_state_path}")
            print(
                "You can now use this file with "
                "DiscordBrowserConnector(storage_state_path='...')"
            )
