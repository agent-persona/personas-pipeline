from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256
from html.parser import HTMLParser
from typing import Any, Iterable
from urllib.parse import urlparse


def stable_id(*parts: str) -> str:
    joined = "::".join(parts)
    return sha256(joined.encode("utf-8")).hexdigest()[:16]


def strip_html(value: str) -> str:
    parser = _HtmlTextExtractor()
    parser.feed(value)
    return parser.text


@dataclass(slots=True)
class ParsedMessage:
    message_id: str
    author_name: str | None
    body: str
    created_at: str | None
    reply_to_message_id: str | None = None
    reply_to_author_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedThread:
    title: str
    author_name: str | None
    description: str | None
    created_at: str | None
    messages: list[ParsedMessage]
    metadata: dict[str, Any] = field(default_factory=dict)
    parser_name: str = "html-paragraphs"


class _HtmlTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    @property
    def text(self) -> str:
        return " ".join(piece.strip() for piece in self._parts if piece.strip())

    def handle_data(self, data: str) -> None:
        clean = " ".join(data.split())
        if clean:
            self._parts.append(clean)


class _DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.author: str | None = None
        self.description: str | None = None
        self.paragraphs: list[str] = []
        self._capture_title = False
        self._capture_text = False
        self._capture_script = False
        self._capture_depth = 0
        self._ignored_depth = 0
        self._current_script_type: str | None = None
        self._buffer: list[str] = []
        self.json_ld_blocks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key.lower(): value for key, value in attrs}
        if tag in {"script", "style"} and tag != "script":
            self._ignored_depth += 1
            return
        if tag == "script":
            self._capture_script = True
            self._current_script_type = (attr_map.get("type") or "").lower()
            if self._current_script_type == "application/ld+json":
                self._buffer.clear()
            return
        if tag == "title":
            self._capture_title = True
        if tag in {"p", "li"}:
            self._capture_text = True
            self._capture_depth += 1
        if tag == "meta":
            name = (attr_map.get("name") or attr_map.get("property") or "").lower()
            content = (attr_map.get("content") or "").strip()
            if not content:
                return
            if name in {"author", "article:author"} and not self.author:
                self.author = content
            elif name in {"description", "og:description"} and not self.description:
                self.description = content

    def handle_endtag(self, tag: str) -> None:
        if tag in {"style"} and self._ignored_depth > 0:
            self._ignored_depth -= 1
            return
        if tag == "script":
            if self._current_script_type == "application/ld+json":
                block = "".join(self._buffer).strip()
                if block:
                    self.json_ld_blocks.append(block)
            self._capture_script = False
            self._current_script_type = None
            self._buffer.clear()
            return
        if tag == "title":
            self._capture_title = False
        if tag in {"p", "li"} and self._capture_depth > 0:
            self._capture_depth -= 1
            if self._capture_depth == 0:
                self._capture_text = False
                text = " ".join(piece.strip() for piece in self._buffer if piece.strip())
                if len(text) >= 40:
                    self.paragraphs.append(text)
                self._buffer.clear()

    def handle_data(self, data: str) -> None:
        if self._ignored_depth > 0:
            return
        if self._capture_script and self._current_script_type == "application/ld+json":
            self._buffer.append(data)
            return
        clean = " ".join(data.split())
        if not clean:
            return
        if self._capture_title and not self.title:
            self.title = clean
        if self._capture_text:
            self._buffer.append(clean)


def parse_payload(raw: str, content_type: str | None, source_url: str) -> ParsedThread:
    content_type = (content_type or "").lower()
    if "json" in content_type or raw.lstrip().startswith("{"):
        parsed_json = _parse_json(raw, source_url)
        if parsed_json is not None:
            return parsed_json
    return _parse_html(raw, source_url)


def parse_html(raw_html: str, source_url: str) -> ParsedThread:
    return _parse_html(raw_html, source_url)


def _parse_json(raw: str, source_url: str) -> ParsedThread | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("post_stream"), dict):
        return _parse_discourse_topic(payload, source_url)
    return None


def _parse_discourse_topic(payload: dict[str, Any], source_url: str) -> ParsedThread:
    posts = payload.get("post_stream", {}).get("posts", [])
    if not isinstance(posts, list):
        posts = []

    post_number_to_message_id: dict[int, str] = {}
    messages: list[ParsedMessage] = []
    topic_id = str(payload.get("id") or stable_id(source_url))

    for post in posts:
        if not isinstance(post, dict):
            continue
        post_number = int(post.get("post_number") or 0)
        message_id = f"discourse-post-{post.get('id') or stable_id(topic_id, str(post_number))}"
        post_number_to_message_id[post_number] = message_id

    for post in posts:
        if not isinstance(post, dict):
            continue
        post_number = int(post.get("post_number") or 0)
        cooked = post.get("cooked") or ""
        reply_to_post_number = post.get("reply_to_post_number")
        reply_to_user = post.get("reply_to_user")
        if isinstance(reply_to_user, dict):
            reply_to_author_name = reply_to_user.get("username") or reply_to_user.get("name")
        else:
            reply_to_author_name = reply_to_user if isinstance(reply_to_user, str) else None

        messages.append(
            ParsedMessage(
                message_id=post_number_to_message_id.get(post_number, f"discourse-post-{post_number}"),
                author_name=post.get("username") or post.get("name"),
                body=strip_html(cooked),
                created_at=post.get("created_at"),
                reply_to_message_id=post_number_to_message_id.get(int(reply_to_post_number or 0)),
                reply_to_author_name=reply_to_author_name,
                metadata={
                    "parser": "discourse-json",
                    "post_number": post_number,
                    "reply_count": post.get("reply_count"),
                    "reads": post.get("reads"),
                    "score": post.get("score"),
                    "topic_id": topic_id,
                },
            )
        )

    first_author = messages[0].author_name if messages else None
    first_created_at = messages[0].created_at if messages else None
    return ParsedThread(
        title=payload.get("title") or source_url,
        author_name=first_author,
        description=payload.get("excerpt"),
        created_at=first_created_at,
        messages=[message for message in messages if message.body],
        metadata={
            "parser": "discourse-json",
            "topic_id": topic_id,
            "slug": payload.get("slug"),
            "posts_count": payload.get("posts_count"),
        },
        parser_name="discourse-json",
    )


def _parse_html(raw_html: str, source_url: str) -> ParsedThread:
    parser = _DocumentParser()
    parser.feed(raw_html)

    json_ld_thread = _parse_json_ld_thread(
        blocks=parser.json_ld_blocks,
        fallback_title=parser.title or source_url,
        source_url=source_url,
    )
    if json_ld_thread is not None and json_ld_thread.messages:
        if not json_ld_thread.description:
            json_ld_thread.description = parser.description
        if not json_ld_thread.author_name:
            json_ld_thread.author_name = parser.author
        return json_ld_thread

    messages = [
        ParsedMessage(
            message_id=f"web-message-{stable_id(source_url, str(index), paragraph)}",
            author_name=parser.author,
            body=paragraph,
            created_at=None,
            metadata={"parser": "html-paragraphs", "ordinal": index, "source_kind": "paragraph"},
        )
        for index, paragraph in enumerate(parser.paragraphs)
    ]
    return ParsedThread(
        title=parser.title or source_url,
        author_name=parser.author,
        description=parser.description,
        created_at=None,
        messages=messages,
        metadata={"parser": "html-paragraphs"},
        parser_name="html-paragraphs",
    )


def _parse_json_ld_thread(
    *,
    blocks: list[str],
    fallback_title: str,
    source_url: str,
) -> ParsedThread | None:
    root_title = fallback_title
    root_author: str | None = None
    root_description: str | None = None
    root_created_at: str | None = None
    messages: list[ParsedMessage] = []

    for block in blocks:
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        for node in _iter_json_nodes(payload):
            node_type = _node_types(node)
            if node_type & {"discussionforumposting", "socialmediaposting", "article", "blogposting", "question"}:
                root_title = (
                    _string_value(node, "headline")
                    or _string_value(node, "name")
                    or root_title
                )
                root_author = root_author or _person_name(node.get("author"))
                root_description = root_description or _string_value(node, "description")
                root_created_at = root_created_at or _string_value(node, "dateCreated")
                comments = node.get("comment")
                if comments:
                    messages.extend(
                        _parse_comment_nodes(
                            comments=comments,
                            source_url=source_url,
                            parent_message_id=None,
                            parent_author_name=None,
                        )
                    )

    if not messages:
        return None

    return ParsedThread(
        title=root_title,
        author_name=root_author,
        description=root_description,
        created_at=root_created_at,
        messages=messages,
        metadata={"parser": "json-ld-discussion"},
        parser_name="json-ld-discussion",
    )


def _parse_comment_nodes(
    *,
    comments: Any,
    source_url: str,
    parent_message_id: str | None,
    parent_author_name: str | None,
) -> list[ParsedMessage]:
    result: list[ParsedMessage] = []
    comment_items = comments if isinstance(comments, list) else [comments]

    for index, comment in enumerate(comment_items):
        if not isinstance(comment, dict):
            continue
        message_id = (
            _string_value(comment, "@id")
            or _string_value(comment, "url")
            or f"web-message-{stable_id(source_url, str(index), _string_value(comment, 'text') or _string_value(comment, 'commentText') or '')}"
        )
        author_name = _person_name(comment.get("author"))
        body = (
            _string_value(comment, "text")
            or _string_value(comment, "articleBody")
            or _string_value(comment, "commentText")
        )
        if not body:
            continue
        parsed = ParsedMessage(
            message_id=message_id,
            author_name=author_name,
            body=body,
            created_at=_string_value(comment, "dateCreated"),
            reply_to_message_id=parent_message_id,
            reply_to_author_name=parent_author_name,
            metadata={"parser": "json-ld-discussion", "source_kind": "comment"},
        )
        result.append(parsed)
        child_comments = comment.get("comment")
        if child_comments:
            result.extend(
                _parse_comment_nodes(
                    comments=child_comments,
                    source_url=source_url,
                    parent_message_id=message_id,
                    parent_author_name=author_name,
                )
            )
    return result


def _iter_json_nodes(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict):
        yield payload
        for value in payload.values():
            yield from _iter_json_nodes(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_json_nodes(item)


def _node_types(node: dict[str, Any]) -> set[str]:
    raw_type = node.get("@type")
    if isinstance(raw_type, list):
        return {str(item).lower() for item in raw_type}
    if isinstance(raw_type, str):
        return {raw_type.lower()}
    return set()


def _string_value(node: dict[str, Any], key: str) -> str | None:
    value = node.get(key)
    if isinstance(value, str):
        return " ".join(value.split()) or None
    return None


def _person_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _string_value(value, "name") or _string_value(value, "alternateName")
    return None


def looks_like_js_shell(raw_html: str, parsed: ParsedThread) -> bool:
    if parsed.parser_name == "discourse-json":
        return False
    lowered = raw_html.lower()
    script_count = lowered.count("<script")
    if any(marker in lowered for marker in ("id=\"__next\"", "data-reactroot", "window.__nuxt__", "__next_data__")):
        return True
    visible_text = re.sub(r"<[^>]+>", " ", raw_html)
    visible_text = " ".join(visible_text.split())
    return script_count >= 5 and len(parsed.messages) < 2 and len(visible_text) < 1200


def derive_site_name(url: str, fallback: str) -> str:
    domain = urlparse(url).netloc
    return domain or fallback
