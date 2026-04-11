"""Memory backends for twin session continuity (experiment 4.06).

Four architectures with increasing sophistication:
  - NoMemory:        baseline, no cross-session state
  - ScratchpadMemory: rolling plain-text summary of prior sessions
  - EpisodicMemory:  structured JSON episodes per session
  - VectorMemory:    TF-IDF similarity retrieval over prior turns
"""

from __future__ import annotations

import json
import math
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Turn:
    """A single conversational exchange."""
    role: str           # "user" or "assistant"
    content: str
    session_id: str
    timestamp: float = 0.0


@dataclass
class Episode:
    """A structured summary of one session."""
    session_id: str
    started_at: float
    ended_at: float
    turn_count: int
    topics: list[str]
    user_mood: str
    open_threads: list[str]
    summary: str


class MemoryBackend(ABC):
    """Protocol for twin memory backends."""

    @abstractmethod
    def get_context(self, session_id: str) -> str:
        """Return context string to prepend to the system prompt."""
        ...

    @abstractmethod
    def store_turn(self, turn: Turn) -> None:
        """Store a conversational turn."""
        ...

    @abstractmethod
    def end_session(self, session_id: str) -> None:
        """Signal that a session has ended (for backends that summarize)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ============================================================================
# 1. NoMemory — baseline
# ============================================================================

class NoMemory(MemoryBackend):
    """No cross-session memory. Every session starts fresh."""

    name = "none"

    def get_context(self, session_id: str) -> str:
        return ""

    def store_turn(self, turn: Turn) -> None:
        pass

    def end_session(self, session_id: str) -> None:
        pass


# ============================================================================
# 2. ScratchpadMemory — rolling plain-text summary
# ============================================================================

class ScratchpadMemory(MemoryBackend):
    """Maintains a rolling scratchpad of key points from prior sessions.

    After each session ends, the last N exchanges are condensed into bullet
    points and appended to the scratchpad. The scratchpad is injected into
    the system prompt for subsequent sessions.
    """

    name = "scratchpad"

    def __init__(self, max_lines: int = 20) -> None:
        self.max_lines = max_lines
        self.scratchpad_lines: list[str] = []
        self._session_turns: dict[str, list[Turn]] = {}

    def get_context(self, session_id: str) -> str:
        if not self.scratchpad_lines:
            return ""
        lines = "\n".join(f"- {line}" for line in self.scratchpad_lines[-self.max_lines:])
        return (
            "\n\n## Memory (from previous sessions)\n"
            "You remember these key points from past conversations:\n"
            + lines
        )

    def store_turn(self, turn: Turn) -> None:
        self._session_turns.setdefault(turn.session_id, []).append(turn)

    def end_session(self, session_id: str) -> None:
        turns = self._session_turns.pop(session_id, [])
        if not turns:
            return
        # Extract key points from the session
        user_msgs = [t.content for t in turns if t.role == "user"]
        asst_msgs = [t.content for t in turns if t.role == "assistant"]
        # Simple extraction: first user message as topic, last exchange as conclusion
        if user_msgs:
            self.scratchpad_lines.append(
                f"Session {session_id}: User asked about: {user_msgs[0][:100]}"
            )
        if len(user_msgs) > 1:
            for msg in user_msgs[1:]:
                self.scratchpad_lines.append(
                    f"Session {session_id}: User also mentioned: {msg[:80]}"
                )
        if asst_msgs:
            self.scratchpad_lines.append(
                f"Session {session_id}: You responded about: {asst_msgs[-1][:100]}"
            )
        # Trim to max
        if len(self.scratchpad_lines) > self.max_lines:
            self.scratchpad_lines = self.scratchpad_lines[-self.max_lines:]


# ============================================================================
# 3. EpisodicMemory — structured JSON episodes
# ============================================================================

class EpisodicMemory(MemoryBackend):
    """Stores structured episode summaries per session.

    Each session is condensed into an Episode with topics, mood, open
    threads, and a narrative summary. All episodes are injected into the
    system prompt for subsequent sessions.
    """

    name = "episodic"

    def __init__(self, max_episodes: int = 10) -> None:
        self.max_episodes = max_episodes
        self.episodes: list[Episode] = []
        self._session_turns: dict[str, list[Turn]] = {}

    def get_context(self, session_id: str) -> str:
        if not self.episodes:
            return ""
        sections = ["\n\n## Episodic Memory (structured recall from prior sessions)"]
        sections.append(
            "You have detailed memories of past conversations. Use them to "
            "maintain continuity — reference prior topics naturally, remember "
            "the user's preferences, and pick up open threads."
        )
        for ep in self.episodes[-self.max_episodes:]:
            sections.append(f"\n### Session: {ep.session_id}")
            sections.append(f"- Topics discussed: {', '.join(ep.topics)}")
            sections.append(f"- User mood: {ep.user_mood}")
            if ep.open_threads:
                sections.append(f"- Open threads: {', '.join(ep.open_threads)}")
            sections.append(f"- Summary: {ep.summary}")
        return "\n".join(sections)

    def store_turn(self, turn: Turn) -> None:
        self._session_turns.setdefault(turn.session_id, []).append(turn)

    def end_session(self, session_id: str) -> None:
        turns = self._session_turns.pop(session_id, [])
        if not turns:
            return

        user_msgs = [t.content for t in turns if t.role == "user"]
        asst_msgs = [t.content for t in turns if t.role == "assistant"]

        # Extract topics from user messages (simple keyword extraction)
        topics = _extract_topics(user_msgs)

        # Simple mood detection
        mood = _detect_mood(user_msgs)

        # Open threads: questions in last user message
        open_threads = []
        if user_msgs:
            last = user_msgs[-1]
            if "?" in last:
                open_threads.append(last[:80])

        # Build narrative summary
        summary_parts = []
        if user_msgs:
            summary_parts.append(f"User discussed: {user_msgs[0][:80]}")
        if asst_msgs:
            summary_parts.append(f"You shared: {asst_msgs[0][:80]}")
        if len(user_msgs) > 1:
            summary_parts.append(f"Conversation covered {len(user_msgs)} topics")

        timestamps = [t.timestamp for t in turns if t.timestamp > 0]
        started = min(timestamps) if timestamps else 0.0
        ended = max(timestamps) if timestamps else 0.0

        episode = Episode(
            session_id=session_id,
            started_at=started,
            ended_at=ended,
            turn_count=len(turns),
            topics=topics,
            user_mood=mood,
            open_threads=open_threads,
            summary=". ".join(summary_parts),
        )
        self.episodes.append(episode)
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]


# ============================================================================
# 4. VectorMemory — TF-IDF similarity retrieval
# ============================================================================

class VectorMemory(MemoryBackend):
    """Retrieves the most relevant prior turns via TF-IDF cosine similarity.

    Uses a simple in-process TF-IDF implementation (no external deps) to
    find the top-K most relevant prior exchanges when building context.
    """

    name = "vector"

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self._turns: list[Turn] = []
        self._session_turns: dict[str, list[Turn]] = {}

    def get_context(self, session_id: str) -> str:
        # Only retrieve from OTHER sessions (not the current one)
        prior = [t for t in self._turns if t.session_id != session_id]
        if not prior:
            return ""

        # Get the current session's turns as the query
        current = self._session_turns.get(session_id, [])
        if current:
            query = " ".join(t.content for t in current[-3:])  # last 3 turns
        else:
            query = ""

        if not query:
            # No current context yet — return most recent turns
            recent = prior[-self.top_k:]
            return self._format_turns(recent)

        # TF-IDF retrieval
        docs = [t.content for t in prior]
        scores = _tfidf_similarity(query, docs)
        ranked = sorted(zip(prior, scores), key=lambda x: x[1], reverse=True)
        top = [t for t, s in ranked[:self.top_k]]

        return self._format_turns(top)

    def _format_turns(self, turns: list[Turn]) -> str:
        if not turns:
            return ""
        sections = ["\n\n## Retrieved Memories (most relevant prior exchanges)"]
        sections.append(
            "These are the most relevant things from your past conversations. "
            "Reference them naturally if the user brings up related topics."
        )
        for t in turns:
            role_label = "User said" if t.role == "user" else "You said"
            sections.append(f"- [{t.session_id}] {role_label}: {t.content[:150]}")
        return "\n".join(sections)

    def store_turn(self, turn: Turn) -> None:
        self._turns.append(turn)
        self._session_turns.setdefault(turn.session_id, []).append(turn)

    def end_session(self, session_id: str) -> None:
        # No special action needed — turns are already stored
        self._session_turns.pop(session_id, None)


# ============================================================================
# Utility functions
# ============================================================================

def _extract_topics(messages: list[str], max_topics: int = 4) -> list[str]:
    """Extract key topic words from messages (simple frequency-based)."""
    stop_words = {
        "i", "me", "my", "you", "your", "we", "our", "the", "a", "an",
        "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should",
        "can", "may", "might", "shall", "to", "of", "in", "for", "on",
        "with", "at", "by", "from", "as", "into", "about", "that", "this",
        "it", "not", "but", "and", "or", "if", "so", "what", "how", "when",
        "where", "which", "who", "than", "then", "there", "here", "just",
        "also", "very", "much", "more", "most", "some", "any", "all",
        "no", "up", "out", "like", "know", "think", "want", "need",
        "really", "don't", "i'm", "it's", "that's", "they", "them",
    }
    words: list[str] = []
    for msg in messages:
        for word in msg.lower().split():
            cleaned = word.strip(".,!?;:'\"()-")
            if cleaned and len(cleaned) > 2 and cleaned not in stop_words:
                words.append(cleaned)
    counter = Counter(words)
    return [word for word, _ in counter.most_common(max_topics)]


def _detect_mood(messages: list[str]) -> str:
    """Simple keyword-based mood detection."""
    text = " ".join(messages).lower()
    if any(w in text for w in ["frustrated", "annoyed", "angry", "hate", "terrible", "awful"]):
        return "frustrated"
    if any(w in text for w in ["worried", "concerned", "nervous", "afraid", "risk"]):
        return "concerned"
    if any(w in text for w in ["excited", "love", "great", "amazing", "awesome", "fantastic"]):
        return "enthusiastic"
    if any(w in text for w in ["curious", "wondering", "interested", "tell me", "how does"]):
        return "curious"
    return "neutral"


def _tfidf_similarity(query: str, documents: list[str]) -> list[float]:
    """Compute TF-IDF cosine similarity between query and each document."""
    if not documents:
        return []

    all_docs = documents + [query]

    # Build vocabulary
    vocab: dict[str, int] = {}
    for doc in all_docs:
        for word in set(doc.lower().split()):
            if word not in vocab:
                vocab[word] = len(vocab)

    if not vocab:
        return [0.0] * len(documents)

    # Compute TF-IDF vectors
    n_docs = len(all_docs)
    df: Counter[str] = Counter()
    for doc in all_docs:
        for word in set(doc.lower().split()):
            df[word] += 1

    def tfidf_vec(text: str) -> list[float]:
        words = text.lower().split()
        tf: Counter[str] = Counter(words)
        vec = [0.0] * len(vocab)
        for word, count in tf.items():
            if word in vocab:
                idf = math.log(n_docs / (1 + df.get(word, 0)))
                vec[vocab[word]] = (count / max(len(words), 1)) * idf
        return vec

    query_vec = tfidf_vec(query)
    scores = []
    for doc in documents:
        doc_vec = tfidf_vec(doc)
        # Cosine similarity
        dot = sum(a * b for a, b in zip(query_vec, doc_vec))
        norm_q = math.sqrt(sum(a * a for a in query_vec)) or 1.0
        norm_d = math.sqrt(sum(a * a for a in doc_vec)) or 1.0
        scores.append(dot / (norm_q * norm_d))

    return scores
