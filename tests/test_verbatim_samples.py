"""Fixture tests for the verbatim-voice pipeline primitive.

Validates the segmentation-level extraction + style-coherent selection
in isolation, using synthetic RawRecord inputs shaped like the crawled
Slack data (`body` field in payload). When crawler-produced data is
swapped in for validation, the fixtures here are the reference baseline.

Does not exercise synthesis or twin chat — those are integration-level
and run on real clusters. The unit tests here are fast and deterministic.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "segmentation"))

from segmentation.engine.summarizer import (  # noqa: E402
    _candidate_texts,
    _extract_verbatim_samples,
    _is_likely_bot_or_system,
    _pick_style_coherent,
    _style_distance,
    _style_signature,
)
from segmentation.models.record import RawRecord  # noqa: E402


# --- Helpers ---------------------------------------------------------------

def _slack_record(record_id: str, body: str, source: str = "slack") -> RawRecord:
    """Build a RawRecord shaped like the crawled Slack data (body in payload)."""
    return RawRecord(
        record_id=record_id,
        user_id=f"user_{record_id}",
        tenant_id="tenant_test",
        source=source,
        timestamp="2026-04-17T20:00:00Z",
        payload={"body": body},
        behaviors=["chat"],
        pages=[],
    )


# --- Bot / system filter ---------------------------------------------------

def test_bot_filter_catches_welcome_messages():
    assert _is_likely_bot_or_system("Welcome to the channel! Please read the pinned rules.")
    assert _is_likely_bot_or_system("You've been added to the #general channel")
    assert _is_likely_bot_or_system("This channel is for engineering discussions only.")


def test_bot_filter_catches_join_leave():
    assert _is_likely_bot_or_system("@alice has joined the channel")
    assert _is_likely_bot_or_system("bob has left the server")
    assert _is_likely_bot_or_system("carol pinned a message")


def test_bot_filter_catches_all_caps_announcements():
    assert _is_likely_bot_or_system(
        "URGENT: SERVER MAINTENANCE TONIGHT AT 2AM UTC — ALL SERVICES DOWN"
    )


def test_bot_filter_catches_channel_broadcasts():
    """@channel / @here / @everyone messages are admin announcements, not
    peer chat. Exclude from the voice-sample pool. Found empirically on
    real Slack crawls where these were crowding out authentic messages."""
    assert _is_likely_bot_or_system(
        "@channel You should be receiving a Ramp card with $450 on it."
    )
    assert _is_likely_bot_or_system(
        "@hereOffice hours will begin in a few minutes."
    )
    assert _is_likely_bot_or_system(
        "Hey Challengers! @channel Here's the Orientation Deck for you to review."
    )
    assert _is_likely_bot_or_system(
        "@everyone please review the new guidelines by Friday."
    )
    # Normal messages that just reference a channel should still pass through
    assert not _is_likely_bot_or_system(
        "i posted my MVP link in #general if anyone wants to try it"
    )


def test_bot_filter_lets_real_messages_through():
    assert not _is_likely_bot_or_system(
        "hey, just wrapped up that webhook issue. the retry logic was broken."
    )
    assert not _is_likely_bot_or_system(
        "I think we should use idempotent handlers on the consumer side honestly"
    )


# --- Style signature -------------------------------------------------------

def test_style_signature_is_deterministic():
    s1 = _style_signature("hey, whats up? not much tbh")
    s2 = _style_signature("hey, whats up? not much tbh")
    assert s1 == s2


def test_style_signature_distinguishes_casual_from_formal():
    casual = _style_signature("hey, whats up? yeah nah just hanging")
    formal = _style_signature(
        "Dear Sir or Madam, I am writing to inquire about the status of my recent invoice number 48721."
    )
    # Casual fragments don't all start with capitals; formal does (index 6)
    assert formal[6] > casual[6]
    # Casual has more unterminated fragments (index 4)
    assert casual[4] > formal[4]
    # Total distance between them should be substantial
    assert _style_distance(casual, formal) > 0.2


# --- Style-coherent selection ---------------------------------------------

def test_pick_style_coherent_returns_all_when_under_k():
    inputs = ["hey", "yeah", "nah"]
    out = _pick_style_coherent(inputs, k=8)
    assert out == inputs


def test_pick_style_coherent_prefers_the_majority_voice():
    """In a heterogeneous input, picker should return the stylistic majority."""
    # 6 casual fragments
    casual = [
        "hey whats up, just checking in",
        "yeah nah that wont work for us honestly",
        "lol thats wild, didnt see it coming",
        "ok cool, ill take a look later today",
        "fair point, hadn't thought of that tbh",
        "sure sounds good to me, lets do it",
    ]
    # 2 formal stylistic outliers
    formal = [
        "Dear team, I am writing to inform you of the upcoming maintenance window scheduled for next Tuesday at 14:00 UTC. Please plan accordingly.",
        "In accordance with our standard operating procedures, we must formally document every incident before resolution can proceed.",
    ]
    out = _pick_style_coherent(casual + formal, k=6)
    # The 6 casual ones should win — they're tightly clustered
    assert all(s in casual for s in out), f"picker included formal outliers: {out}"


def test_candidate_texts_walks_nested_payload_fields():
    """Style-bearing text can live anywhere in payload — not just 'message'."""
    records = [
        # body field (Slack crawler shape)
        _slack_record("r1", "hey, yeah that makes sense to me"),
        # nested under metadata.comment
        RawRecord(
            record_id="r2", user_id="u2", tenant_id="t", source="zendesk",
            timestamp="2026-04-17T20:00:00Z",
            payload={"metadata": {"comment": "this is really frustrating, the app keeps crashing"}},
            behaviors=[], pages=[],
        ),
        # too short — should be filtered
        _slack_record("r3", "ok"),
        # bot message — should be filtered
        _slack_record("r4", "Welcome to the channel! Please read the pinned rules."),
    ]
    texts = _candidate_texts(records)
    assert "hey, yeah that makes sense to me" in texts
    assert "this is really frustrating, the app keeps crashing" in texts
    assert "ok" not in texts
    assert not any("Welcome" in t for t in texts)


# --- End-to-end extraction -------------------------------------------------

def test_extract_empty_when_no_text_records():
    """GA4-style behavior-only records produce empty verbatim_samples."""
    records = [
        RawRecord(
            record_id=f"r{i}", user_id=f"u{i}", tenant_id="t", source="ga4",
            timestamp="2026-04-17T20:00:00Z",
            payload={"event": "page_view", "duration_s": 120, "scroll_depth": 0.8},
            behaviors=["viewed_docs"], pages=["/docs/api"],
        )
        for i in range(10)
    ]
    out = _extract_verbatim_samples(records, k=8)
    assert out == []


def test_extract_returns_casual_chat_verbatim_discord_style():
    """Chat-register data (shape matches crawled Slack cohort data) produces
    casual-voiced verbatim samples, unfiltered, in their original register.
    """
    records = [
        _slack_record("m0", "hey whats up, starting cohort today"),
        _slack_record("m1", "lol same, just grabbing coffee before the kickoff"),
        _slack_record("m2", "yeah nah i skipped breakfast, big mistake"),
        _slack_record("m3", "anyone got the link to the zoom? cant find it in my inbox"),
        _slack_record("m4", "ok found it, nvm"),  # too short, filtered
        _slack_record("m5", "haha typical monday, everything breaks at once"),
        _slack_record("m6", "fair. ill try to debug it after standup"),
        _slack_record("m7", "sure thing, lemme know if you need a hand with anything"),
        # bot message mixed in — must be excluded
        _slack_record("m8", "Welcome to the #cohort-main channel! Read pinned message for rules."),
    ]
    out = _extract_verbatim_samples(records, k=6)
    # 7 eligible candidates (m0-m3, m5-m7); all should be casual human messages
    assert len(out) >= 6
    assert not any("Welcome" in s for s in out), "bot message leaked in"
    assert not any(s == "ok found it, nvm" for s in out), "below-threshold leaked in"
    # Preserve register (lowercase, contractions, no apostrophes)
    joined = " ".join(out)
    assert "whats" in joined or "cant" in joined or "wont" in joined or "nvm" in joined


def test_extract_returns_formal_verbatim_when_corpus_is_formal():
    """Formal register in → formal register samples out (no forced casualization)."""
    records = [
        _slack_record("t0", "I would like to confirm our meeting at 3pm tomorrow regarding the quarterly review."),
        _slack_record("t1", "Please find attached the revised proposal for your review. We have incorporated the feedback from last week."),
        _slack_record("t2", "Thank you for your prompt response. I appreciate the detailed breakdown of the next steps."),
        _slack_record("t3", "Could you please clarify the scope of the deliverables before we proceed further?"),
        _slack_record("t4", "Apologies for the delay in getting back to you. I have been reviewing the documentation in depth."),
        _slack_record("t5", "We should schedule a follow-up call to discuss the implementation timeline in more detail."),
    ]
    out = _extract_verbatim_samples(records, k=5)
    assert len(out) >= 5
    # Verify the register survived — formal punctuation, capitalization
    caps_start = sum(1 for s in out if s and s[0].isupper())
    assert caps_start >= 4, f"expected formal samples to start with capitals, got {out}"


def test_extract_is_deterministic_same_input():
    """Same input must produce same verbatim_samples — important for reproducibility."""
    records = [
        _slack_record(f"r{i}", f"message number {i} with enough characters to pass the filter")
        for i in range(12)
    ]
    out1 = _extract_verbatim_samples(records, k=6)
    out2 = _extract_verbatim_samples(records, k=6)
    assert out1 == out2


# --- build_cluster_data integration ---------------------------------------

def test_build_cluster_data_populates_verbatim_samples():
    """The full build_cluster_data path emits verbatim_samples in the dict."""
    from segmentation.engine.summarizer import build_cluster_data
    from segmentation.models.features import UserFeatures

    records = [
        _slack_record(f"r{i}", f"hey this is message {i} with real human voice and enough text")
        for i in range(10)
    ]
    users = [
        UserFeatures(
            user_id=f"u{i}",
            tenant_id="tenant_test",
            record_ids=[f"r{i}"],
            behaviors={"chat"},
            pages=set(),
            numeric_features={},
            categorical_features={},
            set_features={},
        )
        for i in range(10)
    ]
    result = build_cluster_data(
        cluster_users=users,
        all_records=records,
        tenant_id="tenant_test",
    )
    assert "verbatim_samples" in result
    assert isinstance(result["verbatim_samples"], list)
    assert len(result["verbatim_samples"]) >= 6  # at least k_default, bounded by pool


def test_build_cluster_data_empty_verbatim_on_behavior_only():
    """Cluster with no text-bearing records produces empty verbatim_samples."""
    from segmentation.engine.summarizer import build_cluster_data
    from segmentation.models.features import UserFeatures

    records = [
        RawRecord(
            record_id=f"r{i}", user_id=f"u{i}", tenant_id="t", source="ga4",
            timestamp="2026-04-17T20:00:00Z",
            payload={"event": "page_view", "duration_s": 120},
            behaviors=["viewed_docs"], pages=["/docs"],
        )
        for i in range(5)
    ]
    users = [
        UserFeatures(
            user_id=f"u{i}",
            tenant_id="tenant_test",
            record_ids=[f"r{i}"],
            behaviors={"viewed_docs"},
            pages={"/docs"},
            numeric_features={"session_duration": 120.0},
            categorical_features={},
            set_features={},
        )
        for i in range(5)
    ]
    result = build_cluster_data(
        cluster_users=users,
        all_records=records,
        tenant_id="tenant_test",
    )
    assert result["verbatim_samples"] == []
