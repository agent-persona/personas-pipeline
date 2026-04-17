"""Text post-processor that makes a TwinReply read like a human typing in chat.

Opt-in only. `TwinChat.__init__` accepts `humanize_config: HumanizeConfig | None`;
default None means zero behavior change — existing callers are unaffected.

Reads PersonaV1 fields directly (no synthesis-package import):
- communication_style.formality ("casual" | "professional" | "formal")
- communication_style.vocabulary_level ("basic" | "intermediate" | "advanced")
- emotional_profile.baseline_mood (open string, fuzzy-matched)
- vocabulary (list[str], folded into casual-filler pool)

The four knobs, each tunable via HumanizeConfig:
1. Typing pace — mood modulates WPM and jitter width; soft-saturation cap.
2. Grammar slips — gated by formality × vocab_level; zero for formal/advanced.
3. Emoji/filler — optionally emitted as a SEPARATE follow-up chunk.
4. Inter-message pauses — scale with previous-chunk length + mood.

This module is pure: `humanize()` returns a list of `HumanChunk` with timings;
it's up to the caller to actually `await asyncio.sleep(...)` between sends.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Register = Literal["high", "medium", "low"]
Formality = Literal["casual", "professional", "formal"]
VocabLevel = Literal["basic", "intermediate", "advanced"]
MoodKey = Literal["anxious", "calm", "frustrated", "enthusiastic", "optimistic", "neutral"]


# ---------------------------------------------------------------------------
# Config — opt-in knob passed into TwinChat
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HumanizeConfig:
    """Pass this to TwinChat(humanize_config=...) to enable post-processing."""
    enable_typos: bool = False        # off by default — typo injection is opinionated
    seed: Optional[int] = None        # deterministic when set
    enable_grammar_slips: bool = True
    enable_emoji_chunks: bool = True  # separate follow-up message for fillers


@dataclass
class HumanChunk:
    """One outgoing message with the timing that should precede it."""
    text: str
    pre_delay_ms: int   # total time (think + type) before this chunk is sent
    typing_ms: int      # just the 'typing indicator' duration inside pre_delay


# ---------------------------------------------------------------------------
# Style resolution from PersonaV1 dict
# ---------------------------------------------------------------------------

_MOOD_KEYWORDS: tuple[tuple[str, MoodKey], ...] = (
    ("anx", "anxious"), ("nerv", "anxious"), ("worr", "anxious"),
    ("frust", "frustrated"), ("angry", "frustrated"), ("irrit", "frustrated"),
    ("enthus", "enthusiastic"), ("excit", "enthusiastic"), ("eager", "enthusiastic"),
    ("optim", "optimistic"), ("cheer", "optimistic"),
    ("calm", "calm"), ("relax", "calm"), ("steady", "calm"),
)

_BASE_FILLERS: dict[Register, tuple[str, ...]] = {
    "high": (),
    "medium": ("hmm", "mm", "ah", "I see", "huh"),
    "low": ("lol", "haha", "fr", "tbh", "lmao", "ngl", ":)", ":/"),
}


def _normalize_mood(raw: Optional[str]) -> MoodKey:
    if not raw:
        return "neutral"
    text = raw.lower()
    for kw, bucket in _MOOD_KEYWORDS:
        if kw in text:
            return bucket
    return "neutral"


def _register_for(formality: str, vocab_level: str) -> Register:
    if formality == "formal" or vocab_level == "advanced":
        return "high"
    if formality == "casual":
        return "low"
    return "medium"


def _merge_fillers(register: Register, persona_vocab: list[str]) -> tuple[str, ...]:
    base = _BASE_FILLERS[register]
    if register == "high" or not persona_vocab:
        return base
    extras = tuple(
        w.strip().lower()
        for w in persona_vocab
        if w and len(w) <= 12 and " " not in w.strip()
    )
    seen: set[str] = set()
    merged: list[str] = []
    for w in (*base, *extras):
        if w not in seen:
            seen.add(w)
            merged.append(w)
    return tuple(merged)


@dataclass(frozen=True)
class _Style:
    register: Register
    formality: Formality
    vocab_level: VocabLevel
    mood: MoodKey
    fillers: tuple[str, ...]


def _resolve_style(persona: dict[str, Any]) -> _Style:
    cs = persona.get("communication_style") or {}
    ep = persona.get("emotional_profile") or {}
    formality = cs.get("formality", "professional")
    vocab_level = cs.get("vocabulary_level", "intermediate")
    if formality not in ("casual", "professional", "formal"):
        formality = "professional"
    if vocab_level not in ("basic", "intermediate", "advanced"):
        vocab_level = "intermediate"
    register = _register_for(formality, vocab_level)
    return _Style(
        register=register,
        formality=formality,  # type: ignore[arg-type]
        vocab_level=vocab_level,  # type: ignore[arg-type]
        mood=_normalize_mood(ep.get("baseline_mood")),
        fillers=_merge_fillers(register, persona.get("vocabulary") or []),
    )


# ---------------------------------------------------------------------------
# Deterministic per-persona profile (WPM / lowercase bias / emoji rate)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Profile:
    wpm: int
    emoji_rate: float
    chunk_aggressiveness: float
    lowercase_bias: float
    trail_off_bias: float


def _profile_for(persona: dict[str, Any]) -> _Profile:
    """Stable per-persona typing habits derived from persona name hash."""
    name = persona.get("name", "unknown")
    h = hashlib.sha1(name.encode("utf-8")).digest()
    return _Profile(
        wpm=45 + int(h[0] / 255 * 55),                 # 45..100
        emoji_rate=0.05 + (h[2] / 255) * 0.25,          # 5%..30%
        chunk_aggressiveness=0.3 + (h[3] / 255) * 0.5,  # 0.3..0.8
        lowercase_bias=0.3 + (h[4] / 255) * 0.5,
        trail_off_bias=0.05 + (h[5] / 255) * 0.15,
    )


# ---------------------------------------------------------------------------
# AI-ism scrub + contractions
# ---------------------------------------------------------------------------

_AI_OPENERS = [
    r"\bGreat question[!.]?\s*",
    r"\bThat's a (really |very )?(interesting|great|good|thoughtful) (question|point|observation)[!.]?\s*",
    r"\bCertainly[!,.]?\s*", r"\bAbsolutely[!,.]?\s*",
    r"\bI appreciate (your|the) (question|thought|perspective)[!.,]?\s*",
    r"\b(It's|It is) worth noting that\s*",
    r"\b(It's|It is) important to (consider|note|remember) that\s*",
    r"\bI hope (this|that) helps[!.]?\s*", r"\bIn conclusion,?\s*",
    r"\bFurthermore,?\s*", r"\bMoreover,?\s*", r"\bAdditionally,?\s*",
]

_AI_WORD_SWAPS = {
    r"\bdelve\s+into\b": "look at", r"\bleverage\b": "use", r"\butilize\b": "use",
    r"\bfacilitate\b": "help", r"\bendeavor\b": "try", r"\bcommence\b": "start",
    r"\bsubsequently\b": "then", r"\bnumerous\b": "lots of",
    r"\bmyriad\b": "lots of", r"\btapestry\b": "mix",
}

_CONTRACTIONS = {
    r"\bdo not\b": "don't", r"\bdoes not\b": "doesn't", r"\bdid not\b": "didn't",
    r"\bis not\b": "isn't", r"\bare not\b": "aren't", r"\bwas not\b": "wasn't",
    r"\bcannot\b": "can't", r"\bcan not\b": "can't", r"\bwill not\b": "won't",
    r"\bwould not\b": "wouldn't", r"\bcould not\b": "couldn't",
    r"\bshould not\b": "shouldn't", r"\bhave not\b": "haven't",
    r"\bI am\b": "I'm", r"\byou are\b": "you're", r"\bthey are\b": "they're",
    r"\bit is\b": "it's", r"\bthat is\b": "that's", r"\bwhat is\b": "what's",
}


def _strip_ai(text: str) -> str:
    out = text
    for pat in _AI_OPENERS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    for pat, rep in _AI_WORD_SWAPS.items():
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    out = out.replace(" — ", " - ").replace("—", "-")
    out = out.replace(" – ", " - ").replace("–", "-")
    out = out.replace(";", ",")
    out = re.sub(r"^\s*[-*]\s+", "", out, flags=re.MULTILINE)
    out = re.sub(r"^\s*\d+\.\s+", "", out, flags=re.MULTILINE)
    return re.sub(r"[ \t]+", " ", out).strip()


def _contractions(text: str) -> str:
    out = text
    for pat, rep in _CONTRACTIONS.items():
        out = re.sub(pat, rep, out)
    return out


# ---------------------------------------------------------------------------
# Grammar slips — gated by formality × vocab_level
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SlipRates:
    lowercase_i: float
    drop_apostrophe: float
    your_youre: float
    alot: float
    drop_trailing_period: float


_SLIP_ZERO = _SlipRates(0, 0, 0, 0, 0)
_SLIP_CASUAL_BASIC = _SlipRates(0.50, 0.35, 0.25, 0.50, 0.35)
_SLIP_CASUAL_INTER = _SlipRates(0.30, 0.20, 0.08, 0.20, 0.25)
_SLIP_PROF_BASIC = _SlipRates(0.10, 0.10, 0.03, 0.08, 0.10)
_SLIP_PROF_INTER = _SlipRates(0.05, 0.05, 0.0, 0.03, 0.08)

_APOSTROPHE_DROPS = {
    "I'm": "im", "i'm": "im", "don't": "dont", "doesn't": "doesnt",
    "didn't": "didnt", "can't": "cant", "won't": "wont", "isn't": "isnt",
    "it's": "its", "that's": "thats", "I'll": "ill",
}


def _slip_rates(formality: str, vocab_level: str) -> _SlipRates:
    if vocab_level == "advanced" or formality == "formal":
        return _SLIP_ZERO
    if formality == "casual":
        return _SLIP_CASUAL_BASIC if vocab_level == "basic" else _SLIP_CASUAL_INTER
    return _SLIP_PROF_BASIC if vocab_level == "basic" else _SLIP_PROF_INTER


def _apply_slips(text: str, rates: _SlipRates, rng: random.Random) -> str:
    if rates == _SLIP_ZERO:
        return text
    out = text
    if rates.lowercase_i > 0:
        out = re.sub(r" I ",
                     lambda m: " i " if rng.random() < rates.lowercase_i else m.group(0),
                     out)
    if rates.drop_apostrophe > 0:
        for word, rep in _APOSTROPHE_DROPS.items():
            out = re.sub(r"\b" + re.escape(word) + r"\b",
                         lambda m, r=rep: r if rng.random() < rates.drop_apostrophe else m.group(0),
                         out)
    if rates.your_youre > 0:
        out = re.sub(r"\byou're\b",
                     lambda m: "your" if rng.random() < rates.your_youre else m.group(0),
                     out, flags=re.IGNORECASE)
    if rates.alot > 0:
        out = re.sub(r"\ba lot\b",
                     lambda m: "alot" if rng.random() < rates.alot else m.group(0),
                     out, flags=re.IGNORECASE)
    if rates.drop_trailing_period > 0 and out.endswith(".") and rng.random() < rates.drop_trailing_period:
        out = out[:-1]
    return out


# ---------------------------------------------------------------------------
# Chunking + per-chunk casualification
# ---------------------------------------------------------------------------

# Split on sentence-ending punct followed by whitespace + any word/quote char.
# Old version required a capital letter after the period, but AI-opener scrubs
# (e.g. "It is worth noting that ") leave the next sentence starting lowercase,
# which collapsed multi-sentence replies into single chunks.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\w\"'(])")


def _split_chunks(text: str, aggressiveness: float, rng: random.Random) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sentences) <= 1:
        return [text]
    total = len(text)
    if total < 80:
        target = 1
    elif total < 180:
        target = 2 if rng.random() < aggressiveness else 1
    elif total < 320:
        target = 2 if rng.random() < 0.5 else 3
    else:
        target = 3
    target = min(target, len(sentences))
    if target <= 1:
        return [text]
    # Preserve sentence ORDER. Pick `target-1` split points that divide the
    # total length roughly evenly. The previous "pack into shortest" algo
    # reordered sentences, which produced nonsensical chunk sequences.
    lens = [len(s) for s in sentences]
    total = sum(lens)
    ideal = [total * (i + 1) / target for i in range(target - 1)]
    cuts: list[int] = []
    running = 0
    j = 0
    for i, L in enumerate(lens):
        running += L
        if j < len(ideal) and running >= ideal[j] and i < len(sentences) - 1:
            cuts.append(i + 1)
            j += 1
    groups: list[list[str]] = []
    prev = 0
    for c in cuts:
        groups.append(sentences[prev:c])
        prev = c
    groups.append(sentences[prev:])
    return [" ".join(g).strip() for g in groups if g]


def _maybe_lowercase_start(text: str, bias: float, rng: random.Random) -> str:
    if not text:
        return text
    words = text.split(" ", 2)
    if len(words) >= 2 and words[0][:1].isupper() and words[1][:1].isupper():
        return text
    if rng.random() < bias:
        return text[0].lower() + text[1:]
    return text


def _maybe_trail_off(text: str, bias: float, rng: random.Random) -> str:
    if text.endswith("."):
        r = rng.random()
        if r < bias:
            return text[:-1] + "..."
        if len(text) < 60 and r < bias + 0.35:
            return text[:-1]
    return text


# ---------------------------------------------------------------------------
# Mood-based timing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _MoodTiming:
    wpm_mult: float
    burstiness: float
    inter_msg_mult: float
    think_mult: float


_MOOD_TIMING: dict[str, _MoodTiming] = {
    "anxious":      _MoodTiming(1.20, 0.55, 0.75, 0.7),
    "frustrated":   _MoodTiming(1.10, 0.50, 1.35, 0.9),
    "enthusiastic": _MoodTiming(1.15, 0.30, 0.80, 0.8),
    "optimistic":   _MoodTiming(1.00, 0.25, 1.00, 1.0),
    "calm":         _MoodTiming(0.90, 0.20, 1.15, 1.2),
    "neutral":      _MoodTiming(1.00, 0.25, 1.00, 1.0),
}


def _think_time(reply: str, mood: _MoodTiming, rng: random.Random) -> float:
    words = max(1, len(reply.split()))
    base = (0.6 + 0.025 * words) * mood.think_mult * rng.uniform(0.75, 1.3)
    return max(0.5, min(4.0, base))


def _type_time(chunk: str, profile: _Profile, mood: _MoodTiming, rng: random.Random) -> float:
    words = max(1, len(chunk.split()))
    base = words * (60.0 / (profile.wpm * mood.wpm_mult))
    jitter = 1.0 + (rng.random() - 0.5) * (0.3 + mood.burstiness)
    base *= jitter
    soft_cap = 2.5 + 0.05 * words
    if base > soft_cap:
        base = soft_cap + (base - soft_cap) * 0.35
    return max(0.4, base)


def _inter_msg_pause(prev_chunk: str, mood: _MoodTiming, rng: random.Random) -> float:
    base = rng.uniform(1.5, 4.0)
    prev_factor = min(2.0, max(0.5, len(prev_chunk) / 80.0))
    pause = base * prev_factor * mood.inter_msg_mult
    if rng.random() < 0.08:
        pause += rng.uniform(8.0, 15.0)
    return pause


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def humanize(
    text: str,
    persona: dict[str, Any],
    config: Optional[HumanizeConfig] = None,
) -> list[HumanChunk]:
    """Transform one TwinReply text into a list of chunks with timing.

    `persona` is a PersonaV1.model_dump() dict — same shape `TwinChat` already
    accepts for system-prompt construction, so callers pay no extra cost.
    """
    cfg = config or HumanizeConfig()
    rng = random.Random(cfg.seed) if cfg.seed is not None else random.Random()

    style = _resolve_style(persona)
    profile = _profile_for(persona)
    mood_t = _MOOD_TIMING.get(style.mood, _MOOD_TIMING["neutral"])

    cleaned = _strip_ai(text)
    cleaned = _contractions(cleaned)
    if cfg.enable_grammar_slips:
        cleaned = _apply_slips(cleaned, _slip_rates(style.formality, style.vocab_level), rng)

    raw_chunks = _split_chunks(cleaned, profile.chunk_aggressiveness, rng)
    if not raw_chunks:
        return []

    out: list[HumanChunk] = []
    inline_emoji_fired = False
    emoji_mult = {"high": 0.0, "medium": 0.35, "low": 1.0}[style.register]
    lc_mult = {"high": 0.0, "medium": 0.4, "low": 1.0}[style.register]
    trail_mult = {"high": 0.3, "medium": 0.7, "low": 1.0}[style.register]

    for i, chunk in enumerate(raw_chunks):
        c = chunk
        base_lc = profile.lowercase_bias if i > 0 else profile.lowercase_bias * 0.6
        c = _maybe_lowercase_start(c, base_lc * lc_mult, rng)
        c = _maybe_trail_off(c, profile.trail_off_bias * trail_mult, rng)
        if i == len(raw_chunks) - 1 and style.fillers:
            inline_prob = profile.emoji_rate * emoji_mult * 0.6
            if rng.random() < inline_prob:
                c = c + " " + rng.choice(style.fillers)
                inline_emoji_fired = True

        typing = _type_time(c, profile, mood_t, rng)
        pre = (_think_time(text, mood_t, rng) + typing) if i == 0 \
            else (_inter_msg_pause(raw_chunks[i - 1], mood_t, rng) + typing)
        out.append(HumanChunk(
            text=c,
            pre_delay_ms=int(pre * 1000),
            typing_ms=int(typing * 1000),
        ))

    if (
        cfg.enable_emoji_chunks
        and style.fillers
        and not inline_emoji_fired
        and rng.random() < profile.emoji_rate * emoji_mult * 0.9
    ):
        filler = rng.choice(style.fillers)
        typing = rng.uniform(0.3, 0.9)
        pre = rng.uniform(1.0, 2.5) * mood_t.inter_msg_mult + typing
        out.append(HumanChunk(
            text=filler,
            pre_delay_ms=int(pre * 1000),
            typing_ms=int(typing * 1000),
        ))

    return out
