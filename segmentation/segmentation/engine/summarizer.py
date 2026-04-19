from __future__ import annotations

import re
import uuid
from collections import Counter

from segmentation.models.features import UserFeatures
from segmentation.models.record import RawRecord

# How many sample records to include in the cluster handoff to synthesis
DEFAULT_SAMPLE_SIZE = 12

# How many verbatim text samples to extract for voice grounding.
# These are raw strings (not records) carried end-to-end through the pipeline.
# Small enough that synthesis prompt stays under context pressure; large
# enough that the style signature is robust to a few outliers.
DEFAULT_VERBATIM_SAMPLE_SIZE = 8

# Character bounds for a message to be eligible as a voice sample.
# Below 20 chars = too terse to signal register ("ok", "thanks").
# Above 2000 chars = long-form content (docs, manifestos) that would
# dominate the style signature and blow prompt budget.
VERBATIM_MIN_LEN = 20
VERBATIM_MAX_LEN = 2000


def build_cluster_data(
    cluster_users: list[UserFeatures],
    all_records: list[RawRecord],
    tenant_id: str,
    tenant_industry: str | None = None,
    tenant_product: str | None = None,
    existing_persona_names: list[str] | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> dict:
    """Build a cluster summary in the exact JSON shape that synthesis expects.

    The output dict is the contract: it conforms to synthesis's `ClusterData`
    Pydantic model. This is the integration seam — segmentation produces it,
    synthesis consumes it. The two services are decoupled by this JSON shape.

    Returns a dict (not a Pydantic model) so segmentation has no runtime
    dependency on synthesis. Either side can validate against the shared
    schema independently.
    """
    cluster_user_ids = {u.user_id for u in cluster_users}
    cluster_record_ids: set[str] = set()
    for u in cluster_users:
        cluster_record_ids.update(u.record_ids)

    # Pull the raw records that belong to this cluster
    cluster_records = [r for r in all_records if r.record_id in cluster_record_ids]

    # Aggregate behaviors and pages across the whole cluster
    behavior_counter: Counter[str] = Counter()
    page_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    for r in cluster_records:
        behavior_counter.update(r.behaviors)
        page_counter.update(r.pages)
        source_counter[r.source] += 1

    top_behaviors = [b for b, _ in behavior_counter.most_common(8)]
    top_pages = [p for p, _ in page_counter.most_common(8)]

    # Filter admin/bot/system records out of the evidence pool before
    # picking the sample. Without this, sample_records for cluster-heavy
    # tenants ends up dominated by @channel broadcasts (which are stylistically
    # coherent and thus selected by behavior-coverage logic), which both
    # misrepresents the cluster and confuses synthesis — the LLM is asked to
    # cite admin announcements as evidence for a peer persona, and it
    # responds by silently omitting source_evidence entirely.
    #
    # Fall back to unfiltered records if filtering removes everything
    # (behavior-only clusters where no record has text-shaped payload).
    _filtered = [
        r for r in cluster_records
        if not any(
            isinstance(v, str) and _is_likely_bot_or_system(v)
            for v in _iter_string_values(r.payload)
        )
    ]
    _pool = _filtered if _filtered else cluster_records
    sample = _pick_representative_sample(_pool, top_behaviors, sample_size)

    # Extract style-coherent verbatim text samples from this cluster's records.
    # These are raw strings — never LLM-rewritten — carried unchanged through
    # synthesis into PersonaV1.verbatim_samples, and read by every downstream
    # generator as voice-grounding context. Empty list when cluster has no
    # text-bearing records (pipeline falls back to pre-verbatim behavior).
    verbatim_samples = _extract_verbatim_samples(
        cluster_records, k=DEFAULT_VERBATIM_SAMPLE_SIZE,
    )

    cluster_id = f"clust_{uuid.uuid4().hex[:12]}"

    return {
        "cluster_id": cluster_id,
        "tenant": {
            "tenant_id": tenant_id,
            "industry": tenant_industry,
            "product_description": tenant_product,
            "existing_persona_names": existing_persona_names or [],
        },
        "summary": {
            "cluster_size": len(cluster_user_ids),
            "top_behaviors": top_behaviors,
            "top_pages": top_pages,
            "conversion_rate": None,
            "avg_session_duration_seconds": _compute_avg_session_duration(cluster_users),
            "top_referrers": [],
            "extra": _build_extra(cluster_users, cluster_records, source_counter),
        },
        "sample_records": [
            {
                "record_id": r.record_id,
                "source": r.source,
                "timestamp": r.timestamp,
                "payload": r.payload,
            }
            for r in sample
        ],
        "verbatim_samples": verbatim_samples,
        "enrichment": _extract_enrichment(cluster_records),
    }


_INTEGRATION_KEYWORDS = {"api", "webhook", "integration", "github", "slack"}


def _extract_enrichment(records: list[RawRecord]) -> dict:
    """Extract firmographic, intent, and technographic signals from raw records."""

    # --- firmographic (HubSpot) ---
    titles: list[str] = []
    company_sizes: list[str] = []
    industries: list[str] = []
    for r in records:
        if r.source != "hubspot":
            continue
        title = r.payload.get("contact_title")
        if title:
            titles.append(title)
        size = r.payload.get("company_size")
        if size:
            company_sizes.append(size)
        industry = r.payload.get("industry")
        if industry:
            industries.append(industry)

    firmographic: dict = {}
    if titles:
        firmographic["titles"] = list(dict.fromkeys(titles))
    if company_sizes:
        firmographic["company_size"] = Counter(company_sizes).most_common(1)[0][0]
    if industries:
        firmographic["industries"] = list(dict.fromkeys(industries))

    # --- intent_signals (Intercom) ---
    intent_signals: list[str] = []
    for r in records:
        if r.source != "intercom":
            continue
        topic = r.payload.get("topic", "")
        message = r.payload.get("message", "")
        preview = message[:100]
        if topic or preview:
            intent_signals.append(f"{topic}: {preview}..." if preview else topic)

    # --- technographic (GA4) ---
    integration_pages: list[str] = []
    tech_events: list[str] = []
    for r in records:
        if r.source != "ga4":
            continue
        for page in r.pages:
            if any(kw in page.lower() for kw in _INTEGRATION_KEYWORDS):
                integration_pages.append(page)
        for behavior in r.behaviors:
            tech_events.append(behavior)

    technographic: dict = {}
    if integration_pages:
        technographic["integration_pages"] = list(dict.fromkeys(integration_pages))
    if tech_events:
        technographic["tech_events"] = list(dict.fromkeys(tech_events))

    return {
        "firmographic": firmographic,
        "intent_signals": intent_signals,
        "technographic": technographic,
        "extra": {},
    }


def _compute_avg_session_duration(users: list[UserFeatures]) -> float | None:
    """Compute average session duration from numeric features, or None."""
    durations = [
        u.numeric_features["session_duration"]
        for u in users
        if "session_duration" in u.numeric_features
    ]
    if not durations:
        return None
    return sum(durations) / len(durations)


def _build_extra(
    users: list[UserFeatures],
    cluster_records: list[RawRecord],
    source_counter: Counter,
) -> dict:
    """Build the extra dict with source breakdown and optional typed features."""
    extra: dict = {
        "n_records": len(cluster_records),
        "source_breakdown": dict(source_counter),
    }

    # Add typed feature aggregates if any user has them
    has_typed = any(
        u.numeric_features or u.categorical_features or u.set_features
        for u in users
    )
    if has_typed:
        # Numeric averages
        numeric_avgs: dict[str, float] = {}
        numeric_keys: set[str] = set()
        for u in users:
            numeric_keys.update(u.numeric_features.keys())
        for key in numeric_keys:
            values = [u.numeric_features[key] for u in users if key in u.numeric_features]
            if values:
                numeric_avgs[key] = sum(values) / len(values)

        # Categorical modes
        categorical_modes: dict[str, str] = {}
        cat_keys: set[str] = set()
        for u in users:
            cat_keys.update(u.categorical_features.keys())
        for key in cat_keys:
            values = [u.categorical_features[key] for u in users if key in u.categorical_features]
            if values:
                counts = Counter(values)
                categorical_modes[key] = counts.most_common(1)[0][0]

        # Set unions
        set_unions: dict[str, list[str]] = {}
        set_keys: set[str] = set()
        for u in users:
            set_keys.update(u.set_features.keys())
        for key in set_keys:
            union: set[str] = set()
            for u in users:
                if key in u.set_features:
                    union |= u.set_features[key]
            set_unions[key] = sorted(union)

        extra["typed_features"] = {
            "numeric_averages": numeric_avgs,
            "categorical_modes": categorical_modes,
            "set_unions": set_unions,
        }

    return extra


def _pick_representative_sample(
    records: list[RawRecord],
    top_behaviors: list[str],
    sample_size: int,
) -> list[RawRecord]:
    """Pick a sample that covers the top behaviors before filling with random."""
    if len(records) <= sample_size:
        return records

    picked: list[RawRecord] = []
    picked_ids: set[str] = set()

    # First pass: pick one record per top behavior
    for behavior in top_behaviors:
        for r in records:
            if r.record_id in picked_ids:
                continue
            if behavior in r.behaviors:
                picked.append(r)
                picked_ids.add(r.record_id)
                break
        if len(picked) >= sample_size:
            return picked

    # Second pass: fill the rest in original order
    for r in records:
        if len(picked) >= sample_size:
            break
        if r.record_id not in picked_ids:
            picked.append(r)
            picked_ids.add(r.record_id)

    return picked


# ---------------------------------------------------------------------------
# Verbatim voice samples — style-coherent raw text carried through pipeline
# ---------------------------------------------------------------------------
#
# Design notes:
#   - Extract from ANY string-valued payload field (not just payload.message);
#     real crawlers put text under 'body', 'content', 'text', 'comment',
#     'question', 'answer', etc. Walk the dict.
#   - Filter out likely bot/system/templated text via cheap text heuristics.
#     Connector-level filtering (author.is_bot flags etc.) is better where
#     available; this is a defensive fallback.
#   - Compute a cheap per-message style signature — pure string stats, no
#     embeddings, no LLM calls. Enough to detect major register differences
#     (casual vs. formal, fragment-heavy vs. essay-style).
#   - Pick the subset whose pairwise style distances are tightest. Goal is
#     register coherence, not diversity.
#   - If the cluster has no text at all, return []. Every downstream consumer
#     must handle the empty case as "no voice signal" and fall back to its
#     prior behavior.


# Heuristic fingerprints for bot / system / templated / broadcast messages
# that shouldn't be used to ground a human persona's voice.
#
# Segmentation-level filtering is a DEFENSIVE FALLBACK — the right home for
# platform-aware filtering is the connector/crawler (which has is_bot,
# is_admin, message_type signals). These heuristics catch the most common
# cross-platform patterns so we don't silently regress on a new data source.
#
# Patterns are intentionally conservative — a handful of admin announcements
# slipping through into the voice pool is better than filtering out real
# peer chat. Callers can pass a custom is_bot_filter to extract_verbatim_samples
# for platform-specific rules.
_BOT_PATTERNS = (
    # Join / leave / membership events (all chat platforms)
    r"\bhas\s+(joined|left)\s+the\s+(channel|server|group|room|space|team)\b",
    r"\bpinned\s+a\s+message\b",
    r"\bwelcome\s+to\s+(the\s+)?(channel|server|community|workspace|discord|slack|team|room|subreddit)\b",
    r"\byou('?ve|\s+have)\s+been\s+(added|invited)\s+to\b",
    r"\bthis\s+(channel|room|space|thread)\s+is\s+for\b",
    r"\brule\s+(violation|enforcement)\b",
    r"^@?(system|admin|mod|moderator|bot|automod|automoderator)\b",
    r"\bauto-?(reply|response|generated|moderated|modded)\b",
    r"^\s*\[?\s*(bot|system|auto|mod)\s*[\]:]",
    # Platform-wide mention broadcasts across chat platforms. Distinctive
    # enough tokens that unanchored matching (no \b) is safe — crawl exports
    # often concatenate without a space, e.g. "@hereOffice hours will begin..."
    #   Slack:   @channel, @here, @everyone
    #   Discord: @here, @everyone, (role mentions @&role)
    #   Teams:   @channel, @team, @general
    #   Matrix:  @room
    #   Zulip:   @all, @everyone
    #   Forums:  @staff, @everyone
    r"@(channel|here|everyone|room|all|team|general|staff)",
    # Generic, distinctive announcement openers across platforms and email.
    # Keep conservative — don't match "Hey team" or "hi everyone" peer chat.
    r"^\s*(attention|psa|fyi|reminder|friendly\s+reminder|heads\s+up)\s*[:!,]",
    r"^\s*public\s+service\s+announcement\b",
    r"^\s*\[?\s*(announcement|update|important|urgent)\s*[\]:]",
)
_BOT_RE = re.compile("|".join(_BOT_PATTERNS), re.IGNORECASE)


def _is_likely_bot_or_system(text: str) -> bool:
    """True if the text smells like a bot or system message.

    Platform-agnostic defensive filter. For platform-aware logic (checking
    author.is_bot flags, message_type enums, role-based filtering), do it
    at the connector/crawler layer and pass cleaner records down.
    """
    if _BOT_RE.search(text):
        return True
    # All-caps announcements (>= 80% of alphabetic chars are uppercase).
    # Length gate ensures short acronyms ("API", "SDK") don't false-positive.
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 20:
        upper = sum(1 for c in letters if c.isupper())
        if upper / len(letters) >= 0.8:
            return True
    return False


def _iter_string_values(obj: object):
    """Depth-first walk of a payload dict yielding every string leaf value."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_string_values(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_string_values(item)


def _candidate_texts(
    records: list[RawRecord],
    is_bot_filter=None,
) -> list[str]:
    """Pull all usable text strings from record payloads, deduped.

    is_bot_filter: optional callable (text: str) -> bool. When provided,
    replaces the default heuristic filter — connectors can pass a
    platform-aware filter that uses e.g. author-is-bot flags or
    message-type enums they've already extracted at ingestion. When
    None, falls back to _is_likely_bot_or_system defensive heuristics.
    """
    filter_fn = is_bot_filter if is_bot_filter is not None else _is_likely_bot_or_system
    seen: set[str] = set()
    out: list[str] = []
    for r in records:
        for value in _iter_string_values(r.payload):
            text = value.strip()
            if not (VERBATIM_MIN_LEN <= len(text) <= VERBATIM_MAX_LEN):
                continue
            if filter_fn(text):
                continue
            if text in seen:
                continue
            seen.add(text)
            out.append(text)
    return out


def _style_signature(text: str) -> tuple[float, ...]:
    """Compute a cheap per-message style signature.

    Returns a fixed-length vector of normalized features. Tuple so it's
    hashable (for dedup) and usable with simple distance metrics.

    Features, in order:
      0. length in chars (normalized by VERBATIM_MAX_LEN)
      1. mean word length (normalized by 10)
      2. punctuation density (punct chars per total chars)
      3. contraction-or-apostrophe density (per word)
      4. fragment score: fraction of sentences missing a final period/?/!
      5. emoji/symbol density (non-ascii alpha chars per total chars)
      6. uppercase-start ratio (sentences starting with capital / total sentences)
      7. word count (normalized by 200)
    """
    n = max(1, len(text))
    tokens = text.split()
    wc = max(1, len(tokens))

    mean_word_len = sum(len(w) for w in tokens) / wc

    punct_chars = sum(1 for c in text if c in ".,;:!?-—–()[]{}\"'`")
    punct_density = punct_chars / n

    apostrophes = text.count("'") + text.count("\u2019")
    contraction_density = apostrophes / wc

    # Split on sentence-ending punct; count how many produced fragments
    # that don't end with terminal punctuation.
    fragments = re.split(r"(?<=[.!?])\s+", text.strip())
    unterminated = sum(1 for f in fragments if f and f[-1] not in ".!?")
    fragment_score = unterminated / max(1, len(fragments))

    non_ascii = sum(1 for c in text if ord(c) > 127)
    emoji_density = non_ascii / n

    starts_cap = sum(1 for f in fragments if f and f[0].isupper())
    cap_start_ratio = starts_cap / max(1, len(fragments))

    return (
        min(len(text) / VERBATIM_MAX_LEN, 1.0),
        min(mean_word_len / 10.0, 1.0),
        min(punct_density * 5.0, 1.0),
        min(contraction_density * 3.0, 1.0),
        fragment_score,
        min(emoji_density * 10.0, 1.0),
        cap_start_ratio,
        min(wc / 200.0, 1.0),
    )


def _style_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Euclidean distance between two style signatures."""
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _pick_style_coherent(
    candidates: list[str],
    k: int,
) -> list[str]:
    """Pick the k candidates whose pairwise style distances are tightest.

    Greedy: compute pairwise distances, score each candidate by mean distance
    to all others, keep the top-k with the lowest mean distance (i.e. closest
    to the style centroid). Simple and stable; no clustering needed.

    When candidates <= k, return all of them (already small).
    """
    if len(candidates) <= k:
        return candidates

    signatures = [_style_signature(c) for c in candidates]
    n = len(candidates)

    mean_dist: list[tuple[float, int]] = []
    for i in range(n):
        total = 0.0
        for j in range(n):
            if i == j:
                continue
            total += _style_distance(signatures[i], signatures[j])
        mean_dist.append((total / (n - 1), i))

    mean_dist.sort()  # ascending by mean distance (closest to centroid first)
    chosen_indices = sorted(idx for _, idx in mean_dist[:k])  # preserve original order
    return [candidates[i] for i in chosen_indices]


def _extract_verbatim_samples(
    records: list[RawRecord],
    k: int,
    is_bot_filter=None,
) -> list[str]:
    """Extract k style-coherent verbatim voice samples from a cluster's records.

    Empty list when cluster has no text-bearing records — callers handle as
    "no voice signal available."

    is_bot_filter: optional callable (text: str) -> bool for connectors
    that want to inject platform-specific filtering. Falls back to the
    default defensive heuristics when None.
    """
    candidates = _candidate_texts(records, is_bot_filter=is_bot_filter)
    if not candidates:
        return []
    return _pick_style_coherent(candidates, k)
