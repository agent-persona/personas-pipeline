"""Experiment 3.03: Retrieval-augmented synthesis.

Lightweight record retrieval via TF-IDF cosine similarity. Embeds cluster
records and retrieves top-k per persona section, so the synthesis prompt
includes only the most relevant evidence per field.

No external embedding provider required — uses bag-of-words TF-IDF
computed locally. This tests the *pattern* (retrieval vs. dump-all),
not the embedding model quality.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from synthesis.models.cluster import SampleRecord

# ── Persona sections and their retrieval queries ──────────────────────
# Each section gets a natural-language query used to find the most
# relevant records.

SECTION_QUERIES: dict[str, str] = {
    "goals": (
        "user goals aspirations objectives wants to achieve improve "
        "reduce increase optimize automate streamline"
    ),
    "pains": (
        "pain points frustrations problems issues complaints struggles "
        "difficulties challenges obstacles blockers slow broken"
    ),
    "motivations": (
        "motivation drives reasons values career growth success "
        "recognition efficiency passion purpose impact"
    ),
    "objections": (
        "objections concerns hesitations resistance pushback cost "
        "price expensive risk complexity migration learning curve"
    ),
    "channels": (
        "channels platforms tools communication social media email "
        "slack twitter linkedin community forum conference"
    ),
    "vocabulary": (
        "terminology jargon language words phrases technical terms "
        "domain specific slang abbreviations"
    ),
    "decision_triggers": (
        "decision trigger buying signal evaluation demo trial "
        "comparison recommendation referral deadline budget approval"
    ),
    "demographics": (
        "age location education role title seniority experience "
        "background team size company"
    ),
    "firmographics": (
        "company size industry revenue funding stage tech stack "
        "infrastructure tools platform enterprise startup"
    ),
}


# ── TF-IDF engine (no external deps) ─────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into words."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_idf(docs: list[list[str]]) -> dict[str, float]:
    """Compute inverse document frequency across a corpus."""
    n = len(docs)
    df: Counter = Counter()
    for tokens in docs:
        for word in set(tokens):
            df[word] += 1
    return {word: math.log((n + 1) / (count + 1)) + 1 for word, count in df.items()}


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Compute TF-IDF vector for a token list."""
    tf = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {word: (count / total) * idf.get(word, 1.0) for word, count in tf.items()}


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[w] * vec_b[w] for w in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Record embedding and retrieval ────────────────────────────────────

@dataclass
class EmbeddedRecord:
    """A record with its TF-IDF vector precomputed."""
    record: SampleRecord
    tokens: list[str]
    vector: dict[str, float]


class RecordIndex:
    """Indexes cluster records for top-k retrieval per section."""

    def __init__(self, records: list[SampleRecord]) -> None:
        self.records = records

        # Tokenize each record's full payload text
        docs: list[list[str]] = []
        for r in records:
            text = _record_to_text(r)
            docs.append(_tokenize(text))

        # Build IDF from the record corpus + section queries
        all_docs = docs + [_tokenize(q) for q in SECTION_QUERIES.values()]
        self.idf = _build_idf(all_docs)

        # Embed records
        self.embedded: list[EmbeddedRecord] = []
        for r, tokens in zip(records, docs):
            vec = _tfidf_vector(tokens, self.idf)
            self.embedded.append(EmbeddedRecord(record=r, tokens=tokens, vector=vec))

    def retrieve(self, section: str, k: int | None = None) -> list[SampleRecord]:
        """Retrieve top-k records most relevant to a persona section.

        Args:
            section: One of SECTION_QUERIES keys (e.g., "goals", "pains").
            k: Number of records to return. None = return all.

        Returns:
            Records sorted by relevance (most relevant first).
        """
        if k is None or k >= len(self.embedded):
            return [e.record for e in self.embedded]

        query = SECTION_QUERIES.get(section, section)
        query_tokens = _tokenize(query)
        query_vec = _tfidf_vector(query_tokens, self.idf)

        scored = [
            (_cosine_similarity(query_vec, e.vector), e.record)
            for e in self.embedded
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in scored[:k]]

    def retrieve_global(self, k: int | None = None) -> list[SampleRecord]:
        """Retrieve top-k records using a broad relevance query.

        For use when not doing per-section retrieval.
        """
        if k is None or k >= len(self.embedded):
            return [e.record for e in self.embedded]

        # Use a combined query of all sections
        combined = " ".join(SECTION_QUERIES.values())
        query_tokens = _tokenize(combined)
        query_vec = _tfidf_vector(query_tokens, self.idf)

        scored = [
            (_cosine_similarity(query_vec, e.vector), e.record)
            for e in self.embedded
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in scored[:k]]


def _record_to_text(record: SampleRecord) -> str:
    """Flatten a record into a searchable text string."""
    parts = [record.source, record.record_id]
    if record.timestamp:
        parts.append(record.timestamp)
    for key, value in record.payload.items():
        parts.append(f"{key} {value}")
    return " ".join(str(p) for p in parts)
