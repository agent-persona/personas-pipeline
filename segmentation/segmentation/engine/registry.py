"""Feature schema registry: maps sources to typed feature extractors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from segmentation.models.features import FeatureType

Registry = dict[str, list["FeatureExtractor"]]


@dataclass(frozen=True)
class FeatureExtractor:
    payload_key: str
    feature_name: str
    feature_type: FeatureType
    normalize: Callable[[str], str] | None = None


def get_extractors(registry: Registry, source: str) -> list[FeatureExtractor]:
    return registry.get(source, [])


DEFAULT_REGISTRY: Registry = {
    "ga4": [
        FeatureExtractor("session_duration", "session_duration", FeatureType.NUMERIC),
    ],
    "hubspot": [
        FeatureExtractor("company_size", "company_size", FeatureType.CATEGORICAL),
        FeatureExtractor("industry", "industry", FeatureType.CATEGORICAL),
        FeatureExtractor("contact_title", "role", FeatureType.CATEGORICAL, normalize=str.strip),
    ],
    "intercom": [
        FeatureExtractor("topic", "topic", FeatureType.CATEGORICAL),
    ],
}
