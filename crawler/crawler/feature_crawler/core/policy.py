from __future__ import annotations

from dataclasses import dataclass
import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport of StrEnum for Python < 3.11."""


from .models import CrawlTarget


class PolicyError(ValueError):
    """Raised when a crawl target violates source policy."""


class CollectionBasis(StrEnum):
    OWNED = "owned"
    CONSENTED = "consented"
    PUBLIC_PERMITTED = "public-permitted"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class SourcePolicy:
    platform: str
    collection_basis: CollectionBasis
    commercial_status: str
    action_required: str
    allow_collect: bool
    allow_infer: bool
    allow_cross_link: bool

    @property
    def notes(self) -> str:
        return f"{self.action_required} ({self.commercial_status})"


PLATFORM_POLICIES: dict[str, SourcePolicy] = {
    "reddit": SourcePolicy(
        platform="reddit",
        collection_basis=CollectionBasis.PUBLIC_PERMITTED,
        commercial_status="non-commercial default; contract required for commercial use",
        action_required="OAuth app + Reddit Builder Policy review",
        allow_collect=True,
        allow_infer=False,
        allow_cross_link=False,
    ),
    "x": SourcePolicy(
        platform="x",
        collection_basis=CollectionBasis.BLOCKED,
        commercial_status="pay-per-use",
        action_required="budget approval + policy review",
        allow_collect=False,
        allow_infer=False,
        allow_cross_link=False,
    ),
    "discord": SourcePolicy(
        platform="discord",
        collection_basis=CollectionBasis.CONSENTED,
        commercial_status="free bot API",
        action_required="bot invited by admin",
        allow_collect=True,
        allow_infer=True,
        allow_cross_link=False,
    ),
    "twitch": SourcePolicy(
        platform="twitch",
        collection_basis=CollectionBasis.PUBLIC_PERMITTED,
        commercial_status="free API",
        action_required="application registration",
        allow_collect=True,
        allow_infer=True,
        allow_cross_link=False,
    ),
    "facebook": SourcePolicy(
        platform="facebook",
        collection_basis=CollectionBasis.BLOCKED,
        commercial_status="app review required",
        action_required="policy + app review",
        allow_collect=False,
        allow_infer=False,
        allow_cross_link=False,
    ),
    "instagram": SourcePolicy(
        platform="instagram",
        collection_basis=CollectionBasis.BLOCKED,
        commercial_status="app review required",
        action_required="policy + app review",
        allow_collect=False,
        allow_infer=False,
        allow_cross_link=False,
    ),
    "web": SourcePolicy(
        platform="web",
        collection_basis=CollectionBasis.PUBLIC_PERMITTED,
        commercial_status="depends on site terms",
        action_required="allowlist + terms review",
        allow_collect=True,
        allow_infer=False,
        allow_cross_link=False,
    ),
    "linkedin": SourcePolicy(
        platform="linkedin",
        collection_basis=CollectionBasis.CONSENTED,
        commercial_status="self-serve sign-in; partner approval or legal review needed for deeper member data",
        action_required="consent path or approved scraper/vendor review",
        allow_collect=True,
        allow_infer=False,
        allow_cross_link=False,
    ),
}


def _coerce_basis(value: CollectionBasis | str | None) -> CollectionBasis | None:
    if value is None or isinstance(value, CollectionBasis):
        return value
    return CollectionBasis(value)


class PolicyRegistry:
    def row_for(self, platform: str) -> SourcePolicy:
        try:
            return PLATFORM_POLICIES[platform]
        except KeyError as exc:
            raise PolicyError(f"unknown platform: {platform}") from exc

    def assert_allowed(
        self,
        *,
        platform: str,
        collection_basis: CollectionBasis | str | None = None,
        use_fallback: bool = False,
    ) -> SourcePolicy:
        row = self.row_for(platform)
        effective_basis = _coerce_basis(collection_basis) or row.collection_basis
        allow_collect = effective_basis != CollectionBasis.BLOCKED and row.allow_collect
        if not allow_collect:
            raise PolicyError(f"{platform} is policy-blocked: {row.notes}")
        if use_fallback and effective_basis == CollectionBasis.BLOCKED:
            raise PolicyError(f"fallback crawling not allowed for collection basis {effective_basis}")
        return SourcePolicy(
            platform=row.platform,
            collection_basis=effective_basis,
            commercial_status=row.commercial_status,
            action_required=row.action_required,
            allow_collect=True,
            allow_infer=row.allow_infer,
            allow_cross_link=row.allow_cross_link,
        )


def resolve_policy(target: CrawlTarget) -> SourcePolicy:
    if target.platform == "web":
        basis = _coerce_basis(target.collection_basis) or CollectionBasis.BLOCKED
        allow_collect = basis != CollectionBasis.BLOCKED
        allow_infer = allow_collect and target.allow_persona_inference
        return SourcePolicy(
            platform="web",
            collection_basis=basis,
            commercial_status="depends on site terms",
            action_required="allowlist + terms review",
            allow_collect=allow_collect,
            allow_infer=allow_infer,
            allow_cross_link=False,
        )
    if target.platform == "linkedin":
        basis = _coerce_basis(target.collection_basis) or CollectionBasis.CONSENTED
        allow_collect = basis != CollectionBasis.BLOCKED
        allow_infer = (
            allow_collect
            and target.allow_persona_inference
            and basis in {CollectionBasis.OWNED, CollectionBasis.CONSENTED}
        )
        return SourcePolicy(
            platform="linkedin",
            collection_basis=basis,
            commercial_status="self-serve sign-in; partner approval or legal review needed for deeper member data",
            action_required="consent path or approved scraper/vendor review",
            allow_collect=allow_collect,
            allow_infer=allow_infer,
            allow_cross_link=False,
        )
    row = PLATFORM_POLICIES[target.platform]
    basis = _coerce_basis(target.collection_basis) or row.collection_basis
    return SourcePolicy(
        platform=row.platform,
        collection_basis=basis,
        commercial_status=row.commercial_status,
        action_required=row.action_required,
        allow_collect=basis != CollectionBasis.BLOCKED and row.allow_collect,
        allow_infer=row.allow_infer,
        allow_cross_link=row.allow_cross_link,
    )


def assert_target_allowed(target: CrawlTarget) -> SourcePolicy:
    policy = resolve_policy(target)
    if not policy.allow_collect:
        raise PolicyError(
            f"{target.platform} target '{target.target_id}' is blocked: {policy.notes}"
        )
    if target.allow_persona_inference and not policy.allow_infer:
        raise PolicyError(
            f"{target.platform} target '{target.target_id}' is not cleared for persona inference."
        )
    if target.allow_cross_linking and not policy.allow_cross_link:
        raise PolicyError(
            f"{target.platform} target '{target.target_id}' is not cleared for cross-linking."
        )
    return policy
