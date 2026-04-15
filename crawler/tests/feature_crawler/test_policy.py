from __future__ import annotations

import unittest

from crawler.feature_crawler.policy import CollectionBasis, PolicyError, PolicyRegistry


class PolicyRegistryTest(unittest.TestCase):
    def test_reddit_collect_is_allowed_but_inference_is_not(self) -> None:
        registry = PolicyRegistry()
        row = registry.assert_allowed(platform="reddit")
        self.assertEqual(row.platform, "reddit")
        self.assertEqual(row.allow_infer, False)

    def test_web_requires_non_blocked_collection_basis(self) -> None:
        registry = PolicyRegistry()
        row = registry.assert_allowed(
            platform="web",
            collection_basis=CollectionBasis.PUBLIC_PERMITTED,
        )
        self.assertEqual(row.platform, "web")

    def test_linkedin_allows_inference_only_for_consented_or_owned_basis(self) -> None:
        registry = PolicyRegistry()
        consented = registry.assert_allowed(
            platform="linkedin",
            collection_basis=CollectionBasis.CONSENTED,
        )
        self.assertEqual(consented.platform, "linkedin")

        public_row = registry.assert_allowed(
            platform="linkedin",
            collection_basis=CollectionBasis.PUBLIC_PERMITTED,
        )
        self.assertFalse(public_row.allow_infer)

        with self.assertRaises(PolicyError):
            registry.assert_allowed(
                platform="linkedin",
                collection_basis=CollectionBasis.BLOCKED,
            )


if __name__ == "__main__":
    unittest.main()
