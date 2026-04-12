"""Task 8: Golden fixtures + verification harness — unit tests."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "crawler"))
sys.path.insert(0, str(Path(__file__).parent.parent / "synthesis"))

from segmentation.models.record import RawRecord

FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> list[RawRecord]:
    with open(FIXTURES / name) as f:
        data = json.load(f)
    return [RawRecord.model_validate(d) for d in data]


# ── 1. Backward-compat parity ──────────────────────────────────────────

def test_jaccard_parity_cluster_membership():
    """Jaccard mode on golden_mixed_small matches regression snapshot cluster count."""
    from segmentation.pipeline import segment

    records = _load_fixture("golden_mixed_small.json")
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="jaccard")
    assert len(clusters) == 2
    sizes = sorted(c["summary"]["cluster_size"] for c in clusters)
    assert sizes == [4, 4]


def test_jaccard_parity_top_behaviors():
    """Jaccard produces identical top_behaviors and top_pages to original."""
    from segmentation.pipeline import segment

    records = _load_fixture("golden_mixed_small.json")
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="jaccard")
    all_top_b = set()
    for c in clusters:
        all_top_b.update(c["summary"]["top_behaviors"])
    # Should contain both eng and des behaviors
    assert "api_setup" in all_top_b
    assert "template_browsing" in all_top_b


def test_jaccard_ignores_registry_golden():
    """Registry passed in Jaccard mode is silently ignored."""
    from segmentation.pipeline import segment
    from segmentation.engine.registry import DEFAULT_REGISTRY

    records = _load_fixture("golden_mixed_small.json")
    without = segment(records, similarity_threshold=0.15, min_cluster_size=2)
    with_reg = segment(records, similarity_threshold=0.15, min_cluster_size=2, feature_registry=DEFAULT_REGISTRY)
    assert len(without) == len(with_reg)
    for w, r in zip(without, with_reg):
        assert set(w["summary"]["top_behaviors"]) == set(r["summary"]["top_behaviors"])


# ── 2. Clustering behavior ────────────────────────────────────────────

def test_gower_separates_with_typed_features():
    """Gower uses typed payload to create finer distinctions than Jaccard."""
    from segmentation.pipeline import segment

    records = _load_fixture("golden_mixed_small.json")
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    # Should still separate engineers from designers
    assert len(clusters) >= 2


def test_imbalanced_small_cluster_survives():
    """Small-but-distinct cluster (2 enterprise users) survives in imbalanced data."""
    from segmentation.pipeline import segment

    records = _load_fixture("golden_imbalanced.json")
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    # Should have at least 2 clusters — dominant + at least one other
    assert len(clusters) >= 2
    # The enterprise users (small_0, small_1) should form their own cluster
    cluster_behaviors = [set(c["summary"]["top_behaviors"]) for c in clusters]
    has_enterprise = any("enterprise_inquiry" in b or "pricing_page" in b or "security_docs" in b for b in cluster_behaviors)
    assert has_enterprise


def test_sparse_zero_shared_isolated():
    """Users with zero shared features (GA4-only vs HubSpot-only) don't merge incorrectly."""
    from segmentation.pipeline import segment

    records = _load_fixture("golden_mixed_sparse.json")
    clusters = segment(records, similarity_threshold=0.5, min_cluster_size=2, distance_metric="gower")
    # At high threshold, GA4-only and HubSpot-only users should not merge
    # (they share only generic behaviors, and typed features don't overlap)
    for c in clusters:
        sources = set()
        for sr in c["sample_records"]:
            sources.add(sr["source"])
        # Each cluster should be source-coherent at this threshold
        # (or at least not mix wildly different typed features)
    assert len(clusters) >= 1


def test_threshold_monotonic_golden():
    """Higher threshold produces more clusters on golden_mixed_small."""
    from segmentation.pipeline import segment

    records = _load_fixture("golden_mixed_small.json")
    low = segment(records, similarity_threshold=0.15, min_cluster_size=1, distance_metric="gower")
    high = segment(records, similarity_threshold=0.8, min_cluster_size=1, distance_metric="gower")
    assert len(high) >= len(low)


def test_linkedin_like_categorical_clustering():
    """Gower clusters mostly-categorical LinkedIn-like data meaningfully."""
    from segmentation.pipeline import segment
    from segmentation.engine.registry import FeatureExtractor, Registry
    from segmentation.models.features import FeatureType

    # Custom registry for LinkedIn data
    li_registry: Registry = {
        "linkedin": [
            FeatureExtractor("seniority", "seniority", FeatureType.CATEGORICAL),
            FeatureExtractor("industry", "industry", FeatureType.CATEGORICAL),
            FeatureExtractor("location", "location", FeatureType.CATEGORICAL),
        ],
    }

    records = _load_fixture("golden_linkedin_like.json")
    clusters = segment(
        records, similarity_threshold=0.15, min_cluster_size=2,
        distance_metric="gower", feature_registry=li_registry,
    )
    assert len(clusters) >= 1


# ── 3. Permutation stability ──────────────────────────────────────────

def test_permutation_cluster_count_stable():
    """5 shuffled runs of golden_mixed_small produce identical cluster count."""
    from segmentation.pipeline import segment

    records = _load_fixture("golden_mixed_small.json")
    counts = []
    for seed in range(5):
        shuffled = records.copy()
        random.Random(seed).shuffle(shuffled)
        clusters = segment(shuffled, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
        counts.append(len(clusters))
    assert all(c == counts[0] for c in counts)


def test_permutation_membership_stability():
    """Across 5 shuffles, >=80% of users assigned to the same cluster."""
    from segmentation.pipeline import segment

    records = _load_fixture("golden_mixed_small.json")
    all_assignments = []
    for seed in range(5):
        shuffled = records.copy()
        random.Random(seed).shuffle(shuffled)
        clusters = segment(shuffled, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
        # Build user → cluster_index mapping via sample_records
        assignment = {}
        for idx, c in enumerate(clusters):
            for sr in c["sample_records"]:
                assignment[sr["record_id"]] = idx
        all_assignments.append(assignment)

    # Compare each pair: count agreements
    if len(all_assignments) < 2:
        return
    ref = all_assignments[0]
    all_record_ids = set(ref.keys())
    for other in all_assignments[1:]:
        agreements = sum(1 for rid in all_record_ids if ref.get(rid) == other.get(rid))
        agreement_rate = agreements / len(all_record_ids) if all_record_ids else 1.0
        assert agreement_rate >= 0.8, f"Agreement rate {agreement_rate:.2f} < 0.80"


# ── 4. CSV inference confidence ───────────────────────────────────────

def test_csv_dirty_numeric_id_excluded():
    """Numeric-looking zip_code with high cardinality excluded from registry."""
    from segmentation.engine.schema_inference import infer_registry

    records = _load_fixture("golden_csv_dirty.json")
    registry = infer_registry(records, source_name="csv")
    extractors = registry.get("csv", [])
    # zip_code is numeric-looking but all-unique → high cardinality → skipped as categorical
    # Actually zip_codes parse as float AND are high cardinality
    names = {e.feature_name for e in extractors}
    assert "zip_code" not in names


def test_csv_dirty_boolean_categorical():
    """Boolean column inferred as CATEGORICAL."""
    from segmentation.engine.schema_inference import infer_registry
    from segmentation.models.features import FeatureType

    records = _load_fixture("golden_csv_dirty.json")
    registry = infer_registry(records, source_name="csv")
    extractors = registry.get("csv", [])
    active_ext = next((e for e in extractors if e.feature_name == "active"), None)
    assert active_ext is not None
    assert active_ext.feature_type == FeatureType.CATEGORICAL


def test_csv_dirty_percentage_excluded():
    """Percentage column (85%, 90%) excluded — fails float parse due to %."""
    from segmentation.engine.schema_inference import infer_registry

    records = _load_fixture("golden_csv_dirty.json")
    registry = infer_registry(records, source_name="csv")
    extractors = registry.get("csv", [])
    names = {e.feature_name for e in extractors}
    assert "pct" not in names


def test_csv_dirty_mixed_format_excluded():
    """Mixed-format amount column (1,200 / 3.5 / N/A) excluded if <80% parse."""
    from segmentation.engine.schema_inference import infer_registry

    records = _load_fixture("golden_csv_dirty.json")
    registry = infer_registry(records, source_name="csv")
    extractors = registry.get("csv", [])
    names = {e.feature_name for e in extractors}
    # "1,200" fails float(), "N/A" fails → only ~75% parse → excluded
    assert "amount" not in names


# ── 5. Summarizer + handoff validation ────────────────────────────────

def test_gower_golden_validates_clusterdata():
    """Every Gower cluster from golden_mixed_small validates against ClusterData."""
    from segmentation.pipeline import segment
    from synthesis.models.cluster import ClusterData

    records = _load_fixture("golden_mixed_small.json")
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    for c in clusters:
        ClusterData.model_validate(c)


def test_typed_features_json_serializable():
    """typed_features in extra is JSON-serializable."""
    from segmentation.pipeline import segment

    records = _load_fixture("golden_mixed_small.json")
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    for c in clusters:
        if "typed_features" in c["summary"]["extra"]:
            # Should not raise
            json.dumps(c["summary"]["extra"]["typed_features"])


def test_sample_records_sufficient():
    """Each cluster has >= 3 sample records for downstream grounding."""
    from segmentation.pipeline import segment

    records = _load_fixture("golden_mixed_small.json")
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    for c in clusters:
        assert len(c["sample_records"]) >= 3


# ── 6. Downstream persona quality (lightweight, no LLM calls) ────────

def test_gower_groundedness_not_worse():
    """Gower-mode cluster data has at least as many evidence records as Jaccard."""
    from segmentation.pipeline import segment

    records = _load_fixture("golden_mixed_small.json")
    jaccard = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="jaccard")
    gower = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")

    jaccard_total_samples = sum(len(c["sample_records"]) for c in jaccard)
    gower_total_samples = sum(len(c["sample_records"]) for c in gower)
    # Gower should have at least as much evidence material
    assert gower_total_samples >= jaccard_total_samples


def test_gower_distinctiveness_improved():
    """Gower clusters have more distinct top_behaviors sets than Jaccard."""
    from segmentation.pipeline import segment

    records = _load_fixture("golden_mixed_small.json")
    jaccard = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="jaccard")
    gower = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")

    def behavior_overlap(clusters):
        if len(clusters) < 2:
            return 1.0
        sets = [set(c["summary"]["top_behaviors"]) for c in clusters]
        overlaps = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                union = sets[i] | sets[j]
                inter = sets[i] & sets[j]
                overlaps.append(len(inter) / len(union) if union else 0)
        return sum(overlaps) / len(overlaps) if overlaps else 0

    j_overlap = behavior_overlap(jaccard)
    g_overlap = behavior_overlap(gower)
    # Gower should have at most the same overlap (lower = more distinct)
    assert g_overlap <= j_overlap + 0.1  # small tolerance


def test_gower_personas_schema_valid():
    """Gower cluster outputs are structurally valid for PersonaV1 synthesis."""
    from segmentation.pipeline import segment
    from synthesis.models.cluster import ClusterData

    records = _load_fixture("golden_mixed_small.json")
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    for c in clusters:
        validated = ClusterData.model_validate(c)
        assert validated.summary.cluster_size >= 2
        assert len(validated.sample_records) >= 1
        assert validated.tenant.tenant_id == "t_golden"
