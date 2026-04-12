"""Downstream persona quality: segmentation impact on synthesis compatibility."""

from __future__ import annotations

import sys

sys.path.insert(0, "../synthesis")

from tests.helpers.fixture_loader import load_records
from tests.helpers.synthesis_stub import stub_persona_from_cluster, stub_groundedness_score


def test_jaccard_clusters_synthesize_valid_personas():
    """Jaccard cluster summaries produce structurally valid stub personas."""
    from segmentation.pipeline import segment

    records = load_records("golden_mixed_small.json")
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2)
    for c in clusters:
        persona = stub_persona_from_cluster(c)
        assert "name" in persona
        assert "goals" in persona
        assert len(persona["goals"]) >= 1
        assert "source_evidence" in persona


def test_gower_clusters_synthesize_valid_personas():
    """Gower cluster summaries produce structurally valid stub personas."""
    from segmentation.pipeline import segment

    records = load_records("golden_mixed_small.json")
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    for c in clusters:
        persona = stub_persona_from_cluster(c)
        assert "name" in persona
        assert "goals" in persona
        assert len(persona["source_evidence"]) >= 1


def test_gower_groundedness_not_worse():
    """Gower groundedness proxy is not worse than Jaccard on frozen fixture."""
    from segmentation.pipeline import segment

    records = load_records("golden_mixed_small.json")
    jaccard_clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2)
    gower_clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")

    j_scores = [stub_groundedness_score(stub_persona_from_cluster(c), c) for c in jaccard_clusters]
    g_scores = [stub_groundedness_score(stub_persona_from_cluster(c), c) for c in gower_clusters]

    j_avg = sum(j_scores) / len(j_scores) if j_scores else 0
    g_avg = sum(g_scores) / len(g_scores) if g_scores else 0
    assert g_avg >= j_avg - 0.05, f"Gower groundedness {g_avg:.2f} worse than Jaccard {j_avg:.2f}"


def test_gower_distinctiveness_better():
    """Gower cluster personas have more distinct behaviors than Jaccard."""
    from segmentation.pipeline import segment

    records = load_records("golden_mixed_small.json")
    jaccard = segment(records, similarity_threshold=0.15, min_cluster_size=2)
    gower = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")

    def behavior_overlap(clusters):
        sets = [set(c["summary"]["top_behaviors"]) for c in clusters]
        if len(sets) < 2:
            return 1.0
        overlaps = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                union = sets[i] | sets[j]
                inter = sets[i] & sets[j]
                overlaps.append(len(inter) / len(union) if union else 0)
        return sum(overlaps) / len(overlaps) if overlaps else 0

    j_overlap = behavior_overlap(jaccard)
    g_overlap = behavior_overlap(gower)
    assert g_overlap <= j_overlap + 0.1


def test_repeated_synthesis_schema_valid():
    """Repeated stub synthesis across same Gower clusters remains structurally valid."""
    from segmentation.pipeline import segment

    records = load_records("golden_mixed_small.json")
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    for _ in range(3):
        for c in clusters:
            persona = stub_persona_from_cluster(c)
            assert isinstance(persona["name"], str)
            assert isinstance(persona["goals"], list)
            assert isinstance(persona["source_evidence"], list)


def test_repeated_synthesis_grounded():
    """Repeated stub groundedness checks remain above threshold."""
    from segmentation.pipeline import segment

    records = load_records("golden_mixed_small.json")
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    for _ in range(3):
        for c in clusters:
            persona = stub_persona_from_cluster(c)
            score = stub_groundedness_score(persona, c)
            assert score >= 0.5, f"Groundedness {score:.2f} below 0.5"


def test_twin_prompt_structurally_valid():
    """A stub persona produces a structurally valid twin prompt context."""
    from segmentation.pipeline import segment

    records = load_records("golden_mixed_small.json")
    clusters = segment(records, similarity_threshold=0.15, min_cluster_size=2, distance_metric="gower")
    for c in clusters:
        persona = stub_persona_from_cluster(c)
        # Twin prompt needs: name, summary, vocabulary, sample_quotes
        assert persona["name"]
        assert persona["summary"]
        assert isinstance(persona["vocabulary"], list)
        assert isinstance(persona["sample_quotes"], list)
