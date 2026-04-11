"""Tests for Experiment 1.01: Schema width.

Verifies:
1. PersonaMinimal and PersonaMaximal schemas produce valid JSON schemas
2. Tool definition changes based on schema_width
3. Schema width map is complete
4. Field count metric works
5. Twin drift metric works
6. Default behavior is preserved (no schema_width = PersonaV1)
7. Groundedness handles missing fields gracefully
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from synthesis.models.persona import (
    PersonaMinimal,
    PersonaMaximal,
    PersonaV1,
    SchemaWidth,
    SCHEMA_WIDTH_MAP,
)
from synthesis.engine.prompt_builder import build_tool_definition

from experiment_1_01 import (
    compute_twin_drift,
    count_fields,
)


# ── Schema variant tests ──────────────────────────────────────────────

class TestSchemaVariants:
    def test_minimal_has_few_fields(self):
        schema = PersonaMinimal.model_json_schema()
        props = schema["properties"]
        # Should have: schema_version, name, summary, goals, pains, source_evidence
        assert len(props) <= 7

    def test_minimal_required_fields(self):
        schema = PersonaMinimal.model_json_schema()
        props = schema["properties"]
        assert "name" in props
        assert "summary" in props
        assert "goals" in props
        assert "pains" in props
        assert "source_evidence" in props

    def test_minimal_no_demographics(self):
        schema = PersonaMinimal.model_json_schema()
        props = schema["properties"]
        assert "demographics" not in props
        assert "firmographics" not in props
        assert "vocabulary" not in props

    def test_maximal_has_many_fields(self):
        schema = PersonaMaximal.model_json_schema()
        props = schema["properties"]
        assert len(props) >= 20

    def test_maximal_has_extra_fields(self):
        schema = PersonaMaximal.model_json_schema()
        props = schema["properties"]
        assert "backstory" in props
        assert "daily_routine" in props
        assert "communication_style" in props
        assert "brand_affinities" in props
        assert "frustration_triggers" in props
        assert "success_metrics" in props
        assert "career_trajectory" in props
        assert "pet_peeves" in props

    def test_maximal_inherits_v1_fields(self):
        schema = PersonaMaximal.model_json_schema()
        props = schema["properties"]
        # Should still have all V1 fields
        assert "name" in props
        assert "goals" in props
        assert "pains" in props
        assert "vocabulary" in props
        assert "sample_quotes" in props

    def test_current_is_persona_v1(self):
        assert SCHEMA_WIDTH_MAP["current"] is PersonaV1

    def test_all_widths_valid_json_schema(self):
        for width, cls in SCHEMA_WIDTH_MAP.items():
            schema = cls.model_json_schema()
            assert "properties" in schema, f"{width} missing properties"
            assert "name" in schema["properties"], f"{width} missing name"

    def test_schema_width_map_complete(self):
        assert "minimal" in SCHEMA_WIDTH_MAP
        assert "current" in SCHEMA_WIDTH_MAP
        assert "maximal" in SCHEMA_WIDTH_MAP


# ── Tool definition tests ─────────────────────────────────────────────

class TestToolDefinition:
    def test_default_is_current(self):
        tool = build_tool_definition()
        assert tool["name"] == "create_persona"
        assert "minimal" not in tool["description"]
        assert "expanded" not in tool["description"]

    def test_minimal_tool(self):
        tool = build_tool_definition(schema_width="minimal")
        assert "minimal" in tool["description"]
        schema = tool["input_schema"]
        assert "name" in schema["properties"]
        # Should NOT have demographics
        assert "demographics" not in schema["properties"]

    def test_maximal_tool(self):
        tool = build_tool_definition(schema_width="maximal")
        assert "expanded" in tool["description"]
        schema = tool["input_schema"]
        assert "backstory" in schema["properties"]

    def test_current_tool_explicit(self):
        tool = build_tool_definition(schema_width="current")
        assert tool["name"] == "create_persona"


# ── Field count tests ────────────────────────────────────────────────

class TestFieldCount:
    def test_minimal_persona(self):
        persona = {
            "schema_version": "1.0",
            "name": "Test",
            "summary": "A test",
            "goals": ["g1", "g2"],
            "pains": ["p1", "p2"],
            "source_evidence": [{"claim": "c", "record_ids": ["r1"], "field_path": "goals.0", "confidence": 1.0}],
        }
        assert count_fields(persona) == 5  # name, summary, goals, pains, source_evidence

    def test_excludes_meta(self):
        persona = {"name": "Test", "_meta": {"cost": 1.0}, "schema_version": "1.0"}
        assert count_fields(persona) == 1  # only name

    def test_excludes_empty(self):
        persona = {"name": "Test", "goals": [], "summary": ""}
        assert count_fields(persona) == 1  # only name (goals=[], summary="" excluded)


# ── Twin drift tests ─────────────────────────────────────────────────

class TestTwinDrift:
    def test_identical_responses(self):
        drift = compute_twin_drift(["hello world", "hello world", "hello world"])
        assert drift == 1.0

    def test_completely_different(self):
        drift = compute_twin_drift(["alpha beta", "gamma delta", "epsilon zeta"])
        assert drift == 0.0

    def test_partial_overlap(self):
        drift = compute_twin_drift(["I like python", "I use python daily"])
        assert 0.0 < drift < 1.0

    def test_single_response(self):
        assert compute_twin_drift(["only one"]) == 1.0

    def test_empty(self):
        assert compute_twin_drift([]) == 1.0


# ── Groundedness with minimal schema ─────────────────────────────────

class TestGroundednessMinimal:
    def test_minimal_persona_no_crash(self):
        """Groundedness check should handle PersonaMinimal (no motivations/objections)."""
        from synthesis.engine.groundedness import check_groundedness
        from synthesis.models.cluster import ClusterData

        # Build a minimal persona
        persona = PersonaMinimal(
            name="Test",
            summary="A test persona",
            goals=["goal1", "goal2"],
            pains=["pain1", "pain2"],
            source_evidence=[
                {"claim": "c1", "record_ids": ["rec_001"], "field_path": "goals.0", "confidence": 1.0},
                {"claim": "c2", "record_ids": ["rec_001"], "field_path": "goals.1", "confidence": 1.0},
                {"claim": "c3", "record_ids": ["rec_001"], "field_path": "pains.0", "confidence": 1.0},
                {"claim": "c4", "record_ids": ["rec_001"], "field_path": "pains.1", "confidence": 0.9},
            ],
        )

        # Build a minimal cluster with matching record IDs
        cluster = ClusterData.model_validate({
            "cluster_id": "test",
            "tenant": {
                "tenant_id": "t",
                "industry": "tech",
                "product_description": "product",
            },
            "summary": {
                "cluster_size": 1,
                "top_behaviors": [],
                "top_pages": [],
            },
            "sample_records": [
                {"record_id": "rec_001", "source": "test", "payload": {}},
            ],
            "enrichment": {},
        })

        report = check_groundedness(persona, cluster)
        # Should not crash and should pass (goals and pains are covered)
        assert report.score >= 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
