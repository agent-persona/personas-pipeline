"""Tests for Experiment 1.19: Schema artifact format.

Verifies:
1. build_system_prompt produces correct output for each format
2. Default behavior is preserved (schema_format=None)
3. All three renderers produce valid, non-empty output
4. Tool definition is unchanged across formats
5. Jaccard similarity helper works correctly
6. analyze_persona computes correct stats
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from synthesis.engine.prompt_builder import (
    SYSTEM_PROMPT,
    _render_jsonschema_schema,
    _render_pydantic_schema,
    _render_typescript_schema,
    build_system_prompt,
    build_tool_definition,
)

from experiment_1_19 import (
    analyze_persona,
    jaccard_similarity,
)


# ── System prompt tests ───────────────────────────────────────────────

class TestBuildSystemPrompt:
    def test_control_no_schema(self):
        """None format returns base prompt unchanged."""
        prompt = build_system_prompt(schema_format=None)
        assert prompt == SYSTEM_PROMPT
        assert "Expected output schema" not in prompt

    def test_pydantic_format(self):
        """Pydantic format includes Python class definitions."""
        prompt = build_system_prompt(schema_format="pydantic")
        assert "Expected output schema (Pydantic model (Python))" in prompt
        assert "class PersonaV1(BaseModel):" in prompt
        assert "list[str]" in prompt
        assert "MUST conform" in prompt

    def test_jsonschema_format(self):
        """JSON Schema format includes valid JSON."""
        prompt = build_system_prompt(schema_format="jsonschema")
        assert "Expected output schema (JSON Schema)" in prompt
        assert '"properties"' in prompt
        assert '"type"' in prompt
        assert "MUST conform" in prompt
        # The JSON Schema portion should be parseable
        start = prompt.index("```\n") + 4
        end = prompt.index("\n```", start)
        schema_text = prompt[start:end]
        parsed = json.loads(schema_text)
        assert "properties" in parsed

    def test_typescript_format(self):
        """TypeScript format includes TS interfaces."""
        prompt = build_system_prompt(schema_format="typescript")
        assert "Expected output schema (TypeScript interface)" in prompt
        assert "interface PersonaV1" in prompt
        assert "string[]" in prompt
        assert "MUST conform" in prompt

    def test_all_formats_include_base_prompt(self):
        """All formats still contain the base system prompt."""
        for fmt in ["pydantic", "jsonschema", "typescript"]:
            prompt = build_system_prompt(schema_format=fmt)
            assert "persona synthesis expert" in prompt
            assert "Grounded" in prompt
            assert "source_evidence" in prompt


# ── Renderer tests ────────────────────────────────────────────────────

class TestRenderers:
    def test_pydantic_renderer(self):
        output = _render_pydantic_schema()
        assert "class PersonaV1(BaseModel):" in output
        assert "goals: list[str]" in output
        assert "Demographics" in output
        assert len(output) > 200

    def test_jsonschema_renderer(self):
        output = _render_jsonschema_schema()
        parsed = json.loads(output)
        assert "properties" in parsed
        assert "name" in parsed["properties"]
        assert "goals" in parsed["properties"]

    def test_typescript_renderer(self):
        output = _render_typescript_schema()
        assert "interface PersonaV1" in output
        assert "interface Demographics" in output
        assert "goals: string[]" in output
        assert len(output) > 200

    def test_renderers_cover_same_fields(self):
        """All renderers mention the key persona fields."""
        key_fields = ["name", "summary", "goals", "pains", "motivations"]
        for renderer in [_render_pydantic_schema, _render_jsonschema_schema, _render_typescript_schema]:
            output = renderer()
            for field in key_fields:
                assert field in output, f"{renderer.__name__} missing field: {field}"


# ── Tool definition unchanged ─────────────────────────────────────────

class TestToolDefinitionUnchanged:
    def test_tool_schema_same_regardless_of_format(self):
        """The tool definition (input_schema) must be identical across all variants."""
        base = build_tool_definition()
        # Tool definition doesn't take schema_format — it's always the same
        assert base["name"] == "create_persona"
        assert "properties" in base["input_schema"]


# ── Metric helpers ────────────────────────────────────────────────────

class TestJaccardSimilarity:
    def test_identical_sets(self):
        assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        assert jaccard_similarity({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self):
        sim = jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        assert abs(sim - 0.5) < 0.01  # 2/4

    def test_empty_sets(self):
        assert jaccard_similarity(set(), set()) == 1.0

    def test_one_empty(self):
        assert jaccard_similarity({"a"}, set()) == 0.0


class TestAnalyzePersona:
    def test_basic_analysis(self):
        persona = {
            "summary": "A short summary of this persona.",
            "goals": ["reduce cost", "improve speed", "scale globally"],
            "pains": ["slow deploys"],
            "motivations": ["career growth"],
            "objections": ["too expensive"],
            "channels": ["twitter", "linkedin"],
            "vocabulary": ["ship", "deploy", "iterate"],
            "decision_triggers": ["demo"],
            "sample_quotes": ["Let's ship it"],
        }
        analysis = analyze_persona(persona)
        assert analysis["item_counts"]["goals"] == 3
        assert analysis["item_counts"]["pains"] == 1
        assert analysis["total_items"] == 13
        assert analysis["summary_length"] > 0
        assert analysis["mean_tokens_per_item"] > 0

    def test_empty_fields(self):
        persona = {f: [] for f in [
            "goals", "pains", "motivations", "objections",
            "channels", "vocabulary", "decision_triggers", "sample_quotes",
        ]}
        persona["summary"] = ""
        analysis = analyze_persona(persona)
        assert analysis["total_items"] == 0
        assert analysis["mean_tokens_per_item"] == 0


# ── Run with pytest ───────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
