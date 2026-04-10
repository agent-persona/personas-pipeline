"""Tests for Experiment 1.17: Length budgets per field.

Verifies:
1. prompt_builder produces correct schema modifications at each multiplier
2. Default behavior is preserved (budget_multiplier=None)
3. Hedge detection regex works correctly
4. Info density calculation is correct
5. Field length stats are computed properly
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from synthesis.engine.prompt_builder import (
    FIELD_BASE_BUDGETS,
    _apply_length_budgets,
    build_messages,
    build_retry_messages,
    build_tool_definition,
)
from synthesis.models.persona import PersonaV1

# Import metrics from experiment script
from experiment_1_17 import (
    HEDGE_PATTERNS,
    compute_field_length_stats,
    compute_hedge_rate,
    compute_info_density,
    estimate_tokens,
)


# ── Tool definition tests ─────────────────────────────────────────────

class TestToolDefinition:
    def test_default_has_no_budget_hints(self):
        """Control: budget_multiplier=None produces unmodified schema."""
        tool = build_tool_definition(budget_multiplier=None)
        schema_str = str(tool["input_schema"])
        assert "TARGET:" not in schema_str
        assert "token" not in schema_str.lower() or "tokens" not in schema_str.lower()
        assert tool["name"] == "create_persona"
        assert "IMPORTANT: Respect" not in tool["description"]

    def test_budget_multiplier_injects_hints(self):
        """Each multiplier injects TARGET hints into schema descriptions."""
        for mult in [0.4, 1.0, 4.0]:
            tool = build_tool_definition(budget_multiplier=mult)
            schema = tool["input_schema"]
            # At least some fields should have TARGET hints
            found = False
            for field_name in FIELD_BASE_BUDGETS:
                prop = schema.get("properties", {}).get(field_name, {})
                desc = prop.get("description", "")
                items_desc = prop.get("items", {}).get("description", "")
                if "TARGET:" in desc or "TARGET:" in items_desc:
                    found = True
                    break
            assert found, f"No TARGET hint found for multiplier={mult}"
            assert "IMPORTANT: Respect" in tool["description"]

    def test_tight_budget_values(self):
        """0.4x multiplier produces ~20-token targets."""
        tool = build_tool_definition(budget_multiplier=0.4)
        schema = tool["input_schema"]
        # summary base=50, 50*0.4=20
        props = schema["properties"]
        for field_name, base in FIELD_BASE_BUDGETS.items():
            expected = int(base * 0.4)
            prop = props.get(field_name, {})
            desc = prop.get("description", "") + prop.get("items", {}).get("description", "")
            assert str(expected) in desc, (
                f"Field {field_name}: expected ~{expected} in description, got: {desc}"
            )

    def test_relaxed_budget_values(self):
        """4.0x multiplier produces ~200-token targets."""
        tool = build_tool_definition(budget_multiplier=4.0)
        schema = tool["input_schema"]
        props = schema["properties"]
        # summary base=50, 50*4=200
        summary_prop = props.get("summary", {})
        desc = summary_prop.get("description", "")
        assert "200" in desc, f"Expected 200 in summary description, got: {desc}"

    def test_schema_still_valid_with_budgets(self):
        """Schema with budget hints should still be valid JSON Schema structure."""
        for mult in [0.4, 1.0, 4.0]:
            tool = build_tool_definition(budget_multiplier=mult)
            schema = tool["input_schema"]
            assert "properties" in schema
            assert "required" in schema or "type" in schema
            assert tool["name"] == "create_persona"


# ── Message building tests ────────────────────────────────────────────

class TestMessageBuilding:
    def _make_stub_cluster(self):
        from synthesis.models.cluster import (
            ClusterData,
            ClusterSummary,
            EnrichmentPayload,
            SampleRecord,
            TenantContext,
        )
        return ClusterData(
            cluster_id="test-cluster",
            tenant=TenantContext(tenant_id="test-tenant"),
            summary=ClusterSummary(cluster_size=10, top_behaviors=["click"]),
            sample_records=[
                SampleRecord(record_id="rec_001", source="ga4", payload={"action": "click"}),
            ],
            enrichment=EnrichmentPayload(),
        )

    def test_messages_no_budget(self):
        """Default messages contain no budget hints."""
        cluster = self._make_stub_cluster()
        msgs = build_messages(cluster, budget_multiplier=None)
        assert len(msgs) == 1
        assert "token budgets" not in msgs[0]["content"].lower()

    def test_messages_with_budget(self):
        """Budget messages contain budget instruction."""
        cluster = self._make_stub_cluster()
        msgs = build_messages(cluster, budget_multiplier=1.0)
        assert "token budgets" in msgs[0]["content"].lower()
        assert "multiplier=1.0" in msgs[0]["content"]

    def test_retry_messages_with_budget(self):
        """Retry messages include budget reminder."""
        cluster = self._make_stub_cluster()
        msgs = build_retry_messages(cluster, ["error1"], budget_multiplier=0.4)
        content = msgs[0]["content"]
        assert "error1" in content
        assert "multiplier=0.4" in content


# ── Metric computation tests ─────────────────────────────────────────

class TestHedgeDetection:
    def test_detects_hedges(self):
        """Known hedge words are detected."""
        hedgy = "This potentially could maybe help the user"
        matches = HEDGE_PATTERNS.findall(hedgy)
        assert len(matches) >= 2  # "potentially", "maybe" at minimum

    def test_clean_text(self):
        """Clean, direct text has no hedge matches."""
        clean = "Reduces deployment time by 40%"
        matches = HEDGE_PATTERNS.findall(clean)
        assert len(matches) == 0

    def test_hedge_rate_computation(self):
        """Hedge rate correctly computes fraction of hedged items."""
        persona = {
            "goals": ["Reduce cost by 50%", "Potentially improve speed"],
            "pains": ["Slow builds", "Perhaps unreliable deploys"],
            "motivations": ["Ship faster"],
            "objections": ["Too expensive"],
            "channels": [],
            "vocabulary": [],
            "decision_triggers": [],
            "sample_quotes": [],
        }
        rate, count, total = compute_hedge_rate(persona)
        assert total == 6
        assert count >= 2  # "Potentially" and "Perhaps"
        assert 0 < rate < 1


class TestInfoDensity:
    def test_no_whitespace_text(self):
        """Text with no spaces has density 1.0."""
        assert compute_info_density({"field": "abc"}) == 1.0

    def test_half_whitespace(self):
        """Text that's ~half spaces has density ~0.5."""
        density = compute_info_density({"field": "a b c d"})
        assert 0.5 < density < 0.7

    def test_nested_structure(self):
        """Handles nested dicts and lists."""
        density = compute_info_density({
            "goals": ["reduce cost", "improve speed"],
            "nested": {"inner": "value"},
        })
        assert 0 < density <= 1.0


class TestFieldLengthStats:
    def test_basic_stats(self):
        """Computes mean/median/max/min correctly."""
        persona = {
            "goals": ["short", "a slightly longer goal sentence here"],
            "pains": ["x"],
            "motivations": [],
            "objections": [],
            "channels": [],
            "vocabulary": [],
            "decision_triggers": [],
            "sample_quotes": [],
        }
        stats = compute_field_length_stats(persona)
        assert stats["min"] >= 1
        assert stats["max"] >= stats["min"]
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_empty_persona(self):
        """Empty lists produce zeroes."""
        persona = {f: [] for f in [
            "goals", "pains", "motivations", "objections",
            "channels", "vocabulary", "decision_triggers", "sample_quotes",
        ]}
        stats = compute_field_length_stats(persona)
        assert stats["mean"] == 0


class TestTokenEstimation:
    def test_short_text(self):
        assert estimate_tokens("hi") >= 1

    def test_longer_text(self):
        text = "This is a moderately long sentence with about fifteen words in it."
        tokens = estimate_tokens(text)
        assert 10 <= tokens <= 25


# ── Run with pytest ───────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
