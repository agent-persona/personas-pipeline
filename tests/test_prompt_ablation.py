"""Tests for prompt section ablation (Experiment 2.16)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure imports work
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "synthesis"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "segmentation"))
sys.path.insert(0, str(REPO_ROOT / "crawler"))
sys.path.insert(0, str(REPO_ROOT / "orchestration"))
sys.path.insert(0, str(REPO_ROOT / "twin"))


# ---------------------------------------------------------------------------
# Section identification tests
# ---------------------------------------------------------------------------

class TestSectionIdentification:
    def test_system_sections_has_7_entries(self):
        from synthesis.engine.prompt_builder import SYSTEM_PROMPT_SECTIONS
        assert len(SYSTEM_PROMPT_SECTIONS) == 7

    def test_system_section_names(self):
        from synthesis.engine.prompt_builder import SYSTEM_PROMPT_SECTIONS
        expected = [
            "preamble", "quality_grounded", "quality_distinctive",
            "quality_actionable", "quality_consistent",
            "evidence_rules", "evidence_example",
        ]
        assert list(SYSTEM_PROMPT_SECTIONS.keys()) == expected

    def test_user_message_sections_has_6_entries(self):
        from synthesis.engine.prompt_builder import USER_MESSAGE_SECTIONS
        assert len(USER_MESSAGE_SECTIONS) == 6

    def test_user_message_section_names(self):
        from synthesis.engine.prompt_builder import USER_MESSAGE_SECTIONS
        expected = [
            "tenant_context", "cluster_summary", "sample_records",
            "enrichment", "available_record_ids", "instruction_footer",
        ]
        assert USER_MESSAGE_SECTIONS == expected


# ---------------------------------------------------------------------------
# Section removal correctness tests
# ---------------------------------------------------------------------------

class TestSectionRemoval:
    def test_build_system_prompt_no_exclusions_matches_original(self):
        """Safety net: default build must match SYSTEM_PROMPT exactly."""
        from synthesis.engine.prompt_builder import (
            SYSTEM_PROMPT, build_system_prompt,
        )
        assert build_system_prompt() == SYSTEM_PROMPT

    def test_build_system_prompt_no_exclusions_returns_string(self):
        from synthesis.engine.prompt_builder import build_system_prompt
        result = build_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize("section", [
        "preamble", "quality_grounded", "quality_distinctive",
        "quality_actionable", "quality_consistent",
        "evidence_rules", "evidence_example",
    ])
    def test_build_system_prompt_excludes_each_section(self, section):
        from synthesis.engine.prompt_builder import (
            SYSTEM_PROMPT, SYSTEM_PROMPT_SECTIONS, build_system_prompt,
        )
        result = build_system_prompt(exclude_sections={section})
        # Should be shorter
        assert len(result) < len(SYSTEM_PROMPT)
        # Excluded section text should not appear
        excluded_text = SYSTEM_PROMPT_SECTIONS[section]
        assert excluded_text not in result

    def test_build_system_prompt_exclude_multiple(self):
        from synthesis.engine.prompt_builder import SYSTEM_PROMPT, build_system_prompt
        result = build_system_prompt(
            exclude_sections={"evidence_rules", "evidence_example"}
        )
        assert len(result) < len(SYSTEM_PROMPT)

    def test_build_system_prompt_exclude_all_returns_empty(self):
        from synthesis.engine.prompt_builder import (
            SYSTEM_PROMPT_SECTIONS, build_system_prompt,
        )
        all_sections = set(SYSTEM_PROMPT_SECTIONS.keys())
        result = build_system_prompt(exclude_sections=all_sections)
        assert result == ""

    def test_build_system_prompt_exclude_nonexistent_section_no_effect(self):
        from synthesis.engine.prompt_builder import SYSTEM_PROMPT, build_system_prompt
        result = build_system_prompt(exclude_sections={"nonexistent_section"})
        assert result == SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Metric collection tests
# ---------------------------------------------------------------------------

class TestMetricsCollector:
    def test_collect_metrics_empty_dir(self):
        from evaluation.experiments.metrics_collector import MetricsCollector
        mc = MetricsCollector()
        with tempfile.TemporaryDirectory() as tmp:
            result = mc.collect_from_output(Path(tmp))
            assert result["personas_generated"] == 0
            assert result["schema_validity"] == 0.0

    def test_collect_metrics_success_path(self):
        from evaluation.experiments.metrics_collector import MetricsCollector
        mc = MetricsCollector()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Write mock persona files
            for i in range(2):
                data = {
                    "persona": {
                        "schema_version": "v1",
                        "name": f"Persona {i}",
                        "summary": "A test persona",
                        "demographics": {"age_range": "30-40"},
                        "firmographics": {"company_size": "50-200"},
                        "goals": ["goal1"],
                        "pains": ["pain1"],
                        "motivations": ["motivation1"],
                        "objections": ["objection1"],
                        "channels": ["email"],
                        "vocabulary": ["word1"],
                        "decision_triggers": ["trigger1"],
                        "sample_quotes": ["quote1"],
                        "journey_stages": [{"stage": "awareness", "behavior": "searching"}],
                        "source_evidence": [{
                            "claim": "test",
                            "record_ids": ["rec_001"],
                            "field_path": "goals.0",
                            "confidence": 0.9,
                        }],
                    },
                    "groundedness": 0.95,
                    "cost_usd": 0.003,
                    "attempts": 1,
                }
                (tmp_path / f"persona_{i:02d}.json").write_text(
                    json.dumps(data, indent=2)
                )
            result = mc.collect_from_output(tmp_path)
            assert result["personas_generated"] == 2.0
            assert result["mean_groundedness"] == 0.95
            assert result["total_cost_usd"] == pytest.approx(0.006)
            assert result["cost_per_persona"] == pytest.approx(0.003)

    def test_compute_signal_strength_strong(self):
        from evaluation.experiments.metrics_collector import MetricsCollector
        mc = MetricsCollector()
        baseline = {"target": 0.5, "schema_validity": 1.0, "mean_groundedness": 0.9}
        experiment = {"target": 0.7, "schema_validity": 1.0, "mean_groundedness": 0.92}
        signal = mc.compute_signal_strength(baseline, experiment, "target", "increase")
        assert signal == "strong"

    def test_compute_signal_strength_noise(self):
        # direction is "increase" but experiment is lower — confirmed=False, no regressions → noise
        from evaluation.experiments.metrics_collector import MetricsCollector
        mc = MetricsCollector()
        baseline = {"target": 0.5, "schema_validity": 1.0}
        experiment = {"target": 0.495, "schema_validity": 1.0}
        signal = mc.compute_signal_strength(baseline, experiment, "target", "increase")
        assert signal == "noise"

    def test_compute_signal_strength_negative_schema(self):
        from evaluation.experiments.metrics_collector import MetricsCollector
        mc = MetricsCollector()
        baseline = {"target": 0.5, "schema_validity": 1.0}
        experiment = {"target": 0.6, "schema_validity": 0.8}
        signal = mc.compute_signal_strength(baseline, experiment, "target", "increase")
        assert signal == "negative"

    def test_build_comparison(self):
        from evaluation.experiments.config import ExperimentConfig
        from evaluation.experiments.metrics_collector import MetricsCollector
        mc = MetricsCollector()
        config = ExperimentConfig(
            number="2.16", title="test", description="test",
            files=[], primary_metric="target",
            branch_name="exp-2.16-test",
        )
        baseline = {"target": 0.5, "schema_validity": 1.0, "mean_groundedness": 0.9, "total_cost_usd": 0.01, "personas_generated": 2.0}
        experiment = {"target": 0.7, "schema_validity": 1.0, "mean_groundedness": 0.92, "total_cost_usd": 0.012, "personas_generated": 2.0}
        result = mc.build_comparison(config, baseline, experiment)
        assert result.signal_strength == "strong"
        assert result.recommendation == "adopt"
        assert result.deltas["target"] == pytest.approx(0.2)
