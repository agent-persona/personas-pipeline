"""Tests for Experiment 6.12: Power-user heuristic."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "evals"))

from power_user_check import (
    PowerUserReport,
    PowerUserScore,
    check_power_user,
    score_persona_power_user,
)

POWER_PERSONA = {
    "name": "Alex the DevOps Engineer",
    "goals": ["automate deployment pipelines", "scale infrastructure"],
    "pains": ["manual configuration", "monitoring gaps"],
    "vocabulary": ["terraform", "kubernetes", "CI/CD", "API", "webhook", "docker"],
    "firmographics": {"role_titles": ["Senior DevOps Engineer", "Platform Lead"]},
    "summary": "Infrastructure architect automating everything",
    "sample_quotes": ["If it's not in the pipeline, it doesn't exist"],
}

CASUAL_PERSONA = {
    "name": "Sarah the Blogger",
    "goals": ["grow readership", "improve writing quality"],
    "pains": ["writer's block", "low traffic"],
    "vocabulary": ["content", "audience", "engagement", "storytelling"],
    "firmographics": {"role_titles": ["Content Creator"]},
    "summary": "Freelance blogger building her audience",
    "sample_quotes": ["Good writing takes time"],
}


class TestScoring:
    def test_power_user_high_score(self):
        s = score_persona_power_user(POWER_PERSONA)
        assert s.total_score > 0.3
        assert s.is_power_user
        assert s.goal_matches > 0
        assert s.vocab_matches > 0
        assert s.role_matches > 0

    def test_casual_user_low_score(self):
        s = score_persona_power_user(CASUAL_PERSONA)
        assert s.total_score < 0.25
        assert not s.is_power_user

    def test_empty_persona(self):
        s = score_persona_power_user({})
        assert s.total_score == 0.0
        assert not s.is_power_user


class TestCheckPowerUser:
    def test_found_when_present(self):
        report = check_power_user([POWER_PERSONA, CASUAL_PERSONA])
        assert report.power_user_found
        assert report.best_match == "Alex the DevOps Engineer"
        assert report.inclusion_rate == 1.0

    def test_not_found_when_absent(self):
        report = check_power_user([CASUAL_PERSONA])
        assert not report.power_user_found
        assert report.inclusion_rate == 0.0

    def test_scores_sorted_descending(self):
        report = check_power_user([CASUAL_PERSONA, POWER_PERSONA])
        assert report.scores[0].total_score >= report.scores[1].total_score

    def test_empty_list(self):
        report = check_power_user([])
        assert not report.power_user_found
        assert report.n_personas == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
