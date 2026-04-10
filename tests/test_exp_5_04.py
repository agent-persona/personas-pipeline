"""Tests for Experiment 5.04: Position & verbosity bias.

Verifies:
1. LLMJudge.pairwise() parses A/B/TIE from backend response
2. pad_persona_with_filler increases content length
3. normalize_winner correctly handles swapped order
4. Filler doesn't corrupt required persona fields
5. BiasReport computes flip_rate and length_wins_rate correctly
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "evals"))

import asyncio
import json

from evaluation.judges import LLMJudge, JudgeScore

from pairwise_biases import (
    FILLER_FIELDS,
    FILLER_SENTENCES,
    BiasReport,
    PositionBiasResult,
    VerbosityBiasResult,
    compute_report,
    normalize_winner,
    pad_persona_with_filler,
)


# ── Mock backend for testing ──────────────────────────────────────────

class MockJudgeBackend:
    """Returns predefined responses for testing judge parsing."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_index = 0
        self.total_cost = 0.0

    async def score(self, prompt: str) -> str:
        resp = self.responses[self.call_index % len(self.responses)]
        self.call_index += 1
        return resp


SAMPLE_PERSONA_1 = {
    "name": "Alex the Engineer",
    "summary": "A DevOps engineer focused on automation.",
    "goals": ["reduce deploy time", "improve observability"],
    "pains": ["slow CI pipelines", "alert fatigue"],
    "motivations": ["career growth"],
    "objections": ["too expensive"],
    "channels": ["twitter"],
    "vocabulary": ["deploy", "pipeline"],
    "decision_triggers": ["demo"],
    "sample_quotes": ["Ship it!"],
}

SAMPLE_PERSONA_2 = {
    "name": "Maya the Designer",
    "summary": "A freelance brand designer.",
    "goals": ["find clients", "build portfolio"],
    "pains": ["inconsistent income"],
    "motivations": ["creative freedom"],
    "objections": ["steep learning curve"],
    "channels": ["instagram"],
    "vocabulary": ["brand", "visual"],
    "decision_triggers": ["referral"],
    "sample_quotes": ["Design is communication."],
}


# ── Judge parsing tests ───────────────────────────────────────────────

class TestPairwiseJudge:
    def test_parses_a_winner(self):
        backend = MockJudgeBackend(["A\nPersona A is more grounded."])
        judge = LLMJudge(backend=backend)
        winner, rationale = asyncio.get_event_loop().run_until_complete(
            judge.pairwise(SAMPLE_PERSONA_1, SAMPLE_PERSONA_2)
        )
        assert winner == "A"

    def test_parses_b_winner(self):
        backend = MockJudgeBackend(["B\nPersona B is more distinctive."])
        judge = LLMJudge(backend=backend)
        winner, _ = asyncio.get_event_loop().run_until_complete(
            judge.pairwise(SAMPLE_PERSONA_1, SAMPLE_PERSONA_2)
        )
        assert winner == "B"

    def test_parses_tie(self):
        backend = MockJudgeBackend(["TIE\nBoth are equally good."])
        judge = LLMJudge(backend=backend)
        winner, _ = asyncio.get_event_loop().run_until_complete(
            judge.pairwise(SAMPLE_PERSONA_1, SAMPLE_PERSONA_2)
        )
        assert winner == "TIE"

    def test_handles_lowercase_response(self):
        backend = MockJudgeBackend(["a - Persona A is better"])
        judge = LLMJudge(backend=backend)
        winner, _ = asyncio.get_event_loop().run_until_complete(
            judge.pairwise(SAMPLE_PERSONA_1, SAMPLE_PERSONA_2)
        )
        assert winner == "A"

    def test_no_backend_returns_tie(self):
        judge = LLMJudge(backend=None)
        winner, _ = asyncio.get_event_loop().run_until_complete(
            judge.pairwise(SAMPLE_PERSONA_1, SAMPLE_PERSONA_2)
        )
        assert winner == "TIE"

    def test_pairwise_sends_both_personas(self):
        """Verify the prompt includes both persona JSONs."""
        prompts_seen = []

        class CapturingBackend:
            total_cost = 0.0
            async def score(self, prompt: str) -> str:
                prompts_seen.append(prompt)
                return "A\nFirst is better"

        judge = LLMJudge(backend=CapturingBackend())
        asyncio.get_event_loop().run_until_complete(
            judge.pairwise(SAMPLE_PERSONA_1, SAMPLE_PERSONA_2)
        )
        assert len(prompts_seen) == 1
        assert "Alex the Engineer" in prompts_seen[0]
        assert "Maya the Designer" in prompts_seen[0]
        assert "Persona A" in prompts_seen[0]
        assert "Persona B" in prompts_seen[0]


# ── Padding tests ─────────────────────────────────────────────────────

class TestPadding:
    def test_padding_increases_length(self):
        padded = pad_persona_with_filler(SAMPLE_PERSONA_1)
        original_len = len(json.dumps(SAMPLE_PERSONA_1))
        padded_len = len(json.dumps(padded))
        assert padded_len > original_len * 1.2  # at least 20% longer

    def test_padding_adds_to_list_fields(self):
        padded = pad_persona_with_filler(SAMPLE_PERSONA_1)
        for field in FILLER_FIELDS:
            if field in SAMPLE_PERSONA_1:
                assert len(padded[field]) > len(SAMPLE_PERSONA_1[field])

    def test_padding_preserves_original_items(self):
        padded = pad_persona_with_filler(SAMPLE_PERSONA_1)
        for field in FILLER_FIELDS:
            if field in SAMPLE_PERSONA_1:
                for item in SAMPLE_PERSONA_1[field]:
                    assert item in padded[field]

    def test_padding_expands_summary(self):
        padded = pad_persona_with_filler(SAMPLE_PERSONA_1)
        assert len(padded["summary"]) > len(SAMPLE_PERSONA_1["summary"])

    def test_padding_does_not_mutate_original(self):
        original_copy = json.dumps(SAMPLE_PERSONA_1)
        pad_persona_with_filler(SAMPLE_PERSONA_1)
        assert json.dumps(SAMPLE_PERSONA_1) == original_copy


# ── Normalize winner tests ────────────────────────────────────────────

class TestNormalizeWinner:
    def test_not_swapped(self):
        assert normalize_winner("A", swapped=False) == "A"
        assert normalize_winner("B", swapped=False) == "B"
        assert normalize_winner("TIE", swapped=False) == "TIE"

    def test_swapped(self):
        assert normalize_winner("A", swapped=True) == "B"
        assert normalize_winner("B", swapped=True) == "A"
        assert normalize_winner("TIE", swapped=True) == "TIE"


# ── Report computation tests ─────────────────────────────────────────

class TestComputeReport:
    def test_flip_rate_all_flipped(self):
        position = [
            PositionBiasResult("p1 vs p2", "A", "B", flipped=True, consistent_winner="INCONSISTENT"),
        ]
        mock_backend = MockJudgeBackend([])
        mock_backend.total_cost = 0.01
        report = compute_report(position, [], mock_backend, 10.0)
        assert report.flip_rate == 1.0

    def test_flip_rate_none_flipped(self):
        position = [
            PositionBiasResult("p1 vs p2", "A", "A", flipped=False, consistent_winner="persona_1"),
        ]
        mock_backend = MockJudgeBackend([])
        mock_backend.total_cost = 0.01
        report = compute_report(position, [], mock_backend, 10.0)
        assert report.flip_rate == 0.0

    def test_length_wins_rate(self):
        verbosity = [
            VerbosityBiasResult("p1 vs p2 (A padded)", "A", "B", "A", length_won=True),
            VerbosityBiasResult("p1 vs p2 (B padded)", "B", "A", "B", length_won=False),
        ]
        mock_backend = MockJudgeBackend([])
        mock_backend.total_cost = 0.01
        report = compute_report([], verbosity, mock_backend, 10.0)
        assert report.length_wins_rate == 0.5

    def test_a_preference_calculation(self):
        position = [
            PositionBiasResult("p1 vs p2", "A", "A", flipped=False, consistent_winner="persona_1"),
        ]
        mock_backend = MockJudgeBackend([])
        mock_backend.total_cost = 0.0
        report = compute_report(position, [], mock_backend, 5.0)
        assert report.position_a_preference == 1.0  # A always won


# ── Run with pytest ───────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
