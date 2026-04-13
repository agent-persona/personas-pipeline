"""Tests for Experiment 2.02: Critique and reflexion loops.

Verifies:
1. CritiqueScore data structure
2. RevisionRound tracking
3. Reflexion addendum prompt construction
4. Critic prompt content
5. Edge cases
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "synthesis"))

from synthesis.engine.critique import (
    CRITIC_SYSTEM_PROMPT,
    REFLEXION_ADDENDUM,
    CritiqueScore,
    RevisionRound,
    ReflexionResult,
)


# ── CritiqueScore tests ─────────────────────────────────────────────

class TestCritiqueScore:
    def test_fields(self):
        cs = CritiqueScore(
            overall=0.85,
            dimensions={"grounded": 0.9, "distinctive": 0.7},
            improvements=["improve voice", "add more quotes"],
            cost_usd=0.005,
        )
        assert cs.overall == 0.85
        assert cs.dimensions["distinctive"] == 0.7
        assert len(cs.improvements) == 2

    def test_empty_improvements(self):
        cs = CritiqueScore(overall=1.0, dimensions={}, improvements=[])
        assert cs.improvements == []


# ── RevisionRound tests ─────────────────────────────────────────────

class TestRevisionRound:
    def test_defaults(self):
        rr = RevisionRound(round_num=1)
        assert rr.round_num == 1
        assert rr.pre_critique is None
        assert rr.post_groundedness == 0.0
        assert rr.synthesis_cost_usd == 0.0

    def test_with_critique(self):
        critique = CritiqueScore(
            overall=0.75,
            dimensions={"distinctive": 0.6},
            improvements=["more quotes"],
        )
        rr = RevisionRound(
            round_num=2,
            pre_critique=critique,
            post_persona_name="Revised Persona",
            post_groundedness=0.95,
            synthesis_cost_usd=0.02,
            critique_cost_usd=0.005,
        )
        assert rr.pre_critique.overall == 0.75
        assert rr.post_persona_name == "Revised Persona"


# ── Prompt tests ─────────────────────────────────────────────────────

class TestPrompts:
    def test_critic_has_all_dimensions(self):
        for dim in ("grounded", "distinctive", "coherent", "actionable", "voice_fidelity"):
            assert dim in CRITIC_SYSTEM_PROMPT

    def test_critic_asks_for_improvements(self):
        assert "improvements" in CRITIC_SYSTEM_PROMPT

    def test_critic_asks_json(self):
        assert "JSON" in CRITIC_SYSTEM_PROMPT

    def test_reflexion_addendum_format(self):
        rendered = REFLEXION_ADDENDUM.format(
            round=1,
            improvements="- improve voice\n- add quotes",
            overall=0.72,
            worst_dim="distinctive",
            worst_score=0.55,
        )
        assert "Round 1" in rendered
        assert "improve voice" in rendered
        assert "0.72" in rendered
        assert "distinctive" in rendered
        assert "distinctiveness" in rendered.lower()  # warns about maintaining it

    def test_reflexion_warns_about_genericness(self):
        rendered = REFLEXION_ADDENDUM.format(
            round=1, improvements="x", overall=0.5,
            worst_dim="d", worst_score=0.3,
        )
        assert "generic" in rendered.lower()


# ── ReflexionResult tests ───────────────────────────────────────────

class TestReflexionResult:
    def test_zero_rounds_no_critiques(self):
        """0 revision rounds should have no initial/final critique."""
        from synthesis.models.persona import PersonaV1
        from synthesis.engine.groundedness import GroundednessReport

        rr = ReflexionResult(
            persona=None,  # type: ignore
            groundedness=GroundednessReport(score=1.0),
            total_cost_usd=0.03,
            model_used="test",
            initial_attempts=1,
            revision_rounds=[],
        )
        assert rr.initial_critique is None
        assert rr.final_critique is None
        assert len(rr.revision_rounds) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
