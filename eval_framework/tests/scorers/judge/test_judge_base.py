"""Tests for shared judge scorer utilities in _base.py."""
from __future__ import annotations

import pytest
from persona_eval.schemas import CommunicationStyle, EmotionalProfile, MoralFramework, Persona
from persona_eval.scorers.judge._base import build_persona_block, normalize, parse_score


def test_parse_score_valid_json():
    score, reasoning, ok = parse_score('{"score": 4, "reasoning": "Good."}')
    assert score == 4.0
    assert reasoning == "Good."
    assert ok is True


def test_parse_score_json_with_float():
    score, _, ok = parse_score('{"score": 4.5, "reasoning": "Almost perfect."}')
    assert score == 4.5
    assert ok is True


def test_parse_score_regex_fallback():
    """When JSON is invalid but "score": N pattern is present, use regex."""
    score, _, ok = parse_score('Some text here "score": 3 rest of response.')
    assert score == 3.0
    assert ok is True


def test_parse_score_malformed_returns_3_and_false():
    """Completely malformed → fallback to 3.0, parse_ok=False."""
    score, reasoning, ok = parse_score("I am a mock LLM response.")
    assert score == 3.0
    assert ok is False


def test_parse_score_empty_string():
    score, _, ok = parse_score("")
    assert score == 3.0
    assert ok is False


def test_parse_score_score_outside_regex_range():
    """Score of 10 not matched by regex (only 1–5) → fallback."""
    score, _, ok = parse_score('"score": 10 this is invalid')
    assert score == 3.0
    assert ok is False


def test_normalize_1_to_5():
    assert normalize(1.0) == 0.0
    assert normalize(5.0) == 1.0
    assert normalize(3.0) == 0.5


def test_normalize_clamps_out_of_range():
    assert normalize(0.0) == 0.0   # below range → clamp to 0
    assert normalize(6.0) == 1.0   # above range → clamp to 1


def test_normalize_pass_threshold():
    """normalize(3.0) == 0.5 < 0.6 → fails. normalize(4.0) == 0.75 → passes."""
    assert normalize(3.0) < 0.6
    assert normalize(4.0) >= 0.6


def test_build_persona_block_excludes_empty_fields():
    """Lines ending in ': ' (empty value) must not appear in output."""
    sparse = Persona(id="x", name="Test", occupation="")
    block = build_persona_block(sparse)
    assert "Occupation: \n" not in block
    assert "Occupation: " not in block


def test_build_persona_block_includes_populated_fields():
    p = Persona(id="x", name="Test", occupation="Engineer", bio="Test bio.")
    block = build_persona_block(p)
    assert "Engineer" in block
    assert "Test bio." in block


def test_build_persona_block_includes_bio():
    p = Persona(id="x", name="Test", bio="A rich backstory here.")
    block = build_persona_block(p)
    assert "A rich backstory here." in block
