"""Shared fixtures for judge scorer tests."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from persona_eval.schemas import (
    CommunicationStyle,
    EmotionalProfile,
    MoralFramework,
    Persona,
)
from persona_eval.source_context import SourceContext


@pytest.fixture
def judge_persona():
    return Persona(
        id="judge-test",
        name="Alice Chen",
        age=34,
        occupation="Software Engineer",
        industry="Technology",
        education="MS Computer Science",
        personality_traits=["analytical", "introverted", "detail-oriented"],
        values=["meritocracy", "transparency", "craftsmanship"],
        goals=["lead a team"],
        pain_points=["meeting overload"],
        behaviors=["deep work sessions", "code review thoroughness"],
        interests=["rock climbing"],
        knowledge_domains=["distributed systems", "Python"],
        communication_style=CommunicationStyle(
            tone="direct", formality="informal", vocabulary_level="advanced"
        ),
        emotional_profile=EmotionalProfile(baseline_mood="calm"),
        moral_framework=MoralFramework(
            core_values=["fairness", "honesty"], ethical_stance="utilitarian"
        ),
        bio="Alice is a senior software engineer at a Bay Area startup who cares deeply about code quality.",
    )


@pytest.fixture
def judge_responses():
    return [
        "I spent the afternoon chasing down a race condition in the distributed lock manager. "
        "Turned out to be a subtle ordering bug under high contention. Fixed it, added a regression test.",
        "I don't really do a lot of 'inspiring' — I just write good code and hope the work speaks "
        "for itself. I find most team-building exercises kind of cringeworthy, honestly.",
        "The PR process slows us down but I still think it's worth it. Catches more bugs than it costs.",
    ]


@pytest.fixture
def mock_llm_score_4():
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = '{"score": 4, "reasoning": "Strong persona."}'
    with patch("litellm.completion", return_value=mock):
        yield


@pytest.fixture
def mock_llm_score_2():
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = '{"score": 2, "reasoning": "Weak persona."}'
    with patch("litellm.completion", return_value=mock):
        yield


@pytest.fixture
def mock_llm_score_3():
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = '{"score": 3, "reasoning": "Adequate persona."}'
    with patch("litellm.completion", return_value=mock):
        yield
