from types import SimpleNamespace

import pytest

from evaluation.judges import LLMJudge, _extract_anthropic_text


def test_extract_anthropic_text_skips_thinking_blocks():
    content = [
        SimpleNamespace(type="thinking", thinking="internal"),
        SimpleNamespace(type="text", text='{"scores":{"groundedness":5}}'),
        SimpleNamespace(type="text", text='{"rationale":"ok"}'),
    ]

    assert _extract_anthropic_text(content) == (
        '{"scores":{"groundedness":5}}\n{"rationale":"ok"}'
    )


def test_llmjudge_uses_backend_model_for_metadata():
    backend = SimpleNamespace(model="MiniMax-M2.7")

    judge = LLMJudge(backend=backend)

    assert judge.model == "MiniMax-M2.7"


class _FakeBackend:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.model = "fake-judge"
        self.prompts: list[str] = []

    async def score(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.payload


@pytest.mark.asyncio
async def test_score_persona_with_context_respects_custom_dimensions_and_reference_context():
    backend = _FakeBackend('{"scores":{"groundedness":5,"overall_preference":3},"rationale":"ok"}')
    judge = LLMJudge(backend=backend)

    result = await judge.score_persona_with_context(
        {"name": "Alex"},
        dimensions=("groundedness", "overall_preference"),
        reference_context={"record_ids": ["rec_1"]},
    )

    assert result.overall == 4.0
    assert result.dimensions == {"groundedness": 5.0, "overall_preference": 3.0}
    assert "Reference Context" in backend.prompts[0]
    assert '"record_ids": [' in backend.prompts[0]
