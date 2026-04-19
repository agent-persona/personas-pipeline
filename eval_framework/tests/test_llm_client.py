"""Tests for the LiteLLM client wrapper."""


def test_llm_client_importable():
    from persona_eval.llm_client import LLMClient
    assert LLMClient is not None


def test_llm_client_init():
    from persona_eval.llm_client import LLMClient
    client = LLMClient(model="claude-sonnet-4-20250514")
    assert client.model == "claude-sonnet-4-20250514"


def test_llm_client_format_messages():
    from persona_eval.llm_client import LLMClient
    client = LLMClient(model="test")
    messages = client.format_messages(
        system="You are a test assistant.",
        user="Hello",
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_llm_client_format_messages_with_assistant():
    from persona_eval.llm_client import LLMClient
    client = LLMClient(model="test")
    messages = client.format_messages(
        system="sys", user="usr", assistant="ast",
    )
    assert len(messages) == 3
    assert messages[2]["role"] == "assistant"


def test_llm_client_complete_uses_mock():
    """Verify the conftest auto-mock intercepts litellm.completion."""
    from persona_eval.llm_client import LLMClient
    client = LLMClient(model="test-model")
    result = client.complete([{"role": "user", "content": "hello"}])
    assert result == "I am a mock LLM response."
