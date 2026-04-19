"""Tests for ConversationRunner infrastructure."""

from persona_eval.schemas import Persona


SAMPLE_PERSONA = Persona(
    id="p1", name="Alice Chen", age=32, occupation="Product Manager",
    communication_style={"tone": "professional", "formality": "medium"},
)


def test_turn_importable():
    from persona_eval.conversation import Turn
    t = Turn(role="interviewer", content="Hello", turn_number=0)
    assert t.role == "interviewer"
    assert t.turn_number == 0


def test_conversation_add_turn():
    from persona_eval.conversation import Conversation
    convo = Conversation(persona_id="p1")
    convo.add_turn("interviewer", "What do you do?")
    convo.add_turn("persona", "I'm a product manager.")
    assert convo.length == 2
    assert convo.turns[0].role == "interviewer"
    assert convo.turns[1].turn_number == 1


def test_conversation_to_messages():
    from persona_eval.conversation import Conversation
    convo = Conversation(persona_id="p1")
    convo.add_turn("interviewer", "Question")
    convo.add_turn("persona", "Answer")
    msgs = convo.to_messages()
    assert msgs[0] == {"role": "user", "content": "Question"}
    assert msgs[1] == {"role": "assistant", "content": "Answer"}


def test_conversation_runner_init():
    from persona_eval.conversation import ConversationRunner
    from persona_eval.llm_client import LLMClient
    client = LLMClient()
    runner = ConversationRunner(llm_client=client, persona=SAMPLE_PERSONA)
    assert "Alice Chen" in runner.system_prompt
    assert "Product Manager" in runner.system_prompt


def test_conversation_runner_scripted():
    """ConversationRunner.run_scripted builds correct conversation."""
    from unittest.mock import MagicMock
    from persona_eval.conversation import ConversationRunner

    mock_client = MagicMock()
    mock_client.complete.return_value = "I manage products."
    runner = ConversationRunner(llm_client=mock_client, persona=SAMPLE_PERSONA)
    convo = runner.run_scripted(["What do you do?", "Tell me more."])
    assert convo.length == 4  # 2 questions + 2 responses
    assert convo.turns[0].role == "interviewer"
    assert convo.turns[1].role == "persona"
    assert mock_client.complete.call_count == 2


def test_conversation_runner_dynamic():
    """ConversationRunner.run_dynamic alternates persona/interviewer."""
    from unittest.mock import MagicMock
    from persona_eval.conversation import ConversationRunner

    mock_client = MagicMock()
    mock_client.complete.side_effect = [
        "I think we should discuss that.",  # persona response
        "Can you elaborate?",               # interviewer follow-up
        "Sure, here's more detail.",        # persona response
        "Interesting, what else?",          # interviewer follow-up
    ]
    runner = ConversationRunner(llm_client=mock_client, persona=SAMPLE_PERSONA)
    convo = runner.run_dynamic("Tell me about yourself.", max_turns=2)
    # Should have: initial question + 2 rounds of (persona + interviewer)
    assert convo.length >= 4
    assert convo.turns[0].role == "interviewer"
    assert convo.turns[1].role == "persona"
