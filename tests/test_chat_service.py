from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService


class MockGraph:
    """Mock LangGraph for unit testing ChatService without external dependencies."""

    def __init__(self) -> None:
        self.invocation_count = 0
        self.last_input_state = None

    async def ainvoke(self, state: dict) -> dict:
        self.invocation_count += 1
        self.last_input_state = state

        user_message = state["messages"][-1].content
        reply_text = f"Echo: {user_message}"

        return {
            "messages": state["messages"] + [AIMessage(content=reply_text)],
            "intent": "general",
            "reason": "testing routing",
            "active_agent": "main_agent",
            "lead_state": {"lead_id": "test-lead"},
            "insurance_state": {},
            "memory": {"session_id": "mock-session"},
        }


@pytest.mark.anyio
async def test_chat_service_process_message_new_session() -> None:
    mock_graph = MockGraph()
    service = ChatService(graph=mock_graph)

    response = await service.process_message(message="Hello there!")

    assert isinstance(response, ChatResponse)
    assert response.reply == "Echo: Hello there!"
    assert response.intent == "general"
    assert response.reason == "testing routing"
    assert response.active_agent == "main_agent"
    assert response.session_id is not None
    assert mock_graph.invocation_count == 1


@pytest.mark.anyio
async def test_chat_service_multi_turn_history() -> None:
    mock_graph = MockGraph()
    service = ChatService(graph=mock_graph)
    session_id = "test-session-123"

    # Turn 1
    resp1 = await service.process_message(message="First message", session_id=session_id)
    assert resp1.session_id == session_id
    assert resp1.reply == "Echo: First message"

    # Turn 2 with same session_id
    resp2 = await service.process_message(message="Second message", session_id=session_id)
    assert resp2.session_id == session_id
    assert resp2.reply == "Echo: Second message"

    # Verify that turn 2 received 3 messages in state (Turn 1 human, Turn 1 AI, Turn 2 human)
    last_messages = mock_graph.last_input_state["messages"]
    assert len(last_messages) == 3
    assert last_messages[0].content == "First message"
    assert last_messages[1].content == "Echo: First message"
    assert last_messages[2].content == "Second message"

