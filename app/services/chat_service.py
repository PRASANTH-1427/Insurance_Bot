from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.graphs.workflow import get_graph
from app.schemas.chat import ChatResponse

logger = logging.getLogger(__name__)


class ChatService:
    """
    Service responsible for orchestrating conversation sessions and invoking
    the InsureAssist multi-agent graph.
    """

    def __init__(self, graph: Any | None = None) -> None:
        self._graph = graph
        # In-memory session store: session_id -> state dict
        # In production, replace or back this with Redis, a SQL database, or LangGraph Checkpointer.
        self._sessions: dict[str, dict[str, Any]] = {}

    @property
    def graph(self) -> Any:
        if self._graph is None:
            self._graph = get_graph()
        return self._graph

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Retrieve existing state for a session, or initialize a new empty state."""
        return self._sessions.get(
            session_id,
            {
                "messages": [],
                "lead_state": {},
                "insurance_state": {},
                "memory": {},
                "active_agent": None,
            },
        )

    async def process_message(
        self,
        message: str,
        session_id: str | None = None,
    ) -> ChatResponse:
        """
        Process a user message through the multi-agent graph.

        1. Resolves or generates the session_id.
        2. Retrieves ongoing conversation state and history.
        3. Invokes the multi-agent LangGraph workflow.
        4. Extracts the agent's reply and updates the session store.
        5. Returns a structured ChatResponse.
        """
        session_id = session_id or str(uuid.uuid4())
        session = self.get_session(session_id)

        history: list[BaseMessage] = session.get("messages", [])
        human_msg = HumanMessage(content=message)

        input_state: dict[str, Any] = {
            "messages": history + [human_msg],
            "lead_state": session.get("lead_state", {}),
            "insurance_state": session.get("insurance_state", {}),
            "memory": session.get("memory", {}),
            "active_agent": session.get("active_agent"),
            "intent": None,
            "reason": None,
            "next_node": None,
        }

        logger.info("Executing graph for session_id=%s with active_agent=%s", session_id, session.get("active_agent"))
        result = await self.graph.ainvoke(input_state)

        # Extract all messages and locate the last AI response
        all_messages: list[BaseMessage] = result.get("messages", [])
        ai_messages = [m for m in all_messages if isinstance(m, AIMessage)]
        reply = ai_messages[-1].content if ai_messages else "I'm sorry, something went wrong."

        # Persist updated session state
        self._sessions[session_id] = {
            "messages": all_messages,
            "lead_state": result.get("lead_state", {}),
            "insurance_state": result.get("insurance_state", {}),
            "memory": result.get("memory", {}),
            "active_agent": result.get("active_agent"),
        }

        return ChatResponse(
            session_id=session_id,
            reply=str(reply),
            intent=result.get("intent"),
            reason=result.get("reason"),
            active_agent=result.get("active_agent"),
        )


# Global singleton instance for in-memory session persistence across requests
_chat_service_instance: ChatService | None = None


def get_chat_service() -> ChatService:
    """Dependency provider for ChatService."""
    global _chat_service_instance
    if _chat_service_instance is None:
        _chat_service_instance = ChatService()
    return _chat_service_instance

