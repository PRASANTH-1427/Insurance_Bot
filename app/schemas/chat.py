from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request from the user."""

    message: str = Field(..., description="User message content")
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for multi-turn conversations; auto-generated if omitted",
    )


class ChatResponse(BaseModel):
    """Response sent back to the user."""

    session_id: str = Field(..., description="Active session ID")
    reply: str = Field(..., description="Assistant's response")
    intent: str | None = Field(default=None, description="Classified intent")
    reason: str | None = Field(default=None, description="Internal routing explanation")
    active_agent: str | None = Field(default=None, description="Agent handling this response")

