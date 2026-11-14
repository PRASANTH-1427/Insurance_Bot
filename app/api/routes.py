"""
API Routes
==========

Exposes the InsureAssist multi-agent graph over HTTP.

Endpoints:
  GET  /health  – liveness check
  POST /chat    – send a user message and receive an AI response
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService, get_chat_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["core"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness check endpoint."""
    return {"status": "ok", "service": "insureassist-ai"}


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """
    Send a user message to the InsureAssist multi-agent graph and receive a reply.

    The session_id links turns within the same conversation. If omitted, a new
    session is created automatically.
    """
    try:
        return await chat_service.process_message(
            message=request.message,
            session_id=request.session_id,
        )
    except Exception:
        logger.exception("Failed to process chat message for session_id=%s", request.session_id)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your request.",
        )
