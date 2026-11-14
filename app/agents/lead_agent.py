"""
Lead Agent
==========

Responsibilities:
  - Collect the four required fields from the user: name, email, phone, interest.
  - Validate email and phone formats before saving.
  - Save the lead to the database via the save_lead tool.
  - Update ``LeadState`` on every turn so the graph can track progress.

Approach:
  Uses Pydantic structured output (``with_structured_output``) on each turn so the
  LLM is *forced* to produce a validated, type-safe response. This is more reliable
  than parsing free-text or raw JSON.

Multi-turn flow:
  Turn 1: User expresses interest → agent asks for name.
  Turn 2: User provides name → agent stores it, asks for email.
  Turn 3: User provides email → agent validates it, asks for phone.
  Turn 4: User provides phone → agent validates it, asks for insurance interest.
  Turn 5: User provides interest → agent calls save_lead tool → done.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graphs.state import LeadState, MainAgentState, MemoryState
from app.tools.lead_tools import save_lead, validate_email, validate_phone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output schema for each LeadAgent turn
# ---------------------------------------------------------------------------


class LeadAgentDecision(BaseModel):
    """What the LeadAgent decides on each turn."""

    # Fields extracted from the user's latest message (None if not mentioned)
    extracted_name: Optional[str] = Field(
        default=None,
        description="User's full name if they provided it in this turn, else null.",
    )
    extracted_email: Optional[str] = Field(
        default=None,
        description="User's email address if they provided it in this turn, else null.",
    )
    extracted_phone: Optional[str] = Field(
        default=None,
        description="User's phone number if they provided it in this turn, else null.",
    )
    extracted_interest: Optional[str] = Field(
        default=None,
        description="Insurance type user is interested in (auto/health/life/home) if mentioned, else null.",
    )

    # What to say to the user
    response: str = Field(
        description="The message to send to the user. Ask for the next missing field or confirm completion."
    )

    # Whether all four fields are now available and validated
    ready_to_save: bool = Field(
        description=(
            "True ONLY when all four fields (name, email, phone, interest) are collected "
            "AND email/phone formats look valid. False otherwise."
        )
    )

    # Operational status
    status: Literal["in_progress", "completed", "failed"] = Field(
        description=(
            "'in_progress' while still collecting fields. "
            "'completed' after the lead is saved successfully. "
            "'failed' if the user explicitly refuses to provide info."
        )
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

LEAD_AGENT_SYSTEM_PROMPT = """You are the Lead Collection specialist for InsureAssist.

Your goal is to collect exactly four pieces of information from the user, one at a time:
  1. Full name
  2. Email address
  3. Phone number
  4. Type of insurance they are interested in (auto / health / life / home)

RULES:
- Ask for ONE field at a time. Do not ask for multiple fields in a single message.
- Be friendly, warm, and professional at all times.
- If the user provides an invalid email (no @) or an obviously invalid phone, politely \
ask them to provide a correct one before moving on.
- Once all four fields are collected and look valid, set ready_to_save = true.
- If the user says they don't want to proceed or refuses to share information, set status = "failed".
- Never make up or guess field values. Only use what the user explicitly provides.

FIELD COLLECTION ORDER (follow strictly unless user volunteers info out of order):
  name → email → phone → interest

CURRENT STATE will be injected below as a system context message. Use it to know what \
has already been collected so you don't ask for the same field twice.
"""


# ---------------------------------------------------------------------------
# Lead Agent class
# ---------------------------------------------------------------------------


class LeadAgent:
    """
    Agentic lead collection agent.

    On each turn it:
      1. Reviews the current LeadState to know what fields are already collected.
      2. Calls the LLM (with structured output) to extract new info and decide what to ask next.
      3. Merges extracted fields into LeadState.
      4. If ready_to_save: calls validate_email / validate_phone tools, then save_lead tool.
      5. Returns updated state to the LangGraph.
    """

    def __init__(self, model: str | None = None, temperature: float = 0.2):
        settings = get_settings()
        model = model or settings.groq_model
        api_key = settings.groq_api_key or None
        base_llm = ChatGroq(model=model, temperature=temperature, api_key=api_key)
        self.llm = base_llm.with_structured_output(LeadAgentDecision)
        self._model_name = model

    # ------------------------------------------------------------------
    # LangGraph node entry-point
    # ------------------------------------------------------------------

    async def __call__(self, state: MainAgentState) -> dict:
        """
        Run one turn of the lead collection loop.

        Returns a partial state update (messages, lead_state, memory, active_agent).
        """
        messages = state.get("messages", [])
        lead_state: LeadState = dict(state.get("lead_state", {}))
        memory: MemoryState = dict(state.get("memory", {}))

        # Initialise status on first entry
        if not lead_state.get("status"):
            lead_state["status"] = "in_progress"
            lead_state["collected_fields"] = []

        # Build the LLM prompt
        llm_messages = self._build_messages(messages, lead_state)

        try:
            decision: LeadAgentDecision = await self.llm.ainvoke(llm_messages)
        except Exception as exc:
            logger.error("LeadAgent LLM call failed: %s", exc)
            return {
                "messages": [AIMessage(content="I'm having trouble right now. Could you please try again?")],
                "lead_state": lead_state,
                "active_agent": "lead_agent",
            }

        # Merge newly extracted fields into lead_state
        lead_state = self._merge_extracted_fields(lead_state, decision)

        # If all fields collected: validate and save
        if decision.ready_to_save and lead_state.get("status") == "in_progress":
            save_result, response_override = await self._validate_and_save(lead_state)

            if save_result.get("status") == "saved":
                lead_state["lead_id"] = save_result["lead_id"]
                lead_state["status"] = "completed"
                # Update shared memory with user info
                memory["user_name"] = lead_state.get("name", memory.get("user_name"))
                memory["user_email"] = lead_state.get("email", memory.get("user_email"))
                memory["user_phone"] = lead_state.get("phone", memory.get("user_phone"))
                final_response = response_override or decision.response
            else:
                # Validation failed — keep in_progress, ask user to retry
                final_response = response_override or decision.response
        else:
            lead_state["status"] = decision.status
            final_response = decision.response

        # Track last question asked
        lead_state["last_question"] = final_response

        ai_message = AIMessage(content=final_response)

        logger.info(
            "LeadAgent | status=%s | collected=%s | lead_id=%s",
            lead_state.get("status"),
            lead_state.get("collected_fields"),
            lead_state.get("lead_id"),
        )

        return {
            "messages": [ai_message],
            "lead_state": lead_state,
            "memory": memory,
            "active_agent": "lead_agent",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_messages(self, messages: list, lead_state: LeadState) -> list:
        """Build the prompt list including current state context."""
        llm_messages = [SystemMessage(content=LEAD_AGENT_SYSTEM_PROMPT)]

        # Inject current lead state so LLM knows what's already collected
        collected = lead_state.get("collected_fields", [])
        known = {
            "name": lead_state.get("name"),
            "email": lead_state.get("email"),
            "phone": lead_state.get("phone"),
            "interest": lead_state.get("interest"),
        }
        state_note = (
            f"[Current Lead State]\n"
            f"Already collected fields: {collected or 'none yet'}\n"
            f"Known values: {', '.join(f'{k}={v}' for k, v in known.items() if v) or 'none yet'}\n"
            f"Status: {lead_state.get('status', 'in_progress')}"
        )
        llm_messages.append(SystemMessage(content=state_note))

        # Include last 10 messages
        recent = messages[-10:] if len(messages) > 10 else messages
        llm_messages.extend(recent)

        return llm_messages

    @staticmethod
    def _merge_extracted_fields(lead_state: LeadState, decision: LeadAgentDecision) -> LeadState:
        """Merge fields extracted by the LLM into lead_state."""
        collected = list(lead_state.get("collected_fields") or [])

        field_map = {
            "name": decision.extracted_name,
            "email": decision.extracted_email,
            "phone": decision.extracted_phone,
            "interest": decision.extracted_interest,
        }

        for field, value in field_map.items():
            if value and field not in collected:
                lead_state[field] = value  # type: ignore[literal-required]
                collected.append(field)

        lead_state["collected_fields"] = collected
        return lead_state

    @staticmethod
    async def _validate_and_save(lead_state: LeadState) -> tuple[dict, Optional[str]]:
        """
        Validate email/phone and save the lead if valid.

        Returns:
            (save_result dict, optional response override string)
        """
        # Validate email
        email_result = await validate_email.ainvoke({"email": lead_state["email"]})
        if not email_result["is_valid"]:
            lead_state["email"] = None
            if "email" in lead_state.get("collected_fields", []):
                lead_state["collected_fields"].remove("email")
            return {"status": "invalid"}, (
                f"{email_result['message']} Could you please provide your email again?"
            )

        # Validate phone
        phone_result = await validate_phone.ainvoke({"phone": lead_state["phone"]})
        if not phone_result["is_valid"]:
            lead_state["phone"] = None
            if "phone" in lead_state.get("collected_fields", []):
                lead_state["collected_fields"].remove("phone")
            return {"status": "invalid"}, (
                f"{phone_result['message']} Could you please provide your phone number again?"
            )

        # Save to store
        save_result = await save_lead.ainvoke({
            "name": lead_state["name"],
            "email": lead_state["email"],
            "phone": lead_state["phone"],
            "interest": lead_state["interest"],
        })
        return save_result, save_result.get("message")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_lead_agent(model: str | None = None) -> LeadAgent:
    """Create and return a configured LeadAgent instance."""
    return LeadAgent(model=model)
