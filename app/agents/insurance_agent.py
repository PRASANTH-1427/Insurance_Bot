"""
Insurance Agent
===============

Responsibilities:
  - Look up an existing policy by policy number.
  - Identify which field the user wants to update.
  - Apply the update via the update_policy_field tool.
  - Update ``InsuranceState`` on every turn.

Approach:
  Uses Pydantic structured output on each turn for reliable field extraction.
  Multi-turn: asks for policy number → what to update → confirms → applies update.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from langchain_core.messages import AIMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graphs.state import InsuranceState, MainAgentState, MemoryState
from app.tools.insurance_tools import (
    estimate_premium,
    list_policy_types,
    lookup_policy,
    update_policy_field,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class InsuranceAgentDecision(BaseModel):
    """What the InsuranceAgent decides on each turn."""

    # Fields extracted from user message
    extracted_policy_number: Optional[str] = Field(
        default=None,
        description="Policy number if the user provided one in this turn, else null.",
    )
    extracted_update_field: Optional[str] = Field(
        default=None,
        description="The field the user wants to update (e.g. contact_email, coverage_amount, beneficiary), else null.",
    )
    extracted_update_value: Optional[str] = Field(
        default=None,
        description="The new value for the update field if provided, else null.",
    )

    # What to say
    response: str = Field(
        description="Message to send to the user. Ask for next missing info or confirm what was done."
    )

    # Readiness flags
    ready_to_lookup: bool = Field(
        description="True when we have a policy number and should call lookup_policy."
    )
    ready_to_update: bool = Field(
        description="True when we have policy_number, update_field, and update_value and should apply the update."
    )

    # Status
    status: Literal["in_progress", "completed", "failed"] = Field(
        description="'in_progress' still working, 'completed' update applied, 'failed' user declined."
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

INSURANCE_AGENT_SYSTEM_PROMPT = """You are the Policy Update specialist for InsureAssist.

Your goal is to help existing policy holders update their insurance policy. The flow is:
  1. Ask for the policy number if not provided yet.
  2. Look up the policy (you'll receive lookup results).
  3. Ask what field they want to update.
     Allowed update fields: contact_email, contact_phone, coverage_amount, beneficiary, status
  4. Ask for the new value.
  5. Apply the update.

RULES:
- Be friendly and professional.
- If the policy is not found, tell the user and ask them to check the number.
- Only update fields from the allowed list.
- Once the update is successfully applied, set status = 'completed'.
- If the user wants to cancel, set status = 'failed'.

CURRENT STATE will be injected as a system context message below.
"""


# ---------------------------------------------------------------------------
# Insurance Agent class
# ---------------------------------------------------------------------------


class InsuranceAgent:
    """
    Agentic policy update agent.

    On each turn:
      1. Reads InsuranceState to know what's been collected.
      2. Calls LLM (structured output) to extract info and decide next step.
      3. If ready_to_lookup: calls lookup_policy tool.
      4. If ready_to_update: calls update_policy_field tool.
      5. Returns updated state.
    """

    def __init__(self, model: str | None = None, temperature: float = 0.2):
        settings = get_settings()
        model = model or settings.groq_model
        api_key = settings.groq_api_key or None
        base_llm = ChatGroq(model=model, temperature=temperature, api_key=api_key)
        self.llm = base_llm.with_structured_output(InsuranceAgentDecision)
        self._model_name = model

    # ------------------------------------------------------------------
    # LangGraph node entry-point
    # ------------------------------------------------------------------

    async def __call__(self, state: MainAgentState) -> dict:
        """Run one turn of the policy update loop."""
        messages = state.get("messages", [])
        ins_state: InsuranceState = dict(state.get("insurance_state", {}))
        memory: MemoryState = dict(state.get("memory", {}))

        if not ins_state.get("status"):
            ins_state["status"] = "in_progress"
            ins_state["collected_fields"] = []

        llm_messages = self._build_messages(messages, ins_state)

        try:
            decision: InsuranceAgentDecision = await self.llm.ainvoke(llm_messages)
        except Exception as exc:
            logger.error("InsuranceAgent LLM call failed: %s", exc)
            return {
                "messages": [AIMessage(content="I'm having trouble right now. Please try again.")],
                "insurance_state": ins_state,
                "active_agent": "insurance_agent",
            }

        # Merge extracted fields
        ins_state = self._merge_extracted_fields(ins_state, decision)

        final_response = decision.response

        # Lookup policy if we have a number and haven't looked it up yet
        if decision.ready_to_lookup and ins_state.get("policy_number") and not ins_state.get("policy_holder_name"):
            lookup_result = await lookup_policy.ainvoke({"policy_number": ins_state["policy_number"]})
            if lookup_result["found"]:
                ins_state["policy_holder_name"] = lookup_result.get("lead_id", "unknown")
                collected = list(ins_state.get("collected_fields") or [])
                if "policy_number" not in collected:
                    collected.append("policy_number")
                ins_state["collected_fields"] = collected
                final_response = (
                    f"I found your policy (Type: {lookup_result['policy_type']}, "
                    f"Coverage: {lookup_result['coverage_amount']}). "
                    "What would you like to update? You can change: contact email, contact phone, "
                    "coverage amount, or beneficiary."
                )
            else:
                final_response = lookup_result["message"]

        # Apply update if all info is ready
        elif decision.ready_to_update and ins_state.get("policy_number") and ins_state.get("update_field") and ins_state.get("update_value"):
            update_result = await update_policy_field.ainvoke({
                "policy_number": ins_state["policy_number"],
                "field": ins_state["update_field"],
                "new_value": ins_state["update_value"],
            })
            if update_result["success"]:
                ins_state["status"] = "completed"
                final_response = update_result["message"] + " Is there anything else I can help you with?"
            else:
                final_response = update_result["message"]
        else:
            ins_state["status"] = decision.status

        ins_state["last_question"] = final_response
        ai_message = AIMessage(content=final_response)

        logger.info(
            "InsuranceAgent | status=%s | policy=%s | field=%s",
            ins_state.get("status"),
            ins_state.get("policy_number"),
            ins_state.get("update_field"),
        )

        return {
            "messages": [ai_message],
            "insurance_state": ins_state,
            "memory": memory,
            "active_agent": "insurance_agent",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_messages(self, messages: list, ins_state: InsuranceState) -> list:
        llm_messages = [SystemMessage(content=INSURANCE_AGENT_SYSTEM_PROMPT)]

        state_note = (
            f"[Current Insurance State]\n"
            f"Policy number: {ins_state.get('policy_number', 'not provided yet')}\n"
            f"Policy found: {'yes' if ins_state.get('policy_holder_name') else 'not yet'}\n"
            f"Update field: {ins_state.get('update_field', 'not specified yet')}\n"
            f"Update value: {ins_state.get('update_value', 'not specified yet')}\n"
            f"Status: {ins_state.get('status', 'in_progress')}"
        )
        llm_messages.append(SystemMessage(content=state_note))

        recent = messages[-10:] if len(messages) > 10 else messages
        llm_messages.extend(recent)

        return llm_messages

    @staticmethod
    def _merge_extracted_fields(ins_state: InsuranceState, decision: InsuranceAgentDecision) -> InsuranceState:
        collected = list(ins_state.get("collected_fields") or [])

        if decision.extracted_policy_number and not ins_state.get("policy_number"):
            ins_state["policy_number"] = decision.extracted_policy_number
            if "policy_number" not in collected:
                collected.append("policy_number")

        if decision.extracted_update_field and not ins_state.get("update_field"):
            ins_state["update_field"] = decision.extracted_update_field
            if "update_field" not in collected:
                collected.append("update_field")

        if decision.extracted_update_value and not ins_state.get("update_value"):
            ins_state["update_value"] = decision.extracted_update_value
            if "update_value" not in collected:
                collected.append("update_value")

        ins_state["collected_fields"] = collected
        return ins_state


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_insurance_agent(model: str | None = None) -> InsuranceAgent:
    """Create and return a configured InsuranceAgent instance."""
    return InsuranceAgent(model=model)
