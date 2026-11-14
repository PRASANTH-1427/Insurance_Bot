"""
State definitions for the Insurance Bot multi-agent system.

Architecture:
  MainAgentState (top-level)
    ├── messages          – full conversation history (LangGraph managed)
    ├── active_agent      – which sub-agent is currently handling the request
    ├── intent            – classified intent: "lead" | "insurance" | "general" | "unknown"
    ├── reason            – internal routing explanation from the MainAgent LLM
    ├── memory            – shared memory across agents (user context, facts)
    ├── lead_state        – state owned by the Lead sub-agent
    └── insurance_state   – state owned by the Insurance sub-agent
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Sub-agent states
# ---------------------------------------------------------------------------


class LeadState(TypedDict, total=False):
    """State owned exclusively by the Lead Creation sub-agent."""

    # Fields being collected
    name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    interest: Optional[str]            # e.g. "auto", "health", "life", "home"

    # Workflow tracking
    collected_fields: list[str]        # which fields have been collected so far
    lead_id: Optional[str]            # DB ID after the lead is saved
    status: Optional[str]             # "in_progress" | "completed" | "failed"
    last_question: Optional[str]       # the last question asked to the user


class InsuranceState(TypedDict, total=False):
    """State owned exclusively by the Insurance Update sub-agent."""

    # Policy lookup
    policy_number: Optional[str]
    policy_holder_name: Optional[str]

    # Fields being updated
    update_field: Optional[str]        # which field the user wants to update
    update_value: Optional[str]        # the new value
    update_type: Optional[str]         # "contact" | "coverage" | "beneficiary" | etc.

    # Workflow tracking
    collected_fields: list[str]
    status: Optional[str]             # "in_progress" | "completed" | "failed"
    last_question: Optional[str]


class MemoryState(TypedDict, total=False):
    """Shared memory visible to all agents — persists across turns."""

    user_name: Optional[str]          # once we learn the user's name
    user_email: Optional[str]
    user_phone: Optional[str]
    off_topic_count: int              # how many times user went off-topic
    session_id: Optional[str]


# ---------------------------------------------------------------------------
# Top-level / Main Agent state
# ---------------------------------------------------------------------------


class MainAgentState(TypedDict):
    """
    The single source-of-truth state for the entire LangGraph.

    ``messages`` uses LangGraph's ``add_messages`` reducer so appending
    to the history is handled automatically.
    """

    # Conversation history – LangGraph merges these automatically
    messages: Annotated[list[BaseMessage], add_messages]

    # Routing & control
    active_agent: Optional[Literal["lead_agent", "insurance_agent", "main_agent"]]
    intent: Optional[Literal["lead", "insurance", "general", "unknown"]]
    next_node: Optional[str]           # explicit override for graph routing

    # Internal routing explanation produced by the MainAgent LLM (not shown to user)
    reason: Optional[str]

    # Sub-agent states (namespaced; only the relevant sub-agent reads/writes its own)
    lead_state: LeadState
    insurance_state: InsuranceState

    # Shared memory
    memory: MemoryState


# Convenience alias kept for backward compatibility
AgentState = MainAgentState
