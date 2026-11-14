"""
LangGraph Workflow
==================

Builds and compiles the full multi-agent StateGraph for InsureAssist.

Graph structure:
  ┌─────────────────────────────────────────────────────────┐
  │                      START                              │
  │                        ↓                                │
  │                   main_agent                            │
  │                 (classify intent)                       │
  │           ┌──────────┼──────────┐                       │
  │        "lead"   "insurance"  "end"                      │
  │           ↓           ↓        ↓                        │
  │       lead_agent  ins_agent   END                       │
  │           ↓           ↓                                 │
  │    (done/in_prog) (done/in_prog)                        │
  │           ↓           ↓                                 │
  │          END         END                                │
  └─────────────────────────────────────────────────────────┘

On each user turn the graph always enters at ``main_agent``.
The main_agent re-entry logic routes back to the active sub-agent without
re-classifying intent, so multi-turn conversations work seamlessly.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from app.agents.insurance_agent import create_insurance_agent
from app.agents.lead_agent import create_lead_agent
from app.agents.main_agent import create_main_agent
from app.config import get_settings
from app.graphs.state import MainAgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing helpers (conditional edge functions)
# ---------------------------------------------------------------------------


def route_from_main(state: MainAgentState) -> str:
    """
    Decide where to go after the main_agent node.

    Reads ``state['next_node']`` which MainAgent sets. Falls back to 'end'.
    """
    next_node = state.get("next_node", "end")
    logger.debug("route_from_main → %s", next_node)
    return next_node


def route_from_lead(state: MainAgentState) -> str:
    """
    After lead_agent runs, decide whether to end the turn or loop.

    The turn always ends here — the next user message will re-enter the graph
    at main_agent, which will route back to lead_agent if status is still
    'in_progress'.
    """
    status = state.get("lead_state", {}).get("status", "in_progress")
    logger.debug("route_from_lead | status=%s → END", status)
    return END


def route_from_insurance(state: MainAgentState) -> str:
    """After insurance_agent runs, end the turn (same pattern as lead)."""
    status = state.get("insurance_state", {}).get("status", "in_progress")
    logger.debug("route_from_insurance | status=%s → END", status)
    return END


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(model: str | None = None):
    """
    Build and compile the InsureAssist multi-agent LangGraph.

    Args:
        model: Groq model name to use for all agents. Defaults to configured GROQ_MODEL.

    Returns:
        A compiled LangGraph runnable.
    """
    model = model or get_settings().groq_model
    # Instantiate agents
    main_agent = create_main_agent(model=model)
    lead_agent = create_lead_agent(model=model)
    insurance_agent = create_insurance_agent(model=model)

    # Create the graph
    builder = StateGraph(MainAgentState)

    # --- Nodes ---
    builder.add_node("main_agent", main_agent)
    builder.add_node("lead_agent", lead_agent)
    builder.add_node("insurance_agent", insurance_agent)

    # --- Entry point ---
    builder.set_entry_point("main_agent")

    # --- Edges from main_agent ---
    builder.add_conditional_edges(
        "main_agent",
        route_from_main,
        {
            "lead_agent": "lead_agent",
            "insurance_agent": "insurance_agent",
            "end": END,
        },
    )

    # --- Edges from lead_agent ---
    builder.add_conditional_edges(
        "lead_agent",
        route_from_lead,
        {END: END},
    )

    # --- Edges from insurance_agent ---
    builder.add_conditional_edges(
        "insurance_agent",
        route_from_insurance,
        {END: END},
    )

    compiled = builder.compile()
    logger.info("InsureAssist graph compiled successfully.")
    return compiled


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

# Lazy-loaded graph instance — built on first access
_graph = None


def get_graph(model: str | None = None):
    """Return the compiled graph, building it on first call."""
    global _graph
    if _graph is None:
        model = model or get_settings().groq_model
        _graph = build_graph(model=model)
    return _graph
