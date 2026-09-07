"""Orchestrator (Member 4).

Wires the three agents built by Members 1-3 into one LangGraph workflow:

    START -> repository_agent -> recommendation_agent -> coding_agent -> testing_agent -> END

Public entry points:

    from orchestrator import solve_issue          # one call, one repair run
    from orchestrator.graph import build_graph    # the compiled LangGraph
    from orchestrator.api import app              # FastAPI app exposing POST /solve
"""

from orchestrator.manager_agent import ManagerAgent, solve_issue
from orchestrator.state import AgentState, new_state

__all__ = ["AgentState", "ManagerAgent", "new_state", "solve_issue"]
