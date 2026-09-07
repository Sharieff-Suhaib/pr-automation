"""Shared state for the LangGraph workflow.

Every node is a plain function `AgentState -> partial AgentState`. LangGraph
merges the returned dict into the running state, so a node returns only the keys
it actually produced. That keeps each node independently testable: build a state
by hand, call the node, assert on what comes back.

Keys are grouped by the agent that fills them, which is also the order the graph
runs them in.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    """The blackboard every node reads from and writes to.

    `total=False` because early nodes have not produced the later keys yet.
    """

    # --- Input (from the Manager Agent) ---
    issue: str
    repo_url: str

    # --- Repository Agent (Member 1) ---
    repo_path: str
    language: str
    relevant_files: list[str]
    relevant_code: list[dict[str, Any]]

    # --- Recommendation Agent (Member 2) ---
    similar_bugs: list[dict[str, Any]]
    bug_type: str
    strategy: str
    tools: list[str]
    tests: list[str]

    # --- Coding Agent (Member 3) ---
    patch: str
    patch_source: str  # "ollama" | "stub" | "none"

    # --- Testing Agent (Member 3) ---
    working_repo: str
    changed_files: list[str]
    patch_status: str  # "APPLIED" | "PATCH_APPLY_FAILED" | "SKIPPED" | ...
    test_result: dict[str, Any]

    # --- Control / configuration ---
    top_k: int
    max_repair_files: int  # 0 = the coding agent's default
    codegen_backend: str  # "ollama" (default) | "stub"
    stub_patch_path: str  # fixture diff, only read when codegen_backend == "stub"
    status: str  # "running" | "solved" | "failed"

    # --- Bookkeeping. `operator.add` so each node appends instead of clobbering. ---
    trace: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]


def new_state(
    issue: str,
    repo_url: str,
    top_k: int = 5,
    max_repair_files: int = 0,
    codegen_backend: str = "ollama",
    stub_patch_path: str = "",
) -> AgentState:
    """Build the initial state for one repair run."""
    return AgentState(
        issue=issue,
        repo_url=repo_url,
        top_k=top_k,
        max_repair_files=max_repair_files,
        codegen_backend=codegen_backend,
        stub_patch_path=stub_patch_path,
        status="running",
        trace=[],
        errors=[],
    )


def trace_entry(agent: str, summary: str, **details: Any) -> dict[str, Any]:
    """One structured line in the audit trail."""
    return {"agent": agent, "summary": summary, **details}
