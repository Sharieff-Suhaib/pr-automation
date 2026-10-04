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
import os
from typing import Annotated, Any, TypedDict

# Patches to try per issue (the repair retry limit); MAX_REPAIR_ATTEMPTS in the brief.
DEFAULT_MAX_ATTEMPTS = int(os.environ.get("AGENT_SWE_MAX_REPAIR_ATTEMPTS", "3"))


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

    # --- Test environment ---
    # The interpreter every test run uses: a cached venv with the repository's
    # dependencies, or this project's interpreter (see environment.py).
    test_env: dict[str, Any]

    # --- Reproduction Agent ---
    # A test written to reproduce the issue, kept only if it fails on the
    # unpatched code (see src/agents/testing_agent/reproduction.py). Its run is
    # also the baseline, so this node fills `baseline_result` too.
    reproduction: dict[str, Any]

    # --- Coding Agent (Member 3) ---
    patch: str
    patch_source: str  # "ollama-function" | "ollama" (model wrote the diff) | "stub" | "none"

    # --- Testing Agent ---
    working_repo: str
    changed_files: list[str]
    patch_status: str  # "APPLIED" | "PATCH_APPLY_FAILED" | "SKIPPED" | ...
    baseline_result: dict[str, Any]  # the suite on the unpatched repo; run once, then reused
    test_result: dict[str, Any]  # the suite on the patched copy
    test_verdict: dict[str, Any]  # TestVerdict.to_dict(): before/after comparison

    # --- Retry loop ---
    # testing_agent -> coding_agent until solved or `max_attempts` patches tried.
    attempt: int  # patches tested so far
    patch_error: str  # why the latest coding_agent pass produced no patch ("" if it did)
    test_feedback: str  # why the latest attempt was rejected; read by coding_agent
    best_attempt: dict[str, Any]  # the best attempt so far; reported if none is solved
    attempts: Annotated[list[dict[str, Any]], operator.add]  # one summary per attempt
    last_attempt: dict[str, Any]  # the full record of the attempt just tested

    # --- Reflection Agent ---
    # Analysis of the latest failed attempt; its feedback is prepended to `test_feedback`.
    reflection: dict[str, Any]
    reflections: Annotated[list[dict[str, Any]], operator.add]  # one per failed attempt

    # --- Control / configuration ---
    top_k: int
    codegen_backend: str  # "ollama" (default) | "stub"
    stub_patch_path: str  # fixture diff, only read when codegen_backend == "stub"
    max_attempts: int  # patches to try before giving up (default 3; stub backend: 1)
    reproduce: bool  # write a reproduction test before the patch (default True)
    isolated_env: bool  # build a venv with the repository's dependencies (default True)
    stub_repro_path: str  # fixture test, only read when codegen_backend == "stub"
    status: str  # "running" | "solved" | "unverified" | "failed"

    # --- Bookkeeping. `operator.add` so each node appends instead of clobbering. ---
    trace: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]


def new_state(
    issue: str,
    repo_url: str,
    top_k: int = 5,
    codegen_backend: str = "ollama",
    stub_patch_path: str = "",
    reproduce: bool = True,
    stub_repro_path: str = "",
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    isolated_env: bool = True,
) -> AgentState:
    """Build the initial state for one repair run."""
    return AgentState(
        issue=issue,
        repo_url=repo_url,
        top_k=top_k,
        codegen_backend=codegen_backend,
        stub_patch_path=stub_patch_path,
        reproduce=reproduce,
        stub_repro_path=stub_repro_path,
        max_attempts=max_attempts,
        isolated_env=isolated_env,
        attempt=0,
        attempts=[],
        reflections=[],
        status="running",
        trace=[],
        errors=[],
    )


def trace_entry(agent: str, summary: str, **details: Any) -> dict[str, Any]:
    """One structured line in the audit trail."""
    return {"agent": agent, "summary": summary, **details}
