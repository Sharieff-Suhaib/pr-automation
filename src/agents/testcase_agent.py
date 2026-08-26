"""Test Case Recommendation Agent.

Answers: *which tests should be run to validate this repair?*

STAGE A: stub.
STAGE F will map the recommended code chunks to the test files that exercise
them (by naming convention and by import graph), so the Testing Agent runs a
focused subset instead of the whole suite — which matters because running a
full suite per candidate patch is the slowest step in the loop.
"""

from __future__ import annotations

from src.agents.state import AgentState, trace_entry


def recommend(state: AgentState) -> AgentState:
    """Recommend tests relevant to the code being repaired."""
    chunks = state.get("relevant_chunks", [])

    # --- STAGE F: resolve real test files for these chunks ---
    tests = [f"[STUB] tests covering {chunk.qualified_name()}" for chunk in chunks]
    # ---------------------------------------------------------

    return AgentState(
        recommended_tests=tests,
        trace=[trace_entry("testcase_agent", f"[STUB] {len(tests)} test target(s)")],
    )
