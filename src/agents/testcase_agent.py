"""Test Case Recommendation Agent.

Answers: *which tests should be run to validate this repair?*

Delegates to recommendation_agent.test_recommender: an LLM prompt over the
issue and relevant code (falling back to a naming-convention heuristic).
STAGE F will additionally map recommended tests to real test files in the repo
(by naming convention and by import graph), so the Testing Agent runs a
focused subset instead of the whole suite — which matters because running a
full suite per candidate patch is the slowest step in the loop.
"""

from __future__ import annotations

from src.agents.recommendation_agent.test_recommender import recommend as recommend_tests
from src.agents.state import AgentState, trace_entry


def recommend(state: AgentState) -> AgentState:
    """Recommend tests relevant to the code being repaired."""
    chunks = state.get("relevant_chunks", [])
    issue = state.get("issue")
    issue_text = issue.as_text() if issue else ""
    code = "\n\n".join(chunk.source for chunk in chunks)

    tests = recommend_tests(issue_text, code=code) if (issue_text or code) else []

    return AgentState(
        recommended_tests=tests,
        trace=[trace_entry("testcase_agent", f"{len(tests)} test target(s)")],
    )
