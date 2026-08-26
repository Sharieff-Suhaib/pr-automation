"""Tool Recommendation Agent.

Answers: *which development tools does this repair need?*

The brief is explicit that the agent should avoid recommending unnecessary
tools, so recommendations are driven by evidence in the repository (a pytest
config, a ruff section in pyproject) rather than a fixed list.
"""

from __future__ import annotations

import os

from src.agents.state import AgentState, trace_entry

# Marker files -> the tool they imply.
_EVIDENCE: list[tuple[str, tuple[str, ...]]] = [
    ("pytest", ("pytest.ini", "conftest.py", "tox.ini", "setup.cfg", "pyproject.toml")),
    ("ruff", ("ruff.toml", ".ruff.toml")),
    ("git", (".git",)),
]


def recommend(state: AgentState) -> AgentState:
    """Recommend tools based on what the repository actually contains."""
    repo_path = state.get("repo_path", "")
    tools: list[str] = []

    if repo_path and os.path.isdir(repo_path):
        for tool, markers in _EVIDENCE:
            if any(os.path.exists(os.path.join(repo_path, marker)) for marker in markers):
                tools.append(tool)

    # Python repairs always need a test runner and an AST parser for patching.
    for default in ("pytest", "ast"):
        if default not in tools:
            tools.append(default)

    return AgentState(
        recommended_tools=tools,
        trace=[trace_entry("tool_agent", f"tools={tools}")],
    )
