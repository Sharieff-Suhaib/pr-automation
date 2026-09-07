"""Manager Agent -- the entry point into the workflow.

Receives a GitHub issue and a repository URL, starts the LangGraph run, and
shapes the final state into the JSON the API returns.

CLI:
    python -m orchestrator.manager_agent --repo-url <url> --issue "<text>"
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from orchestrator.graph import get_graph
from orchestrator.state import AgentState, new_state


class ManagerAgent:
    """Starts one repair run per issue and reports the result."""

    def __init__(self, top_k: int = 5, codegen_backend: str = "ollama", stub_patch_path: str = ""):
        self.top_k = top_k
        self.codegen_backend = codegen_backend
        self.stub_patch_path = stub_patch_path
        self.graph = get_graph()

    def solve(self, repo_url: str, issue: str) -> dict[str, Any]:
        """Run the full workflow for one issue and return the report."""
        initial = new_state(
            issue=issue,
            repo_url=repo_url,
            top_k=self.top_k,
            codegen_backend=self.codegen_backend,
            stub_patch_path=self.stub_patch_path,
        )
        final: AgentState = self.graph.invoke(initial)
        return build_report(final)


def build_report(state: AgentState) -> dict[str, Any]:
    """Flatten the final graph state into the response the API contract promises."""
    test_result = state.get("test_result") or {}

    return {
        "status": state.get("status", "failed"),
        "repo_url": state.get("repo_url", ""),
        "issue": state.get("issue", ""),
        "repo_path": state.get("repo_path", ""),
        "language": state.get("language", ""),
        "relevant_files": state.get("relevant_files", []),
        "relevant_code": [
            {
                "file": chunk["file"],
                "name": chunk["name"],
                "type": chunk["type"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "distance": chunk.get("distance"),
                "code": chunk["code"],
            }
            for chunk in state.get("relevant_code", [])
        ],
        "similar_bugs": state.get("similar_bugs", []),
        "bug_type": state.get("bug_type", ""),
        "strategy": state.get("strategy", ""),
        "tools": state.get("tools", []),
        "tests": state.get("tests", []),
        "patch": state.get("patch", ""),
        "patch_source": state.get("patch_source", "none"),
        "patch_status": state.get("patch_status", "SKIPPED"),
        "changed_files": state.get("changed_files", []),
        "working_repo": state.get("working_repo", ""),
        # `test_status` is the plain PASS/FAIL/SKIPPED string; `test_result` keeps
        # the counts and runner output behind it.
        "test_status": test_result.get("status", "SKIPPED"),
        "test_result": test_result,
        "errors": state.get("errors", []),
        "trace": state.get("trace", []),
    }


def solve_issue(
    repo_url: str,
    issue: str,
    top_k: int = 5,
    codegen_backend: str = "ollama",
    stub_patch_path: str = "",
) -> dict[str, Any]:
    """One-call convenience wrapper around `ManagerAgent.solve`."""
    manager = ManagerAgent(
        top_k=top_k,
        codegen_backend=codegen_backend,
        stub_patch_path=stub_patch_path,
    )
    return manager.solve(repo_url, issue)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-agent repair workflow on one issue")
    parser.add_argument("--repo-url", required=True, help="Repository URL, or a local directory path")
    parser.add_argument("--issue", required=True, help="The GitHub issue text")
    parser.add_argument("--top-k", type=int, default=5, help="Code chunks to retrieve (default: 5)")
    parser.add_argument(
        "--codegen-backend",
        default="ollama",
        choices=["ollama", "stub"],
        help="'stub' replays a fixture diff instead of calling the model",
    )
    parser.add_argument("--stub-patch", default="", help="Fixture diff used when --codegen-backend stub")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = solve_issue(
        repo_url=args.repo_url,
        issue=args.issue,
        top_k=args.top_k,
        codegen_backend=args.codegen_backend,
        stub_patch_path=args.stub_patch,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "solved" else 1


if __name__ == "__main__":
    raise SystemExit(main())
