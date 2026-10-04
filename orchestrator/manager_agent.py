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
from orchestrator.state import DEFAULT_MAX_ATTEMPTS, AgentState, new_state


class ManagerAgent:
    """Starts one repair run per issue and reports the result."""

    def __init__(
        self,
        top_k: int = 5,
        codegen_backend: str = "ollama",
        stub_patch_path: str = "",
        reproduce: bool = True,
        stub_repro_path: str = "",
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        isolated_env: bool = True,
    ):
        self.top_k = top_k
        self.codegen_backend = codegen_backend
        self.stub_patch_path = stub_patch_path
        self.reproduce = reproduce
        self.stub_repro_path = stub_repro_path
        self.max_attempts = max_attempts
        self.isolated_env = isolated_env
        self.graph = get_graph()

    def solve(self, repo_url: str, issue: str) -> dict[str, Any]:
        """Run the full workflow for one issue and return the report."""
        initial = new_state(
            issue=issue,
            repo_url=repo_url,
            top_k=self.top_k,
            codegen_backend=self.codegen_backend,
            stub_patch_path=self.stub_patch_path,
            reproduce=self.reproduce,
            stub_repro_path=self.stub_repro_path,
            max_attempts=self.max_attempts,
            isolated_env=self.isolated_env,
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
        "test_env": state.get("test_env") or {},
        "reproduction": state.get("reproduction") or {},
        "patch": state.get("patch", ""),
        "patch_source": state.get("patch_source", "none"),
        "patch_status": state.get("patch_status", "SKIPPED"),
        "changed_files": state.get("changed_files", []),
        "working_repo": state.get("working_repo", ""),
        # `test_status` is the plain PASS/FAIL/SKIPPED string of the patched run;
        # `test_result` keeps the counts and runner output behind it.
        "test_status": test_result.get("status", "SKIPPED"),
        "test_result": test_result,
        # The same suite on the unpatched repository, and the before/after
        # comparison that `status` is decided from.
        "baseline_result": state.get("baseline_result") or {},
        "test_verdict": state.get("test_verdict") or {},
        # One row per patch the retry loop tested; the fields above show the
        # solved attempt, or the best one when none was solved.
        "attempts": state.get("attempts", []),
        # The Reflection Agent's analysis of the last failed attempt, and of every one.
        "reflection": state.get("reflection") or {},
        "reflections": state.get("reflections", []),
        "errors": state.get("errors", []),
        "trace": state.get("trace", []),
    }


def solve_issue(
    repo_url: str,
    issue: str,
    top_k: int = 5,
    codegen_backend: str = "ollama",
    stub_patch_path: str = "",
    reproduce: bool = True,
    stub_repro_path: str = "",
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    isolated_env: bool = True,
) -> dict[str, Any]:
    """One-call convenience wrapper around `ManagerAgent.solve`."""
    manager = ManagerAgent(
        top_k=top_k,
        codegen_backend=codegen_backend,
        stub_patch_path=stub_patch_path,
        reproduce=reproduce,
        stub_repro_path=stub_repro_path,
        max_attempts=max_attempts,
        isolated_env=isolated_env,
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
    parser.add_argument("--stub-repro", default="", help="Fixture test used when --codegen-backend stub")
    parser.add_argument(
        "--no-repro-test", action="store_true", help="Skip writing a test that reproduces the issue"
    )
    parser.add_argument(
        "--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
        help=f"Patches to try (default: {DEFAULT_MAX_ATTEMPTS}, env AGENT_SWE_MAX_REPAIR_ATTEMPTS)",
    )
    parser.add_argument(
        "--no-isolated-env", action="store_true", help="Run tests with this interpreter, not a per-repo venv"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = solve_issue(
        repo_url=args.repo_url,
        issue=args.issue,
        top_k=args.top_k,
        codegen_backend=args.codegen_backend,
        stub_patch_path=args.stub_patch,
        reproduce=not args.no_repro_test,
        stub_repro_path=args.stub_repro,
        max_attempts=args.max_attempts,
        isolated_env=not args.no_isolated_env,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "solved" else 1


if __name__ == "__main__":
    raise SystemExit(main())
