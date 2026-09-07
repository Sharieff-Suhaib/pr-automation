"""The LangGraph workflow.

    START
      |
      v
    repository_agent      (Member 1)  issue + repo_url -> relevant code
      |
      v
    recommendation_agent  (Member 2)  -> similar bugs, strategy, tools, tests
      |
      v
    coding_agent          (Member 3)  -> unified diff patch
      |
      v
    testing_agent         (Member 3)  -> apply the patch, run the suite
      |
      v
    END

Each node is a function `AgentState -> partial AgentState`. Nodes never raise:
a failing agent records its message in `errors` and returns empty results, so a
missing dependency (no network, no Ollama) degrades the run instead of ending it.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from orchestrator import adapters
from orchestrator.adapters import AdapterError
from orchestrator.state import AgentState, trace_entry


def repository_agent(state: AgentState) -> AgentState:
    """Member 1: locate the code in the repository that the issue is about."""
    issue = state.get("issue", "")
    repo_url = state.get("repo_url", "")

    try:
        result = adapters.analyze_repository(repo_url, issue, top_k=state.get("top_k", 5))
    except AdapterError as error:
        return AgentState(
            relevant_code=[],
            relevant_files=[],
            errors=[f"repository_agent: {error}"],
            trace=[trace_entry("repository_agent", "failed", error=str(error))],
        )

    return AgentState(
        repo_path=result["repo_path"],
        relevant_code=result["relevant_code"],
        relevant_files=result["relevant_files"],
        language=result["language"],
        trace=[
            trace_entry(
                "repository_agent",
                f"{len(result['relevant_code'])} relevant chunk(s) "
                f"from {result['indexed_chunks']} indexed",
                repo_path=result["repo_path"],
                files=result["relevant_files"],
            )
        ],
    )


def recommendation_agent(state: AgentState) -> AgentState:
    """Member 2: similar bugs -> repair strategy -> tools -> tests."""
    issue = state.get("issue", "")
    code = adapters.format_relevant_code(
        state.get("relevant_code", []), language=state.get("language", "")
    )

    try:
        result = adapters.recommend(issue, code=code, language=state.get("language") or "python")
    except AdapterError as error:
        return AgentState(
            similar_bugs=[],
            strategy="",
            tools=[],
            tests=[],
            errors=[f"recommendation_agent: {error}"],
            trace=[trace_entry("recommendation_agent", "failed", error=str(error))],
        )

    return AgentState(
        similar_bugs=result["similar_bugs"],
        bug_type=result["bug_type"],
        strategy=result["strategy"],
        tools=result["tools"],
        tests=result["tests"],
        trace=[
            trace_entry(
                "recommendation_agent",
                f"bug_type={result['bug_type']}, "
                f"{len(result['similar_bugs'])} similar bug(s), "
                f"{len(result['tests'])} test(s)",
                strategy_source=result["strategy_source"],
            )
        ],
    )


def coding_agent(state: AgentState) -> AgentState:
    """Member 3: turn the issue plus the recommendations into a unified diff."""
    relevant_code = state.get("relevant_code", [])
    if not relevant_code:
        return AgentState(
            patch="",
            patch_source="none",
            errors=["coding_agent: no relevant code to repair."],
            trace=[trace_entry("coding_agent", "skipped: no relevant code")],
        )

    try:
        patch, source = adapters.generate_patch(
            issue=state.get("issue", ""),
            relevant_code=adapters.format_relevant_code(
                relevant_code, language=state.get("language", "")
            ),
            similar_bugs=state.get("similar_bugs", []),
            strategy=state.get("strategy", ""),
            tools=state.get("tools", []),
            tests=state.get("tests", []),
            backend=state.get("codegen_backend", "ollama"),
            stub_patch_path=state.get("stub_patch_path", ""),
            repo_path=state.get("repo_path", ""),
            chunks=relevant_code,
            language=state.get("language", ""),
        )
    except AdapterError as error:
        return AgentState(
            patch="",
            patch_source="none",
            errors=[f"coding_agent: {error}"],
            trace=[trace_entry("coding_agent", "failed", error=str(error))],
        )

    return AgentState(
        patch=patch,
        patch_source=source,
        trace=[
            trace_entry(
                "coding_agent",
                f"{len(patch.splitlines())}-line patch via {source}",
                source=source,
            )
        ],
    )


def testing_agent(state: AgentState) -> AgentState:
    """Member 3: apply the patch to a throwaway copy and run the test suite.

    With no patch to apply, the suite is still run on the unmodified repository
    so the report carries a real baseline instead of an empty placeholder.
    """
    repo_path = state.get("repo_path", "")
    patch = state.get("patch", "")
    language = state.get("language", "")

    if not repo_path:
        return AgentState(
            patch_status="SKIPPED",
            test_result=_skipped("No repository was available to test."),
            status="failed",
            trace=[trace_entry("testing_agent", "skipped: no repository")],
        )

    if not patch.strip():
        try:
            baseline = adapters.run_baseline_tests(repo_path, language=language)
        except AdapterError as error:
            return AgentState(
                patch_status="SKIPPED",
                test_result=_skipped(str(error)),
                status="failed",
                errors=[f"testing_agent: {error}"],
                trace=[trace_entry("testing_agent", "failed", error=str(error))],
            )

        baseline["baseline"] = True
        return AgentState(
            patch_status="SKIPPED",
            test_result=baseline,
            status="failed",
            trace=[
                trace_entry(
                    "testing_agent",
                    f"no patch to apply; baseline suite {baseline['status']}",
                )
            ],
        )

    try:
        result = adapters.apply_and_test(repo_path, patch, language=language)
    except AdapterError as error:
        return AgentState(
            patch_status="SKIPPED",
            test_result=_skipped(str(error)),
            status="failed",
            errors=[f"testing_agent: {error}"],
            trace=[trace_entry("testing_agent", "failed", error=str(error))],
        )

    test_result = result["test_result"]
    solved = result["patch_status"] == "APPLIED" and test_result["status"] == "PASS"

    return AgentState(
        working_repo=result["working_repo"],
        changed_files=result["changed_files"],
        patch_status=result["patch_status"],
        test_result=test_result,
        status="solved" if solved else "failed",
        trace=[
            trace_entry(
                "testing_agent",
                f"patch {result['patch_status']}, tests {test_result['status']} "
                f"({test_result['passed']} passed, {test_result['failed']} failed)",
                working_repo=result["working_repo"],
            )
        ],
    )


def _skipped(reason: str) -> dict:
    return {"status": "SKIPPED", "passed": 0, "failed": 0, "errors": [reason], "output": ""}


def build_graph():
    """Wire the four agents into a compiled LangGraph workflow."""
    builder = StateGraph(AgentState)

    builder.add_node("repository_agent", repository_agent)
    builder.add_node("recommendation_agent", recommendation_agent)
    builder.add_node("coding_agent", coding_agent)
    builder.add_node("testing_agent", testing_agent)

    builder.add_edge(START, "repository_agent")
    builder.add_edge("repository_agent", "recommendation_agent")
    builder.add_edge("recommendation_agent", "coding_agent")
    builder.add_edge("coding_agent", "testing_agent")
    builder.add_edge("testing_agent", END)

    return builder.compile()


_compiled_graph = None


def get_graph():
    """The compiled graph, built once and reused across requests."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
