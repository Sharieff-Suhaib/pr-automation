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
    environment_agent                 -> a cached venv with the repo's dependencies; cleanup
      |
      v
    reproduction_agent                -> a test that fails on the unpatched code; the baseline run
      |
      v
    coding_agent          (Member 3)  -> unified diff patch
      |
      v
    testing_agent                     -> run the suite before and after the patch, compare
      |
      +--> reflection_agent  not solved: analyse why (LLM, rule-based fallback)
      |        |
      |        +--> coding_agent  attempts left: retry with the reflection + test feedback
      |        |
      |        v
      |       END                 max_attempts reached
      v
    END                           solved

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
    code = adapters.format_relevant_code(state.get("relevant_code", []))

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


def environment_agent(state: AgentState) -> AgentState:
    """Prepare the interpreter the repository's tests run with, and clean up old runs.

    A repository that declares dependencies gets a cached virtual environment
    with them; one that does not (or when building fails) uses this project's
    interpreter, as before. Old workspaces and environments beyond the keep
    limits are deleted first.
    """
    repo_path = state.get("repo_path", "")
    if not repo_path:
        return AgentState(trace=[trace_entry("environment_agent", "skipped: no repository")])

    try:
        env = adapters.prepare_test_environment(repo_path, enabled=state.get("isolated_env", True))
    except AdapterError as error:
        return AgentState(
            errors=[f"environment_agent: {error}"],
            trace=[trace_entry("environment_agent", "failed", error=str(error))],
        )

    removed = env.pop("removed")
    summary = f"{env['status']}: {env['detail']}"
    if removed:
        summary += f" Removed {len(removed)} old director{'y' if len(removed) == 1 else 'ies'}."
    update = AgentState(test_env=env, trace=[trace_entry("environment_agent", summary)])
    if env["status"] == "failed":
        update["errors"] = [f"environment_agent: {env['detail']}"]
    return update


def _python(state: AgentState) -> str | None:
    """The interpreter chosen by environment_agent, if any."""
    return (state.get("test_env") or {}).get("python")


def reproduction_agent(state: AgentState) -> AgentState:
    """Write a test that reproduces the issue, and run the baseline with it.

    The test is kept only when it fails on the unpatched code, so it is real
    evidence of the bug; the testing agent then reuses this node's baseline.
    Disabled (`reproduce=False`) or without a repository, the node does nothing
    and the testing agent runs a plain baseline itself.
    """
    repo_path = state.get("repo_path", "")
    if not repo_path or not state.get("reproduce", True):
        reason = "disabled" if repo_path else "no repository"
        return AgentState(trace=[trace_entry("reproduction_agent", f"skipped: {reason}")])

    backend = state.get("codegen_backend", "ollama")
    if backend == "stub" and not state.get("stub_repro_path"):
        return AgentState(trace=[trace_entry("reproduction_agent", "skipped: no stub test given")])

    try:
        reproduction, baseline = adapters.prepare_reproduction(
            repo_path,
            state.get("issue", ""),
            state.get("relevant_code", []),
            language=state.get("language", ""),
            backend=backend,
            stub_repro_path=state.get("stub_repro_path", ""),
            python=_python(state),
        )
    except AdapterError as error:
        return AgentState(
            errors=[f"reproduction_agent: {error}"],
            trace=[trace_entry("reproduction_agent", "failed", error=str(error))],
        )

    summary = f"{reproduction['status']} after {reproduction['attempts']} attempt(s)"
    if reproduction["status"] == "accepted":
        summary += f"; {len(reproduction['tests'])} test(s) fail on the unpatched code"
    return AgentState(
        reproduction=reproduction,
        baseline_result=baseline,
        trace=[trace_entry("reproduction_agent", summary, reason=reproduction["reason"])],
    )


def coding_agent(state: AgentState) -> AgentState:
    """Member 3: turn the issue plus the recommendations into a unified diff.

    On a retry, `test_feedback` (why the previous patch was rejected) is part of
    the prompt; the original code is repaired again from scratch.
    """
    relevant_code = state.get("relevant_code", [])
    attempt = state.get("attempt", 0) + 1
    if not relevant_code:
        return AgentState(
            patch="",
            patch_source="none",
            patch_error="No relevant code to repair.",
            errors=["coding_agent: no relevant code to repair."],
            trace=[trace_entry("coding_agent", "skipped: no relevant code")],
        )

    feedback = state.get("test_feedback", "")
    try:
        patch, source = adapters.generate_patch(
            issue=state.get("issue", ""),
            relevant_code=adapters.format_relevant_code(relevant_code),
            similar_bugs=state.get("similar_bugs", []),
            strategy=state.get("strategy", ""),
            tools=state.get("tools", []),
            tests=state.get("tests", []),
            backend=state.get("codegen_backend", "ollama"),
            stub_patch_path=state.get("stub_patch_path", ""),
            repo_path=state.get("repo_path", ""),
            relevant_chunks=relevant_code,
            feedback=feedback,
            attempt=attempt,
        )
    except AdapterError as error:
        return AgentState(
            patch="",
            patch_source="none",
            patch_error=str(error),
            errors=[f"coding_agent (attempt {attempt}): {error}"],
            trace=[trace_entry("coding_agent", f"attempt {attempt} failed", error=str(error))],
        )

    retry_note = " using the previous attempt's feedback" if feedback else ""
    return AgentState(
        patch=patch,
        patch_source=source,
        patch_error="",
        trace=[
            trace_entry(
                "coding_agent",
                f"attempt {attempt}: {len(patch.splitlines())}-line patch via {source}{retry_note}",
                source=source,
            )
        ],
    )


def testing_agent(state: AgentState) -> AgentState:
    """Run the suite before and after the patch and judge the difference.

    `status` comes from the before/after verdict, not from the patched run alone:
    "solved" needs at least one test to go from failing to passing with none
    breaking. The baseline is taken from the state when a previous pass already
    ran it, so retries against the same repository do not pay for it twice.
    With no patch, the baseline still runs so the report carries real numbers.

    Every pass is one attempt of the retry loop: it is recorded in `attempts`,
    compared with `best_attempt`, and -- unless solved -- explained in
    `test_feedback` for the coding agent. When the loop ends without a solved
    attempt, the best attempt is what the report shows.
    """
    repo_path = state.get("repo_path", "")
    patch = state.get("patch", "")
    language = state.get("language", "")

    if not repo_path:
        return AgentState(
            patch_status="SKIPPED",
            test_result=_skipped("No repository was available to test."),
            test_verdict=_not_tested("No repository was available to test."),
            status="failed",
            trace=[trace_entry("testing_agent", "skipped: no repository")],
        )

    outcome: dict = {"working_repo": "", "changed_files": [], "errors": []}
    try:
        baseline = state.get("baseline_result") or adapters.run_baseline_tests(
            repo_path, language=language, python=_python(state)
        )
        if not patch.strip():
            reason = state.get("patch_error") or "No patch was generated."
            outcome.update(
                patch_status="SKIPPED",
                test_result=_skipped("No patch was generated."),
                test_verdict=_not_tested(f"No patch was generated: {reason}"),
                summary=f"no patch to apply; baseline {_counts(baseline)}",
            )
        else:
            result = adapters.apply_and_test(
                repo_path,
                patch,
                language=language,
                baseline=baseline,
                reproduction=state.get("reproduction"),
                python=_python(state),
            )
            verdict = result["test_verdict"]
            outcome.update(
                working_repo=result["working_repo"],
                changed_files=result["changed_files"],
                patch_status=result["patch_status"],
                test_result=result["test_result"],
                test_verdict=verdict,
                summary=(
                    f"patch {result['patch_status']}; baseline {_counts(baseline)}, "
                    f"patched {_counts(result['test_result'])}; {len(verdict['fail_to_pass'])} fixed, "
                    f"{len(verdict['pass_to_fail'])} broken -> {verdict['status']}"
                ),
            )
    except AdapterError as error:
        baseline = state.get("baseline_result") or {}
        outcome.update(
            patch_status="SKIPPED",
            test_result=_skipped(str(error)),
            test_verdict=_not_tested(str(error)),
            summary="failed",
            errors=[f"testing_agent: {error}"],
        )

    return _record_attempt(state, outcome, baseline)


def _record_attempt(state: AgentState, outcome: dict, baseline: dict) -> AgentState:
    """Bookkeeping shared by every testing pass: history, best attempt, feedback."""
    number = state.get("attempt", 0) + 1
    current = {
        "attempt": number,
        "patch": state.get("patch", ""),
        "patch_source": state.get("patch_source", "none"),
        "patch_error": state.get("patch_error", ""),
        "working_repo": outcome["working_repo"],
        "changed_files": outcome["changed_files"],
        "patch_status": outcome["patch_status"],
        "test_result": outcome["test_result"],
        "test_verdict": outcome["test_verdict"],
        "status": outcome["test_verdict"]["status"],
    }

    best = state.get("best_attempt")
    if adapters.is_better_attempt(current, best):
        best = current

    finished = current["status"] == "solved" or not _can_retry(state, number)
    shown = best if finished else current
    earlier_patches = [row.get("patch", "") for row in state.get("attempts", [])]
    feedback = "" if current["status"] == "solved" else adapters.attempt_feedback(current, earlier_patches)
    verdict = current["test_verdict"]

    update = AgentState(
        attempt=number,
        last_attempt=current,
        best_attempt=best,
        test_feedback=feedback,
        baseline_result=baseline,
        # The report and the routing read these, so they hold the best attempt once
        # the loop is over and the current one while it is still going.
        patch=shown["patch"],
        patch_source=shown["patch_source"],
        working_repo=shown["working_repo"],
        changed_files=shown["changed_files"],
        patch_status=shown["patch_status"],
        test_result=shown["test_result"],
        test_verdict=shown["test_verdict"],
        status=shown["status"] if finished else "running",
        attempts=[
            {
                "attempt": number,
                "patch": current["patch"],
                "patch_source": current["patch_source"],
                "patch_status": current["patch_status"],
                "status": current["status"],
                "reason": verdict["reason"],
                "fixed": len(verdict["fail_to_pass"]),
                "broken": len(verdict["pass_to_fail"]),
            }
        ],
        trace=[
            trace_entry(
                "testing_agent",
                f"attempt {number}: {outcome['summary']}",
                working_repo=current["working_repo"],
                reason=verdict["reason"],
            )
        ],
    )
    if outcome["errors"]:
        update["errors"] = outcome["errors"]
    if finished and shown is not current:
        update["trace"].append(
            trace_entry("testing_agent", f"no attempt solved the issue; reporting attempt {shown['attempt']}")
        )
    return update


def _can_retry(state: AgentState, attempts_made: int) -> bool:
    """Whether another coding pass could change anything."""
    if attempts_made >= state.get("max_attempts", 3):
        return False
    # Replaying the same fixture diff cannot produce a different patch.
    if state.get("codegen_backend", "ollama") == "stub":
        return False
    return bool(state.get("relevant_code"))


def reflection_agent(state: AgentState) -> AgentState:
    """Analyse the attempt that just failed; its guidance leads the coding agent's next prompt.

    Runs after every failed attempt, including the last: then the status is
    `max_retries` and the analysis explains the final failure in the report.
    It never edits code. With the `stub` backend the analysis is rule-based,
    so offline runs need no model.
    """
    attempt = state.get("last_attempt") or {}
    try:
        reflection = adapters.reflect(
            issue=state.get("issue", ""),
            relevant_code=adapters.format_relevant_code(state.get("relevant_code", [])),
            attempt=attempt,
            strategy=state.get("strategy", ""),
            previous_attempts=state.get("attempts", []),
            max_attempts=state.get("max_attempts", 3),
            use_llm=state.get("codegen_backend", "ollama") != "stub",
        )
    except AdapterError as error:
        # Without a reflection the retry still has the testing agent's feedback.
        return AgentState(
            errors=[f"reflection_agent: {error}"],
            trace=[trace_entry("reflection_agent", "failed", error=str(error))],
        )

    if state.get("status") != "running":
        # The testing agent already ended the loop; keep its decision.
        reflection["status"] = "max_retries" if reflection["status"] == "retry" else reflection["status"]
    feedback = reflection.pop("feedback")
    update = AgentState(
        reflection=reflection,
        reflections=[reflection],
        trace=[
            trace_entry(
                "reflection_agent",
                f"attempt {attempt.get('attempt')}: {reflection['failure_type']} -> {reflection['status']} "
                f"({len(reflection['failed_tests'])} failed test(s), {reflection['source']} analysis)",
                root_cause=reflection["root_cause"],
            )
        ],
    )
    if state.get("status") == "running":
        update["test_feedback"] = f"{feedback}\n\n{state.get('test_feedback', '')}".strip()
    return update


def route_after_testing(state: AgentState) -> str:
    """Solved (or nothing was tested) ends the run; any failed attempt goes to reflection."""
    attempt = state.get("last_attempt")
    if not attempt or attempt.get("status") == "solved":
        return END
    return "reflection_agent"


def route_after_reflection(state: AgentState) -> str:
    """Retry while the testing agent left attempts open; otherwise stop (no unbounded loop)."""
    return "coding_agent" if state.get("status") == "running" else END


def _counts(result: dict) -> str:
    return f"{result['status']} ({result['passed']} passed, {result['failed']} failed)"


def _skipped(reason: str) -> dict:
    return {"status": "SKIPPED", "passed": 0, "failed": 0, "errors": [reason], "output": "", "cases": {}}


def _not_tested(reason: str) -> dict:
    """Same shape as `TestVerdict.to_dict()`, without importing the testing agent here."""
    return {
        "status": "failed",
        "reason": reason,
        "fail_to_pass": [],
        "pass_to_pass": [],
        "pass_to_fail": [],
        "fail_to_fail": [],
        "added": {},
        "reproduction": {},
        "stage": "none",
        "per_test": False,
    }


def build_graph():
    """Wire the seven agents into a compiled LangGraph workflow."""
    builder = StateGraph(AgentState)

    builder.add_node("repository_agent", repository_agent)
    builder.add_node("recommendation_agent", recommendation_agent)
    builder.add_node("environment_agent", environment_agent)
    builder.add_node("reproduction_agent", reproduction_agent)
    builder.add_node("coding_agent", coding_agent)
    builder.add_node("testing_agent", testing_agent)
    builder.add_node("reflection_agent", reflection_agent)

    builder.add_edge(START, "repository_agent")
    builder.add_edge("repository_agent", "recommendation_agent")
    builder.add_edge("recommendation_agent", "environment_agent")
    builder.add_edge("environment_agent", "reproduction_agent")
    builder.add_edge("reproduction_agent", "coding_agent")
    builder.add_edge("coding_agent", "testing_agent")
    builder.add_conditional_edges("testing_agent", route_after_testing, ["reflection_agent", END])
    builder.add_conditional_edges("reflection_agent", route_after_reflection, ["coding_agent", END])

    return builder.compile()


_compiled_graph = None


def get_graph():
    """The compiled graph, built once and reused across requests."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
