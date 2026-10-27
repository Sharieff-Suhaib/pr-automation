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
    coding_agent          (Member 3)  -> unified diff patch; the remote model's candidates
      |                                 (one per repair site) are tested as they come,
      |                                 and the first that solves the issue is kept
      |
      v
    testing_agent                     -> run the suite before and after the patch, compare
      |
      +--> reflection_agent  not solved: analyse why (rules; the LLM once the local model takes over)
      |        |
      |        +--> coding_agent  attempts left: retry with the reflection + test feedback
      |        |
      |        v
      |     human_review          max_attempts reached
      v        ^
    human_review                  solved: pause for a person to approve or deny the patch
      |
      +--> commit_and_push -> END approved: commit on a new branch, push it
      v
     END                          denied: nothing is pushed

Each node is a function `AgentState -> partial AgentState`. Nodes never raise:
a failing agent records its message in `errors` and returns empty results, so a
missing dependency (no network, no Ollama) degrades the run instead of ending it.
"""

from __future__ import annotations

import time

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

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


def _test_language(state: AgentState) -> str:
    """The language of the code the coding agent repairs -- its top Python/Java function.

    In a mixed repository the repository-wide language (a vote over the
    retrieved chunks) can differ from it, and the tests must exercise the code
    that is actually patched.
    """
    for chunk in state.get("relevant_code", []):
        if (
            chunk.get("language") in ("python", "java")
            and chunk.get("type") in ("function", "method")
            and not adapters.is_test_file(chunk.get("file", ""))
        ):
            return chunk["language"]
    return state.get("language", "")


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
            language=_test_language(state),
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

    searched = _search_remote(state, attempt)
    if searched["best"]:
        return searched["update"]
    # What the search tried stays in the trace before the local model's attempt.
    searched_update = dict(searched["update"])
    searched_trace = searched_update.pop("trace", [])
    searched_errors = searched_update.pop("errors", [])

    feedback = state.get("test_feedback", "")
    issue = state.get("issue", "")
    reproduction = state.get("reproduction") or {}
    if reproduction.get("status") == "accepted":
        # The test the verdict checks: it names the symptom the fix has to change.
        issue += (
            "\n\nTEST THAT REPRODUCES THE ISSUE (it fails on the current code and must pass after your fix):\n"
            + reproduction["source"]
        )
    try:
        patch, source = adapters.generate_patch(
            issue=issue,
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
            use_remote=False,
        )
    except AdapterError as error:
        return AgentState(
            **searched_update,
            patch="",
            patch_source="none",
            patch_error=str(error),
            errors=[*searched_errors, f"coding_agent (attempt {attempt}): {error}"],
            trace=[*searched_trace, trace_entry("coding_agent", f"attempt {attempt} failed", error=str(error))],
        )

    retry_note = " using the previous attempt's feedback" if feedback else ""
    update = AgentState(
        **searched_update,
        patch=patch,
        patch_source=source,
        patch_error="",
        trace=[
            *searched_trace,
            trace_entry(
                "coding_agent",
                f"attempt {attempt}: {len(patch.splitlines())}-line patch via {source}{retry_note}",
                source=source,
            ),
        ],
    )
    if searched_errors:
        update["errors"] = searched_errors
    return update


def _search_remote(state: AgentState, attempt: int) -> dict:
    """Try the remote model's candidates, each tested at once (generate and validate).

    Returns {"best": candidate or None, "update": partial state}. Without a
    best candidate the update still records what was tried, and the local
    model (which reads the reflection's feedback) writes this attempt's patch.
    """
    site = state.get("remote_site", 0)
    repo_path = state.get("repo_path", "")
    if site is None or not adapters.REPAIR_MODEL_URL or not repo_path or state.get("codegen_backend") == "stub":
        return {"best": None, "update": {"remote_site": None, "prevalidated": {}, "trace": []}}

    language = _test_language(state)
    baseline = state.get("baseline_result") or {}
    try:
        if not baseline:
            baseline = adapters.run_baseline_tests(repo_path, language=language, python=_python(state))
        searched = adapters.search_remote_patch(
            repo_path,
            state.get("relevant_code", []),
            # The issue alone: the reproduction test's code would skew which lines look buggy.
            state.get("issue", ""),
            test=lambda patch: adapters.apply_and_test(
                repo_path, patch, language=language, baseline=baseline,
                reproduction=state.get("reproduction"), python=_python(state),
            ),
            first_site=site,
        )
    except AdapterError as error:
        return {
            "best": None,
            "update": {
                "remote_site": None,
                "prevalidated": {},
                "errors": [f"coding_agent (attempt {attempt}): remote model unusable: {error}"],
                "trace": [trace_entry("coding_agent", "remote model unusable; using the local model", error=str(error))],
            },
        }

    left = searched["next_site"] < searched["sites"]
    trace = [
        trace_entry(
            "coding_agent",
            f"attempt {attempt} candidate {number} ({row['where']}): {row['status']}",
            reason=row["reason"],
        )
        for number, row in enumerate(searched["candidates"], 1)
    ]
    best = searched["best"]
    update: dict = {
        "remote_site": searched["next_site"] if left else None,
        "candidates": [{"attempt": attempt, **row} for row in searched["candidates"]],
        "baseline_result": baseline,
        "prevalidated": {},
        "trace": trace,
    }
    if best:
        update.update(
            patch=best["patch"],
            patch_source="remote-ir4",
            patch_error="",
            prevalidated={"patch": best["patch"], "result": best["result"]},
        )
        update["trace"].append(
            trace_entry(
                "coding_agent",
                f"attempt {attempt}: {best['status']} candidate at {best['where']} "
                f"(sites {site + 1}-{searched['next_site']} of {searched['sites']}) via remote-ir4",
                source="remote-ir4",
            )
        )
    return {"best": best, "update": AgentState(**update)}


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
    language = _test_language(state)

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
            prevalidated = state.get("prevalidated") or {}
            # The coding agent's search already tested this exact patch.
            result = prevalidated["result"] if prevalidated.get("patch") == patch else adapters.apply_and_test(
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

    # An applied patch with no test that ran cannot be judged, so another attempt
    # could not be judged either: hand this one to the reviewer instead. A patch
    # that does not compile ran no test either, but the next one may compile.
    untested = current["patch_status"] == "APPLIED" and adapters.ran_no_tests(current)
    finished = current["status"] == "solved" or untested or not _can_retry(state, number)
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
            # The LLM's guidance is only read by the local model's retry. The remote
            # model's prompt has no room for it, so while repair sites are left, and
            # after the last attempt, the instant rule-based analysis is used.
            use_llm=(
                state.get("codegen_backend", "ollama") != "stub"
                and state.get("status") == "running"
                and state.get("remote_site") is None
            ),
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


def human_review(state: AgentState) -> AgentState:
    """Pause until a person approves or denies the final patch.

    Runs once, after the retry loop has ended. The run is resumed with
    `Command(resume=True)` to approve; any other value denies and ends it.
    A patch that does not apply cannot be pushed, so it is not put to review.
    """
    patch = state.get("patch", "")
    if not patch.strip() or state.get("patch_status") != "APPLIED":
        return AgentState(
            review={"approved": False, "reason": "no patch that applies"},
            trace=[trace_entry("human_review", "skipped: no patch that applies")],
        )

    approved = interrupt(
        {
            "status": state.get("status", ""),
            "patch": patch,
            "verdict": (state.get("test_verdict") or {}).get("reason", ""),
            "root_cause": (state.get("reflection") or {}).get("root_cause", ""),
        }
    ) is True
    summary = "approved" if approved else "denied; workflow terminated, nothing pushed"
    return AgentState(review={"approved": approved}, trace=[trace_entry("human_review", summary)])


def commit_and_push(state: AgentState) -> AgentState:
    """Commit the approved patch on a new branch and push it."""
    try:
        pushed = adapters.commit_and_push(state.get("repo_path", ""), state.get("patch", ""), state.get("issue", ""))
    except AdapterError as error:
        return AgentState(
            errors=[f"commit_and_push: {error}"],
            trace=[trace_entry("commit_and_push", "failed", error=str(error))],
        )
    return AgentState(push=pushed, trace=[trace_entry("commit_and_push", f"pushed {pushed['branch']}", **pushed)])


def route_after_testing(state: AgentState) -> str:
    """Nothing tested ends the run; solved goes to review; any failed attempt goes to reflection."""
    attempt = state.get("last_attempt")
    if not attempt:
        return END
    return "human_review" if attempt.get("status") == "solved" else "reflection_agent"


def route_after_reflection(state: AgentState) -> str:
    """Retry while the testing agent left attempts open; otherwise review (no unbounded loop)."""
    return "coding_agent" if state.get("status") == "running" else "human_review"


def route_after_review(state: AgentState) -> str:
    return "commit_and_push" if (state.get("review") or {}).get("approved") else END


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


def _logged(node):
    """Print when each agent starts and ends: model calls can run for minutes with no other output."""

    def run(state: AgentState) -> AgentState:
        print(f"[{node.__name__}] started", flush=True)
        started = time.monotonic()
        update = node(state)
        summary = "; ".join(entry["summary"] for entry in update.get("trace", []))
        print(f"[{node.__name__}] done in {time.monotonic() - started:.0f}s: {summary}", flush=True)
        return update

    return run


def build_graph():
    """Wire the agents, the review gate and the push into a compiled LangGraph workflow.

    The checkpointer is what lets `human_review` pause a run and resume it later.
    """
    builder = StateGraph(AgentState)

    builder.add_node("repository_agent", _logged(repository_agent))
    builder.add_node("recommendation_agent", _logged(recommendation_agent))
    builder.add_node("environment_agent", _logged(environment_agent))
    builder.add_node("reproduction_agent", _logged(reproduction_agent))
    builder.add_node("coding_agent", _logged(coding_agent))
    builder.add_node("testing_agent", _logged(testing_agent))
    builder.add_node("reflection_agent", _logged(reflection_agent))
    builder.add_node("human_review", _logged(human_review))
    builder.add_node("commit_and_push", _logged(commit_and_push))

    builder.add_edge(START, "repository_agent")
    builder.add_edge("repository_agent", "recommendation_agent")
    builder.add_edge("recommendation_agent", "environment_agent")
    builder.add_edge("environment_agent", "reproduction_agent")
    builder.add_edge("reproduction_agent", "coding_agent")
    builder.add_edge("coding_agent", "testing_agent")
    builder.add_conditional_edges("testing_agent", route_after_testing, ["reflection_agent", "human_review", END])
    builder.add_conditional_edges("reflection_agent", route_after_reflection, ["coding_agent", "human_review"])
    builder.add_conditional_edges("human_review", route_after_review, ["commit_and_push", END])
    builder.add_edge("commit_and_push", END)

    # ponytail: in-memory, so a run awaiting review is lost on restart; SqliteSaver if that matters.
    return builder.compile(checkpointer=InMemorySaver())


_compiled_graph = None


def get_graph():
    """The compiled graph, built once and reused across requests."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
