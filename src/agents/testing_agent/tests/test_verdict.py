from src.agents.testing_agent.verdict import compare_runs, not_tested


def run(cases=None, status="PASS", errors=None):
    """A minimal `run_suite` result."""
    return {"status": status, "passed": 0, "failed": 0, "errors": errors or [], "output": "", "cases": cases or {}}


def test_fixed_tests_with_no_regressions_is_solved():
    verdict = compare_runs(
        run({"t::a": "passed", "t::b": "failed", "t::c": "error"}, "FAIL"),
        run({"t::a": "passed", "t::b": "passed", "t::c": "passed"}),
    )

    assert verdict.status == "solved"
    assert verdict.fail_to_pass == ["t::b", "t::c"]
    assert verdict.pass_to_pass == ["t::a"]
    assert verdict.pass_to_fail == []


def test_a_regression_fails_even_when_the_bug_is_fixed():
    verdict = compare_runs(
        run({"t::a": "passed", "t::b": "failed"}, "FAIL"),
        run({"t::a": "failed", "t::b": "passed"}, "FAIL"),
    )

    assert verdict.status == "failed"
    assert verdict.fail_to_pass == ["t::b"]
    assert verdict.pass_to_fail == ["t::a"]
    assert "broke 1" in verdict.reason


def test_a_test_missing_after_the_patch_counts_as_broken():
    verdict = compare_runs(
        run({"t::a": "passed", "t::b": "failed"}, "FAIL"),
        run({"t::b": "passed"}),
    )

    assert verdict.status == "failed"
    assert verdict.pass_to_fail == ["t::a"]


def test_all_passing_before_and_after_is_unverified():
    verdict = compare_runs(run({"t::a": "passed"}), run({"t::a": "passed"}))

    assert verdict.status == "unverified"
    assert "no test shows the fix" in verdict.reason


def test_failures_left_untouched_are_unverified():
    verdict = compare_runs(
        run({"t::a": "passed", "t::b": "failed"}, "FAIL"),
        run({"t::a": "passed", "t::b": "failed"}, "FAIL"),
    )

    assert verdict.status == "unverified"
    assert verdict.fail_to_fail == ["t::b"]


def test_added_tests_are_reported_and_a_failing_one_blocks_solved():
    baseline = run({"t::a": "failed"}, "FAIL")

    passing = compare_runs(baseline, run({"t::a": "passed", "t::new": "passed"}))
    failing = compare_runs(baseline, run({"t::a": "passed", "t::new": "failed"}, "FAIL"))

    assert passing.status == "solved"
    assert passing.added == {"t::new": "passed"}
    assert failing.status == "failed"
    assert "added 1 failing" in failing.reason


def test_skipped_tests_are_ignored():
    verdict = compare_runs(
        run({"t::a": "skipped", "t::b": "passed"}),
        run({"t::a": "failed", "t::b": "skipped"}, "FAIL"),
    )

    assert verdict.status == "unverified"
    assert verdict.pass_to_fail == []


def test_without_per_test_results_the_overall_status_is_compared():
    solved = compare_runs(run(status="FAIL"), run(status="PASS"))
    unverified = compare_runs(run(status="PASS"), run(status="PASS"))
    failed = compare_runs(run(status="PASS"), run(status="FAIL", errors=["cargo: 1 failed"]))

    assert (solved.status, solved.per_test) == ("solved", False)
    assert unverified.status == "unverified"
    assert failed.status == "failed"
    assert "cargo: 1 failed" in failed.reason


def test_a_still_failing_reproduction_test_blocks_solved():
    verdict = compare_runs(
        run({"t::a": "failed", "repro::r": "failed"}, "FAIL"),
        run({"t::a": "passed", "repro::r": "failed"}, "FAIL"),
        reproduction_tests=["repro::r"],
    )

    assert verdict.status == "unverified"
    assert verdict.reproduction == {"repro::r": "failed"}
    assert "reproduction test still fails" in verdict.reason


def test_not_tested_is_a_failed_verdict():
    verdict = not_tested("No patch was generated.")

    assert verdict.status == "failed"
    assert verdict.reason == "No patch was generated."
