"""Decide whether a patch fixed the bug by comparing two runs of the suite.

A passing suite after the patch is not enough on its own: it also passes when
no test covers the bug at all. So the suite is run on the unpatched repository
(the baseline) and on the patched copy, and every test is sorted by how its
outcome changed:

    fail_to_pass  failed before, passes now     -> evidence the patch fixed something
    pass_to_pass  passed before and now         -> nothing broken
    pass_to_fail  passed before, not any more   -> a regression
    fail_to_fail  failed before and still fails -> untouched (possibly unrelated)

The verdict:

    solved      at least one fail_to_pass, and nothing broken
    unverified  nothing broken, but no test shows the fix either
    failed      something broke, or the patch could not be tested

When a reproduction test for the issue took part (see `reproduction.py`), it is
the most direct evidence there is, so "solved" also requires every one of its
tests to pass after the patch -- a patch that fixes half the issue is not done.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.agents.testing_agent.junit import ERROR, FAILED, PASSED, SKIPPED

SOLVED = "solved"
UNVERIFIED = "unverified"
NOT_SOLVED = "failed"

# The outcome recorded for a baseline test that is absent after the patch, e.g.
# because its module no longer imports. Absent means "no longer passing".
MISSING = "missing"


@dataclass
class TestVerdict:
    """The outcome of comparing a baseline run with a patched run."""

    __test__ = False  # not a pytest test class, despite the name

    status: str
    reason: str
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    pass_to_fail: list[str] = field(default_factory=list)
    fail_to_fail: list[str] = field(default_factory=list)
    # Tests that exist only after the patch, mapped to their outcome.
    added: dict[str, str] = field(default_factory=dict)
    # The issue's reproduction tests, mapped to their outcome after the patch.
    reproduction: dict[str, str] = field(default_factory=dict)
    # The test stage the verdict comes from: syntax, reproduction, related or full.
    stage: str = "full"
    # False when no per-test results were available and the verdict fell back
    # to comparing the overall PASS/FAIL of each run.
    per_test: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def not_tested(reason: str, stage: str = "none") -> TestVerdict:
    """The verdict when the patched code never reached the test runner."""
    return TestVerdict(status=NOT_SOLVED, reason=reason, per_test=False, stage=stage)


def compare_runs(
    baseline: dict[str, Any],
    patched: dict[str, Any],
    reproduction_tests: list[str] | None = None,
) -> TestVerdict:
    """Compare two `run_suite` results and decide whether the patch fixed the bug.

    ``reproduction_tests`` are the ids of the tests written to reproduce the
    issue; each one also appears in the ordinary before/after lists.
    """
    before: dict[str, str] = baseline.get("cases") or {}
    after: dict[str, str] = patched.get("cases") or {}

    if not before and not after:
        return _compare_overall(baseline, patched)

    verdict = TestVerdict(status=NOT_SOLVED, reason="")
    verdict.reproduction = {test_id: after.get(test_id, MISSING) for test_id in reproduction_tests or []}
    still_failing = [t for t, outcome in verdict.reproduction.items() if outcome != PASSED]
    reproduced_fix = bool(verdict.reproduction) and not still_failing
    for test_id, outcome_before in before.items():
        outcome_after = after.get(test_id, MISSING)
        # A skip says nothing about the bug either way.
        if SKIPPED in (outcome_before, outcome_after):
            continue

        passed_before = outcome_before == PASSED
        passed_after = outcome_after == PASSED
        if passed_before and passed_after:
            verdict.pass_to_pass.append(test_id)
        elif passed_before:
            verdict.pass_to_fail.append(test_id)
        elif passed_after:
            verdict.fail_to_pass.append(test_id)
        else:
            verdict.fail_to_fail.append(test_id)

    verdict.added = {test_id: outcome for test_id, outcome in after.items() if test_id not in before}
    added_failing = [t for t, outcome in verdict.added.items() if outcome in (FAILED, ERROR)]

    if verdict.pass_to_fail or added_failing:
        problems = []
        if verdict.pass_to_fail:
            problems.append(f"broke {len(verdict.pass_to_fail)} previously passing test(s)")
        if added_failing:
            problems.append(f"added {len(added_failing)} failing test(s)")
        verdict.status = NOT_SOLVED
        verdict.reason = "The patch " + " and ".join(problems) + "."
    elif verdict.fail_to_pass and verdict.reproduction and not reproduced_fix:
        verdict.status = UNVERIFIED
        verdict.reason = (
            f"Nothing broke and {len(verdict.fail_to_pass)} test(s) now pass, but "
            f"{len(still_failing)} of the issue's {len(verdict.reproduction)} reproduction test(s) still fail."
        )
    elif verdict.fail_to_pass:
        verdict.status = SOLVED
        verdict.reason = (
            f"{len(verdict.fail_to_pass)} test(s) went from failing to passing "
            "and none of the passing tests broke."
        )
        if reproduced_fix:
            verdict.reason += " The issue's reproduction test now passes."
    elif not before:
        verdict.status = UNVERIFIED
        verdict.reason = "The baseline run produced no test results to compare against."
    elif verdict.fail_to_fail:
        verdict.status = UNVERIFIED
        verdict.reason = (
            f"Nothing broke, but the {len(verdict.fail_to_fail)} test(s) that failed "
            "before still fail, so no test shows the fix."
        )
    else:
        verdict.status = UNVERIFIED
        verdict.reason = (
            "Every test passed both before and after the patch, so no test shows the fix."
        )
    return verdict


def _compare_overall(baseline: dict[str, Any], patched: dict[str, Any]) -> TestVerdict:
    """Fallback for runners without per-test results: compare PASS/FAIL only."""
    if patched.get("status") != "PASS":
        errors = patched.get("errors") or []
        detail = f": {errors[0]}" if errors else "."
        return TestVerdict(
            status=NOT_SOLVED,
            reason=f"The test suite fails after the patch{detail}",
            per_test=False,
        )
    if baseline.get("status") == "PASS":
        return TestVerdict(
            status=UNVERIFIED,
            reason="The suite passed both before and after the patch, so no test shows the fix.",
            per_test=False,
        )
    return TestVerdict(
        status=SOLVED,
        reason="The suite failed before the patch and passes after it.",
        per_test=False,
    )
