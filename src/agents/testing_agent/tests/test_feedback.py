from src.agents.testing_agent.feedback import build_feedback, is_better

PATCH = "--- a/users.py\n+++ b/users.py\n@@ -1 +1 @@\n-old\n+new\n"


def verdict(status="failed", reason="r", **lists):
    base = {"fail_to_pass": [], "pass_to_fail": [], "fail_to_fail": [], "reproduction": {}}
    return {"status": status, "reason": reason, **base, **lists}


def test_feedback_names_the_failing_tests_with_their_messages():
    text = build_feedback(
        2,
        PATCH,
        "APPLIED",
        {"messages": {"t::broken": "assert False", "repro::r": "KeyError: 'x'", "t::old": "boom"}},
        verdict(
            reason="The patch broke 1 previously passing test(s).",
            pass_to_fail=["t::broken"],
            fail_to_fail=["repro::r", "t::old"],
            reproduction={"repro::r": "failed"},
        ),
    )

    assert text.startswith("Attempt 2 was rejected: The patch broke 1")
    assert "reproduce the issue and still fail" in text
    assert "- repro::r: KeyError: 'x'" in text
    assert "- t::broken: assert False" in text
    assert "- t::old: boom" in text
    assert text.count("repro::r") == 1  # not repeated under "still fail"
    assert PATCH.strip() in text


def test_feedback_for_a_patch_that_did_not_apply():
    text = build_feedback(1, PATCH, "PATCH_APPLY_FAILED", {"errors": ["corrupt patch at line 4"]}, verdict())

    assert "could not be applied (PATCH_APPLY_FAILED): corrupt patch at line 4" in text


def test_feedback_without_a_patch():
    text = build_feedback(1, "", "SKIPPED", {}, verdict(reason="No patch"), patch_error="Ollama timed out")

    assert "No usable patch was produced. Ollama timed out" in text
    assert "Previous patch" not in text


def test_attempts_rank_by_verdict_then_broken_then_fixed():
    def attempt(status, fixed=0, broken=0):
        return {"status": status, "test_verdict": verdict(
            status, fail_to_pass=["x"] * fixed, pass_to_fail=["y"] * broken)}

    assert is_better(attempt("failed"), None)
    assert is_better(attempt("unverified"), attempt("failed", fixed=3))
    assert is_better(attempt("failed", broken=1), attempt("failed", broken=2))
    assert is_better(attempt("failed", fixed=2), attempt("failed", fixed=1))
    assert not is_better(attempt("failed"), attempt("failed"))  # ties keep the earlier attempt
