"""The Reflection Agent on its own: classification, structured output, LLM fallback."""

from __future__ import annotations

import json

import pytest

from src.agents.reflection_agent import (
    FAILURE_TYPES,
    ReflectionAgent,
    classify_failure,
    failed_tests,
    parse_analysis,
)

PATCH = "--- a/users.py\n+++ b/users.py\n@@ -1 +1 @@\n-    return users[name]\n+    return users[name] or None\n"


def attempt(stage="full", patch_status="APPLIED", patch_error="", messages=None, **verdict_lists):
    verdict = {"status": "failed", "reason": "r", "stage": stage, "fail_to_pass": [], "pass_to_fail": [],
               "fail_to_fail": [], "reproduction": {}, **verdict_lists}
    return {"patch_status": patch_status, "patch_error": patch_error,
            "test_result": {"messages": messages or {}, "errors": ["corrupt patch at line 4"]},
            "test_verdict": verdict}


LLM_REPLY = "Sure:\n" + json.dumps({
    "root_cause": "users[name] still raises KeyError before `or None` runs.",
    "patch_analysis": "The patch only rewrote the return expression.",
    "suggested_changes": ["Use users.get(name)", "Return False from login when the user is None"],
    "repair_guidance": "Look the user up with .get and handle None in login.",
    "confidence": 0.8,
})


def reflect(test_results, patch=PATCH, llm=None, use_llm=True, **kwargs):
    agent = ReflectionAgent(llm=llm or (lambda prompt: LLM_REPLY), use_llm=use_llm)
    return agent.reflect("get_user crashes on unknown users", "def get_user(...): ...", patch, test_results, **kwargs)


@pytest.mark.parametrize(
    ("record", "patch", "expected"),
    [
        (attempt(patch_error="The repaired users.py is not valid Python"), "", "syntax_error"),
        (attempt(patch_error="Ollama timed out"), "", "compilation_error"),
        (attempt(patch_status="PATCH_APPLY_FAILED"), PATCH, "patch_application_error"),
        (attempt(stage="syntax"), PATCH, "syntax_error"),
        (attempt(pass_to_fail=["t::a"], fail_to_fail=["t::b"]), PATCH, "regression"),
        (attempt(fail_to_fail=["t::b"], messages={"t::b": "KeyError: 'ghost'"}), PATCH, "runtime_error"),
        (attempt(fail_to_fail=["t::b"], messages={"t::b": "assert None is False"}), PATCH, "logic_error"),
        (attempt(fail_to_fail=["t::b"]), PATCH, "test_failure"),
        (attempt(), PATCH, "unknown"),
    ],
)
def test_failures_are_classified_from_the_test_results(record, patch, expected):
    assert classify_failure(patch, record) == expected
    assert expected in FAILURE_TYPES


def test_a_failed_attempt_gets_a_structured_llm_reflection():
    prompts = []
    record = attempt(fail_to_fail=["t::b"], messages={"t::b": "KeyError: 'ghost'"})

    result = reflect(record, llm=lambda prompt: prompts.append(prompt) or LLM_REPLY, attempt=1, max_attempts=3)

    assert result.to_dict() == {
        "status": "retry",
        "failure_type": "runtime_error",
        "root_cause": "users[name] still raises KeyError before `or None` runs.",
        "patch_analysis": "The patch only rewrote the return expression.",
        "failed_tests": ["t::b"],
        "suggested_changes": ["Use users.get(name)", "Return False from login when the user is None"],
        "repair_guidance": "Look the user up with .get and handle None in login.",
        "confidence": 0.8,
        "attempt": 1,
        "source": "llm",
    }
    assert "Do NOT suggest repeating the previous patch" in prompts[0]
    assert "KeyError: 'ghost'" in prompts[0] and PATCH.strip() in prompts[0]


@pytest.mark.parametrize("reply", ["no json here", '{"confidence": 0.9}', "{broken"])
def test_an_unusable_llm_reply_falls_back_to_rules(reply):
    result = reflect(attempt(pass_to_fail=["t::a"], messages={"t::a": "assert False"}), llm=lambda p: reply)

    assert result.source == "rules"
    assert result.failure_type == "regression"
    assert "broke behaviour that worked before (t::a: assert False)" in result.root_cause
    assert "Take a different approach from the previous patch." in result.suggested_changes


def test_an_unreachable_llm_falls_back_to_rules():
    def unreachable(prompt):
        raise ConnectionError("Ollama is down")

    assert reflect(attempt(), llm=unreachable).source == "rules"


def test_the_last_attempt_reports_max_retries():
    assert reflect(attempt(), attempt=2, max_attempts=3).status == "retry"
    assert reflect(attempt(), attempt=3, max_attempts=3).status == "max_retries"


def test_a_solved_attempt_needs_no_reflection():
    calls = []
    record = attempt()
    record["test_verdict"]["status"] = "solved"

    result = reflect(record, llm=lambda p: calls.append(p) or LLM_REPLY)

    assert result.status == "solved"
    assert calls == []


def test_feedback_for_the_code_generator_leads_with_the_retry_instruction():
    feedback = reflect(attempt(fail_to_fail=["t::b"]), attempt=1).as_feedback()

    assert feedback.startswith("This is a repair retry. The previous patch failed validation.")
    assert "Root cause: users[name] still raises KeyError" in feedback
    assert "- Use users.get(name)" in feedback
    assert "Failed tests: t::b" in feedback


def test_failed_tests_lists_broken_then_reproduction_then_still_failing():
    record = attempt(
        pass_to_fail=["t::broken"], fail_to_fail=["repro::r", "t::old"],
        reproduction={"repro::r": "failed"}, messages={"t::broken": "boom"},
    )

    assert failed_tests(record) == {"t::broken": "boom", "repro::r": "", "t::old": ""}


def test_parse_analysis_clamps_confidence_and_accepts_a_single_change():
    analysis = parse_analysis('{"root_cause": "x", "suggested_changes": "do y", "confidence": 7}')

    assert analysis == {"root_cause": "x", "suggested_changes": ["do y"], "confidence": 1.0}
