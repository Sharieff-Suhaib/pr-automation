"""Reflection Agent: explain why a repair attempt failed and guide the next one.

It never edits code. Given a failed attempt -- the patch, the before/after test
comparison and the failure messages -- it returns a `ReflectionResult`:

    failure_type        classified from the test results (deterministic)
    root_cause,         analysis from the LLM; when the LLM is unreachable or
    patch_analysis,     answers with unusable JSON, rule-based text is used, so
    suggested_changes,  the retry loop never depends on the model's formatting
    repair_guidance,
    confidence

`ReflectionResult.as_feedback()` is what the Code Generation Agent receives on
the next attempt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
import os
import re
from typing import Any, Callable

from src.agents.reflection_agent.prompts import RETRY_HEADER, SYSTEM_PROMPT, build_reflection_prompt

logger = logging.getLogger("agent_swe.reflection")

MAX_REPAIR_ATTEMPTS = int(os.environ.get("AGENT_SWE_MAX_REPAIR_ATTEMPTS", "3"))

STATUSES = ("retry", "solved", "max_retries")
FAILURE_TYPES = (
    "syntax_error",
    "compilation_error",
    "test_failure",
    "runtime_error",
    "logic_error",
    "regression",
    "patch_application_error",
    "unknown",
)

# prompt -> raw model reply; raises on failure.
LLM = Callable[[str], str]

_EXCEPTION_RE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt))\b")


@dataclass
class ReflectionResult:
    """Structured feedback on one failed (or solved) repair attempt."""

    status: str
    failure_type: str
    root_cause: str
    patch_analysis: str
    failed_tests: list[str] = field(default_factory=list)
    suggested_changes: list[str] = field(default_factory=list)
    repair_guidance: str = ""
    confidence: float = 0.0
    attempt: int = 0
    source: str = "rules"  # "llm" | "rules"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_feedback(self) -> str:
        """The reflection as prompt text for the Code Generation Agent's next attempt."""
        changes = "\n".join(f"- {change}" for change in self.suggested_changes) or "- (none)"
        tests = ", ".join(self.failed_tests) or "(none recorded)"
        return (
            f"{RETRY_HEADER}\n\n"
            f"REFLECTION ON ATTEMPT {self.attempt} ({self.failure_type}):\n"
            f"Root cause: {self.root_cause}\n"
            f"What the previous patch did wrong: {self.patch_analysis}\n"
            f"Failed tests: {tests}\n"
            f"Suggested changes:\n{changes}\n"
            f"Guidance: {self.repair_guidance}"
        )


class ReflectionAgent:
    """Analyses a failed repair attempt. Pass ``llm=None`` to use the coding agent's Ollama model."""

    def __init__(self, llm: LLM | None = None, use_llm: bool = True):
        self.llm = llm
        self.use_llm = use_llm

    def reflect(
        self,
        issue: str,
        relevant_code: str,
        generated_patch: str,
        test_results: dict[str, Any],
        repair_strategy: str | None = None,
        previous_attempts: list[dict[str, Any]] | None = None,
        attempt: int = 1,
        max_attempts: int = MAX_REPAIR_ATTEMPTS,
    ) -> ReflectionResult:
        """Reflect on one attempt.

        ``test_results`` is the testing agent's attempt record: ``patch_status``,
        ``test_result`` (with per-test ``messages``), ``test_verdict`` and
        ``patch_error``.
        """
        verdict = test_results.get("test_verdict") or {}
        if verdict.get("status") == "solved":
            return ReflectionResult("solved", "unknown", "", "The patch passed validation.", attempt=attempt,
                                    confidence=1.0)

        status = "retry" if attempt < max_attempts else "max_retries"
        failed = failed_tests(test_results)
        failure_type = classify_failure(generated_patch, test_results)
        logger.info("[ReflectionAgent] Analyzing failed repair (attempt %d/%d)...", attempt, max_attempts)
        logger.info("[ReflectionAgent] Failed tests: %d", len(failed))
        logger.info("[ReflectionAgent] Failure type: %s", failure_type)

        result = rule_based_reflection(status, failure_type, generated_patch, test_results, failed, attempt)
        if self.use_llm:
            prompt = build_reflection_prompt(
                issue, relevant_code, generated_patch, failure_type, verdict.get("reason", ""),
                failed, repair_strategy, previous_attempts,
            )
            try:
                analysis = parse_analysis(self._complete(prompt))
            except Exception as error:  # the model is optional: any failure falls back to the rules
                logger.warning("[ReflectionAgent] LLM analysis unavailable (%s); using rule-based reflection", error)
                analysis = None
            if analysis:
                result = _merge(result, analysis)

        logger.info("[ReflectionAgent] Root cause identified (%s): %s", result.source, _first_line(result.root_cause))
        logger.info("[ReflectionAgent] Repair guidance generated")
        if status == "retry":
            logger.info("[ReflectionAgent] Retry attempt: %d/%d", attempt + 1, max_attempts)
        else:
            logger.info("[ReflectionAgent] Maximum attempts (%d) reached; stopping", max_attempts)
        return result

    def _complete(self, prompt: str) -> str:
        if self.llm is not None:
            return self.llm(prompt)
        from src.agents.coding_agent.code_generator import _call_ollama  # noqa: PLC0415 -- reuse, no new loader

        return _call_ollama(prompt, system=SYSTEM_PROMPT)


def failed_tests(test_results: dict[str, Any]) -> dict[str, str]:
    """``{test_id: failure message}``: broken tests first, then reproduction, then other still-failing."""
    verdict = test_results.get("test_verdict") or {}
    messages = (test_results.get("test_result") or {}).get("messages") or {}
    reproduction = [t for t, outcome in (verdict.get("reproduction") or {}).items() if outcome != "passed"]
    ordered = [*(verdict.get("pass_to_fail") or []), *reproduction, *(verdict.get("fail_to_fail") or [])]
    return {test_id: messages.get(test_id, "") for test_id in dict.fromkeys(ordered)}


def classify_failure(generated_patch: str, test_results: dict[str, Any]) -> str:
    """Map the testing agent's outcome onto one of FAILURE_TYPES."""
    verdict = test_results.get("test_verdict") or {}
    if not generated_patch.strip():
        error = test_results.get("patch_error", "")
        return "syntax_error" if "not valid Python" in error else "compilation_error" if error else "unknown"
    if test_results.get("patch_status") not in (None, "APPLIED"):
        return "patch_application_error"
    if verdict.get("stage") == "syntax":
        return "syntax_error"
    if verdict.get("pass_to_fail"):
        return "regression"

    messages = [m for m in failed_tests(test_results).values() if m]
    exceptions = [m for m in messages if _EXCEPTION_RE.match(m) and not m.startswith("AssertionError")]
    if exceptions:
        return "runtime_error"
    if messages:
        return "logic_error"
    if failed_tests(test_results):
        return "test_failure"
    return "unknown"


def rule_based_reflection(
    status: str,
    failure_type: str,
    generated_patch: str,
    test_results: dict[str, Any],
    failed: dict[str, str],
    attempt: int,
) -> ReflectionResult:
    """A reflection built only from the test results; used when the LLM is unavailable."""
    verdict = test_results.get("test_verdict") or {}
    first_test, first_message = next(iter(failed.items()), ("", ""))
    evidence = f" ({first_test}: {first_message})" if first_message else (f" ({first_test})" if first_test else "")
    errors = (test_results.get("test_result") or {}).get("errors") or [test_results.get("patch_error", "")]

    templates = {
        "syntax_error": ("The patched code is not valid Python.",
                         ["Return complete, syntactically valid function definitions."]),
        "compilation_error": (f"No usable patch was produced: {errors[0]}",
                              ["Return only the corrected functions, as valid Python."]),
        "patch_application_error": (f"The patch did not apply to the repository: {errors[0]}",
                                    ["Edit the code exactly as shown; do not invent context lines."]),
        "regression": (f"The patch broke behaviour that worked before{evidence}.",
                       ["Keep the existing behaviour that the broken tests check.",
                        "Limit the change to the case described in the issue."]),
        "runtime_error": (f"The code still raises an exception on the issue's input{evidence}.",
                          ["Handle the input that triggers the exception before it is raised.",
                           "Return the value the issue asks for instead of raising."]),
        "logic_error": (f"The code runs but returns the wrong result{evidence}.",
                        ["Re-check the computation against the expected values in the failing test."]),
        "test_failure": ("Tests that should pass after the fix still fail.",
                         ["Change the logic the failing tests exercise."]),
        "unknown": (verdict.get("reason") or "No test shows that the patch fixes the issue.",
                    ["Change the behaviour the issue describes, not only the code's form."]),
    }
    root_cause, changes = templates.get(failure_type, templates["unknown"])
    changed = "no patch" if not generated_patch.strip() else f"{_changed_lines(generated_patch)} changed line(s)"
    return ReflectionResult(
        status=status,
        failure_type=failure_type,
        root_cause=root_cause,
        patch_analysis=f"The previous patch ({changed}) was rejected: {verdict.get('reason', 'see the failing tests')}",
        failed_tests=list(failed),
        suggested_changes=[*changes, "Take a different approach from the previous patch."],
        repair_guidance=f"{root_cause} {' '.join(changes)} Do not repeat the previous patch.",
        confidence=0.3,
        attempt=attempt,
        source="rules",
    )


def parse_analysis(reply: str) -> dict[str, Any] | None:
    """The first JSON object in the reply, keeping only well-typed fields; ``None`` if there is none."""
    decoder = json.JSONDecoder()
    for start in [match.start() for match in re.finditer(r"\{", reply)]:
        try:
            data, _ = decoder.raw_decode(reply[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return _clean_analysis(data)
    return None


def _clean_analysis(data: dict[str, Any]) -> dict[str, Any] | None:
    analysis: dict[str, Any] = {}
    for key in ("root_cause", "patch_analysis", "repair_guidance"):
        if isinstance(data.get(key), str) and data[key].strip():
            analysis[key] = data[key].strip()
    changes = data.get("suggested_changes")
    if isinstance(changes, str):
        changes = [changes]
    if isinstance(changes, list):
        cleaned = [str(change).strip() for change in changes if str(change).strip()]
        if cleaned:
            analysis["suggested_changes"] = cleaned
    try:
        analysis["confidence"] = min(max(float(data.get("confidence")), 0.0), 1.0)
    except (TypeError, ValueError):
        pass
    # Without a root cause or guidance the reply says nothing the rules do not.
    return analysis if {"root_cause", "repair_guidance"} & analysis.keys() else None


def _merge(rules: ReflectionResult, analysis: dict[str, Any]) -> ReflectionResult:
    merged = ReflectionResult(**{**rules.to_dict(), **analysis})
    merged.source = "llm"
    if "confidence" not in analysis:
        merged.confidence = 0.5
    return merged


def _changed_lines(patch: str) -> int:
    return sum(
        1 for line in patch.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else "(none)"
