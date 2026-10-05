"""Prompts for the Reflection Agent."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = (
    "You are a senior engineer reviewing a failed automated bug fix. You explain why "
    "the patch failed and how the next attempt must differ. Reply with one JSON object only."
)

RETRY_HEADER = (
    "This is a repair retry. The previous patch failed validation. "
    "Use the reflection feedback below to generate an improved patch. "
    "Do not repeat the previous patch."
)

_MAX_PATCH = 3000
_MAX_CODE = 4000
_MAX_MESSAGE = 400


def build_reflection_prompt(
    issue: str,
    relevant_code: str,
    generated_patch: str,
    failure_type: str,
    verdict_reason: str,
    failed_tests: dict[str, str],
    repair_strategy: str | None = None,
    previous_attempts: list[dict[str, Any]] | None = None,
) -> str:
    """The brief for analysing one failed attempt. ``failed_tests`` maps test id -> message."""
    tests = "\n".join(
        f"- {test_id}: {_cut(message, _MAX_MESSAGE) or '(no message)'}" for test_id, message in failed_tests.items()
    ) or "(no individual test failures were recorded)"
    history = "\n".join(
        f"- attempt {row.get('attempt')}: {row.get('status')} - {row.get('reason', '')}"
        for row in previous_attempts or []
    ) or "(none)"

    return f"""ISSUE:
{issue.strip()}

FAULTY CODE (before any patch):
{_cut(relevant_code.strip(), _MAX_CODE) or "(not available)"}

REPAIR STRATEGY THAT WAS FOLLOWED:
{repair_strategy or "(none)"}

PATCH THAT FAILED:
{_cut(generated_patch.strip(), _MAX_PATCH) or "(no patch was produced)"}

VALIDATION RESULT:
Failure type: {failure_type}
Verdict: {verdict_reason}
Failed tests:
{tests}

EARLIER ATTEMPTS:
{history}

TASK:
Explain why this patch did not fix the issue and how the next patch must differ.
1. Did the patch address the cause described in the issue, or only a symptom?
2. What do the failing tests and error messages show?
3. Did the patch break behaviour that worked before?
Do NOT suggest repeating the previous patch; the next attempt must change something it did not.

Reply with exactly this JSON object:
{{"root_cause": "<why the tests still fail>",
 "patch_analysis": "<what the patch changed and why that was not enough>",
 "suggested_changes": ["<concrete change 1>", "<concrete change 2>"],
 "repair_guidance": "<one paragraph telling the code generator what to do differently>",
 "confidence": <number between 0 and 1>}}
"""


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n... (truncated)"
