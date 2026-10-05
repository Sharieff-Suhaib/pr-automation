"""Turn a rejected attempt into feedback the coding agent can act on, and rank attempts.

The retry loop sends the coding agent back to work with this text: what was
wrong with the last patch, which tests it broke, which still fail and why. It
always repairs the original code again (the rejected patch is not kept), so the
previous patch is quoted only so the model can avoid repeating it.
"""

from __future__ import annotations

from typing import Any

_MAX_TESTS = 8
_MAX_MESSAGE = 300
_MAX_PATCH = 3000

_RANK = {"solved": 2, "unverified": 1, "failed": 0}


def build_feedback(
    attempt: int,
    patch: str,
    patch_status: str,
    test_result: dict[str, Any],
    verdict: dict[str, Any],
    patch_error: str = "",
    repeated: bool = False,
) -> str:
    """A short plain-text account of why attempt ``attempt`` was not accepted.

    ``repeated`` marks a patch identical to an earlier rejected one.
    """
    lines = [f"Attempt {attempt} was rejected: {verdict.get('reason') or 'no reason recorded'}"]
    if repeated:
        lines.append(
            "This patch is identical to an earlier rejected attempt. "
            "Take a different approach this time."
        )

    if not patch.strip():
        lines.append(f"No usable patch was produced. {patch_error}".rstrip())
        return "\n".join(lines)

    if patch_status != "APPLIED":
        errors = test_result.get("errors") or []
        lines.append(f"The patch could not be applied ({patch_status}): {errors[0] if errors else ''}".rstrip())
    else:
        messages = test_result.get("messages") or {}
        reproduction = [t for t, outcome in (verdict.get("reproduction") or {}).items() if outcome != "passed"]
        broken = verdict.get("pass_to_fail") or []
        still_failing = [t for t in verdict.get("fail_to_fail") or [] if t not in reproduction]

        for title, tests in (
            ("Tests that reproduce the issue and still fail (these must pass)", reproduction),
            ("Tests that passed before your patch and now fail (do not break these)", broken),
            ("Tests that still fail", still_failing),
        ):
            if tests:
                lines.append(f"{title}:")
                lines += [_test_line(test_id, messages) for test_id in tests[:_MAX_TESTS]]
                if len(tests) > _MAX_TESTS:
                    lines.append(f"- ... and {len(tests) - _MAX_TESTS} more")

    lines += ["Previous patch (rejected, do not repeat it):", _truncate(patch.strip(), _MAX_PATCH)]
    return "\n".join(lines)


def is_better(candidate: dict[str, Any], current: dict[str, Any] | None) -> bool:
    """Whether attempt ``candidate`` beats ``current``: verdict, then fewer broken, then more fixed.

    On a full tie the earlier attempt is kept.
    """
    if current is None:
        return True
    return _score(candidate) > _score(current)


def _score(attempt: dict[str, Any]) -> tuple[int, int, int]:
    verdict = attempt.get("test_verdict") or {}
    return (
        _RANK.get(attempt.get("status", "failed"), 0),
        -len(verdict.get("pass_to_fail") or []),
        len(verdict.get("fail_to_pass") or []),
    )


def _test_line(test_id: str, messages: dict[str, str]) -> str:
    message = messages.get(test_id, "")
    return f"- {test_id}" + (f": {_truncate(message, _MAX_MESSAGE)}" if message else "")


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n... (truncated)"
