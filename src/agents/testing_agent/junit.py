"""Read per-test outcomes from a JUnit XML report.

pytest writes one with ``--junitxml=<path>``; most other ecosystems can emit the
same format, which is why the per-test comparison is built on it rather than on
any one runner's console output.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

# Outcome labels used everywhere in the testing agent.
PASSED = "passed"
FAILED = "failed"
ERROR = "error"
SKIPPED = "skipped"


def parse_junit_xml(path: str | Path) -> dict[str, str]:
    """Return ``{test_id: outcome}`` for every ``<testcase>`` in the report.

    ``test_id`` is ``classname::name`` (for pytest, ``test_users::test_login``),
    which stays identical between two runs of the same suite and is what the
    before/after comparison keys on. A missing or malformed report yields ``{}``.
    """
    return parse_junit_report(path)[0]


def parse_junit_report(path: str | Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(outcomes, messages)``; ``messages`` holds only failed and errored tests.

    A message is the runner's one-line reason, e.g. ``KeyError: 'ghost'`` or
    ``assert None is False``. pytest leads with the exception type for anything
    but a bare ``assert``, which is how a broken test is told apart from a real
    failure.
    """
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return {}, {}

    outcomes: dict[str, str] = {}
    messages: dict[str, str] = {}
    for case in root.iter("testcase"):
        classname = case.get("classname", "")
        name = case.get("name", "")
        test_id = f"{classname}::{name}" if classname else name
        outcome, message = _outcome(case)
        outcomes[test_id] = outcome
        if outcome in (FAILED, ERROR):
            messages[test_id] = message
    return outcomes, messages


def _outcome(case: ET.Element) -> tuple[str, str]:
    """Classify one ``<testcase>``; an error outranks a failure, which outranks a skip."""
    children = {child.tag: child for child in case}
    for tag, outcome in ((ERROR, ERROR), ("failure", FAILED), (SKIPPED, SKIPPED)):
        if tag in children:
            return outcome, children[tag].get("message", "")
    return PASSED, ""
