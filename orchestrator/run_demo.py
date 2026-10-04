"""Run the workflow end to end and save the report.

Offline demo -- uses the bundled `sample_repo`, no network and no model needed:
    python -m orchestrator.run_demo --offline

Real run -- clones the repository and generates the patch with Ollama:
    python -m orchestrator.run_demo \
        --repo-url https://github.com/Sharieff-Suhaib/dummy_repo.git \
        --issue "get_user crashes with a KeyError when the username is not registered"

Each run prints a readable summary and writes the full JSON report to
`orchestrator/outputs/`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from orchestrator.manager_agent import solve_issue

ORCHESTRATOR_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = ORCHESTRATOR_DIR / "outputs"
SAMPLE_REPO = ORCHESTRATOR_DIR / "sample_repo"
SAMPLE_PATCH = ORCHESTRATOR_DIR / "fixtures" / "sample_repo_fix.patch"
SAMPLE_REPRO = ORCHESTRATOR_DIR / "fixtures" / "sample_repo_reproduction.py"

SAMPLE_ISSUE = (
    "get_user crashes with a KeyError when the username is not registered. "
    "Looking up a missing user should return None instead of raising, and "
    "login() should reject an unknown user instead of crashing."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-agent repair workflow")
    parser.add_argument("--repo-url", default="", help="Repository URL, or a local directory path")
    parser.add_argument("--issue", default="", help="The GitHub issue text")
    parser.add_argument("--top-k", type=int, default=5, help="Code chunks to retrieve (default: 5)")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the bundled sample repository and replay the fixture diff instead of calling Ollama",
    )
    parser.add_argument(
        "--no-repro-test", action="store_true", help="Skip writing a test that reproduces the issue"
    )
    parser.add_argument("--max-attempts", type=int, default=3, help="Patches to try (default: 3)")
    parser.add_argument(
        "--no-isolated-env", action="store_true", help="Run tests with this interpreter, not a per-repo venv"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.offline:
        repo_url = args.repo_url or str(SAMPLE_REPO)
        issue = args.issue or SAMPLE_ISSUE
        backend, stub_patch, stub_repro = "stub", str(SAMPLE_PATCH), str(SAMPLE_REPRO)
    else:
        if not args.repo_url or not args.issue:
            raise SystemExit("A real run needs --repo-url and --issue (or pass --offline).")
        repo_url, issue = args.repo_url, args.issue
        backend, stub_patch, stub_repro = "ollama", "", ""

    print(f"Repository : {repo_url}")
    print(f"Issue      : {issue}")
    print(f"Patch via  : {backend}\n")

    report = solve_issue(
        repo_url=repo_url,
        issue=issue,
        top_k=args.top_k,
        codegen_backend=backend,
        stub_patch_path=stub_patch,
        reproduce=not args.no_repro_test,
        stub_repro_path=stub_repro,
        max_attempts=args.max_attempts,
        isolated_env=not args.no_isolated_env,
    )

    print(format_report(report))
    output_path = save_report(report)
    print(f"\nFull JSON report: {output_path}")

    return 0 if report["status"] == "solved" else 1


def format_report(report: dict[str, Any]) -> str:
    """Render the report as the plain-text summary printed after a run."""
    lines = ["=" * 70, "RELEVANT FILES (Repository Agent)", "=" * 70]
    lines += _bullets(report["relevant_files"])

    lines += ["", "=" * 70, "SIMILAR BUGS (Recommendation Agent)", "=" * 70]
    for bug in report["similar_bugs"]:
        lines.append(f'- "{bug["issue"]}" (similarity {bug["similarity"]})')
        lines.append(f'    previous fix: {bug["fix"]}')
    if not report["similar_bugs"]:
        lines.append("(none)")

    lines += [
        "",
        "=" * 70,
        "REPAIR STRATEGY",
        "=" * 70,
        f"Bug type : {report['bug_type'] or '(none)'}",
        f"Strategy : {report['strategy'] or '(none)'}",
        "",
        "=" * 70,
        "RECOMMENDED TOOLS",
        "=" * 70,
        *_bullets(report["tools"]),
        "",
        "=" * 70,
        "RECOMMENDED TESTS",
        "=" * 70,
        *_bullets(report["tests"]),
        "",
        *_reproduction_lines(report["reproduction"]),
        "",
        "=" * 70,
        f"PATCH (Code Generation Agent, source={report['patch_source']})",
        "=" * 70,
        report["patch"].rstrip() or "(no patch generated)",
        "",
        "=" * 70,
        "TESTING AGENT",
        "=" * 70,
        f"Test env     : {report['test_env'].get('status', '(none)')} - {report['test_env'].get('detail', '')}",
        f"Patch status : {report['patch_status']}",
        f"Changed files: {', '.join(report['changed_files']) or '(none)'}",
        f"Before patch : {_run_summary(report['baseline_result'])}",
        f"After patch  : {_run_summary(report['test_result'])}",
    ]
    if report["test_result"].get("stages"):
        lines.append(f"Test stages  : {_stages_summary(report['test_result']['stages'])}")

    for error in report["test_result"].get("errors", []):
        lines.append(f"    {error}")

    verdict = report["test_verdict"]
    if verdict:
        lines.append(f"Verdict      : {verdict['status'].upper()} - {verdict['reason']}")
        for label, key in (
            ("Fixed (fail -> pass)", "fail_to_pass"),
            ("Broken (pass -> fail)", "pass_to_fail"),
            ("Still failing", "fail_to_fail"),
        ):
            if verdict.get(key):
                lines.append(f"  {label}:")
                lines += [f"    - {test_id}" for test_id in verdict[key]]
        for test_id, outcome in verdict.get("reproduction", {}).items():
            lines.append(f"  Reproduction test after patch: {test_id} {outcome}")
        if verdict.get("pass_to_pass"):
            lines.append(f"  Still passing: {len(verdict['pass_to_pass'])} test(s)")
        for test_id, outcome in verdict.get("added", {}).items():
            lines.append(f"  Added by the patch: {test_id} ({outcome})")

    if len(report["attempts"]) > 1:
        lines += ["", "=" * 70, "ATTEMPTS (retry loop)", "=" * 70]
        for row in report["attempts"]:
            lines.append(
                f"{row['attempt']}. {row['status'].upper():<10} patch {row['patch_status']} via "
                f"{row['patch_source']}: {row['fixed']} fixed, {row['broken']} broken - {row['reason']}"
            )

    lines += ["", "=" * 70, "TRACE", "=" * 70]
    lines += [f"{entry['agent']:<22} {entry['summary']}" for entry in report["trace"]]

    if report["errors"]:
        lines += ["", "=" * 70, "ERRORS", "=" * 70]
        lines += [f"- {error}" for error in report["errors"]]

    lines += ["", f"WORKFLOW STATUS: {report['status'].upper()}"]
    return "\n".join(lines)


def _reproduction_lines(reproduction: dict[str, Any]) -> list[str]:
    """The REPRODUCTION TEST section: status, why, and the test file when one was kept."""
    lines = ["=" * 70, "REPRODUCTION TEST", "=" * 70]
    if not reproduction:
        return lines + ["(not written)"]

    lines.append(
        f"Status : {reproduction['status'].upper()} "
        f"(source={reproduction.get('source_of', '?')}, {reproduction['attempts']} attempt(s))"
    )
    lines.append(f"Reason : {reproduction['reason']}")
    for test_id, outcome in reproduction["baseline_outcomes"].items():
        message = reproduction["messages"].get(test_id, "")
        lines.append(f"  before patch: {test_id} {outcome}" + (f" ({message})" if message else ""))
    if reproduction["source"]:
        lines += ["", f"# {reproduction['path']}", reproduction["source"].rstrip()]
    return lines


def _stages_summary(stages: list[dict[str, Any]]) -> str:
    """`syntax PASS -> reproduction PASS (2/2) -> full PASS (6/6)`."""
    parts = []
    for stage in stages:
        counts = ""
        if "passed" in stage:
            counts = f" ({stage['passed']}/{stage['passed'] + stage['failed']})"
        parts.append(f"{stage['name']} {stage['status']}{counts}")
    return " -> ".join(parts)


def _run_summary(result: dict[str, Any]) -> str:
    """`PASS (4 passed, 0 failed)` for one run of the suite, or `(not run)`."""
    if not result:
        return "(not run)"
    return f"{result['status']} ({result.get('passed', 0)} passed, {result.get('failed', 0)} failed)"


def _bullets(items: list[str]) -> list[str]:
    """Bullet lines for a list, or a single `(none)` line when it is empty."""
    return [f"- {item}" for item in items] or ["(none)"]


def save_report(report: dict[str, Any]) -> Path:
    """Write the JSON report into `orchestrator/outputs/` and return its path."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = OUTPUTS_DIR / f"run-{stamp}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # A stable filename so the newest run is always easy to open.
    (OUTPUTS_DIR / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
