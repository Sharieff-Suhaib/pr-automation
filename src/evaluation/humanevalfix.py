"""HumanEvalFix (Python) — functional-correctness evaluation of generated patches.

Unlike src/evaluation/evaluate.py, which scores text/AST similarity to a
reference fix, this scores what actually matters: does the patched function pass
its unit tests?

The 164 problems come from `bigcode/humanevalpack`. Each carries a
`buggy_solution` (a manually introduced defect), a `canonical_solution`, and a
`test` suite.

Run:
    # validate the harness itself (no model, ~1 min) — do this first
    python -m src.evaluation.humanevalfix --self-test

    # baseline: untuned base model
    python -m src.evaluation.humanevalfix --no-adapter

    # with the trained adapter
    python -m src.evaluation.humanevalfix

    # quick check on 10 problems
    python -m src.evaluation.humanevalfix --limit 10

Caveats worth remembering when reading the numbers:
  * The bugs are synthetic (introduced into HumanEval solutions), not mined
    from real projects.
  * HumanEval is old and widely scraped, so the base model has very likely seen
    the correct solutions in pretraining. Trust the tuned-vs-baseline delta far
    more than the absolute score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.dataset.loader import RepairExample
from src.evaluation.execute import check_candidate, pass_at_k
from src.training.config import load_config

DATASET_NAME = "bigcode/humanevalpack"
DATASET_CONFIG = "python"


def build_issue(row: dict[str, Any], include_bug_hints: bool) -> str:
    """Compose the bug report shown to the model.

    By default only the docstring (the intended behavior) is given — this is the
    standard, harder HumanEvalFix setting. `include_bug_hints` additionally
    reveals the dataset's `bug_type`/`failure_symptoms` annotations, which makes
    the task substantially easier and is NOT comparable to published numbers.
    """
    docstring = (row.get("docstring") or "").strip()
    issue = (
        f"The function `{row['entry_point']}` is buggy and fails its tests.\n\n"
        f"Expected behavior:\n{docstring}"
    )
    if include_bug_hints:
        hints = [row.get("bug_type"), row.get("failure_symptoms")]
        hint_text = "\n".join(f"- {h}" for h in hints if h and str(h).strip())
        if hint_text:
            issue += f"\n\nDiagnostics:\n{hint_text}"
    return issue


def row_to_example(row: dict[str, Any], include_bug_hints: bool = False) -> RepairExample:
    """Map a HumanEvalPack row onto our RepairExample schema.

    `declaration` (imports + signature) is joined with the solution body so the
    model sees — and returns — a complete function.

    Note there is no fault_location: HumanEvalFix does not say which line is
    wrong. That is realistic, and it exercises the no-localization path of the
    repair representation.
    """
    return RepairExample(
        id=row["task_id"],
        buggy_code=row["declaration"] + row["buggy_solution"],
        fixed_code=row["declaration"] + row["canonical_solution"],
        issue=build_issue(row, include_bug_hints),
        fault_location=None,
        extras={
            "entry_point": row["entry_point"],
            "test": row["test"],
            "import": row.get("import", ""),
            "test_setup": row.get("test_setup", ""),
        },
    )


def load_humanevalfix(limit: int | None = None, include_bug_hints: bool = False):
    """Download (cached after first run) and convert the Python split."""
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")
    rows = list(dataset)
    if limit:
        rows = rows[:limit]
    print(f"[humanevalfix] loaded {len(rows)} problems")
    return [row_to_example(row, include_bug_hints) for row in rows], rows


def run_tests(example: RepairExample, candidate_code: str, timeout: float):
    """Execute one candidate against its problem's test suite."""
    return check_candidate(
        candidate_code=candidate_code,
        test_code=example.extras["test"],
        entry_point=example.extras["entry_point"],
        imports=example.extras.get("import", ""),
        test_setup=example.extras.get("test_setup", ""),
        timeout=timeout,
    )


def self_test(limit: int | None = None, timeout: float = 15.0) -> int:
    """Validate the harness without a model.

    Canonical solutions must pass and buggy solutions must (mostly) fail. If
    that does not hold, the harness is broken and every score it produces is
    meaningless — so this runs before any model evaluation.
    """
    examples, rows = load_humanevalfix(limit=limit)
    canonical_passed = buggy_passed = 0
    canonical_failures: list[str] = []

    for index, (example, row) in enumerate(zip(examples, rows), start=1):
        print(f"[self-test] ({index}/{len(examples)}) {example.id}", end="\r")

        result = run_tests(example, example.fixed_code, timeout)
        if result.passed:
            canonical_passed += 1
        else:
            canonical_failures.append(f"{example.id}: {result.status} — {result.detail}")

        # The buggy version is expected to fail; a pass means the injected
        # defect is not actually covered by the tests.
        if run_tests(example, example.buggy_code, timeout).passed:
            buggy_passed += 1

    total = len(examples)
    print("\n" + "=" * 70)
    print("HARNESS SELF-TEST")
    print("=" * 70)
    print(f"  canonical solutions passing : {canonical_passed}/{total}  (expected: all)")
    print(f"  buggy solutions passing     : {buggy_passed}/{total}  (expected: ~0)")

    if canonical_failures:
        print("\n  canonical failures — harness bug, not model error:")
        for failure in canonical_failures[:10]:
            print(f"    {failure}")

    ok = canonical_passed == total
    print("\n  VERDICT:", "harness is trustworthy" if ok else "HARNESS IS BROKEN — do not trust scores")
    print("=" * 70)
    return 0 if ok else 1


def evaluate(
    generator,
    examples: list[RepairExample],
    num_candidates: int,
    timeout: float,
) -> dict[str, Any]:
    """Generate candidates for each problem and score them by test execution."""
    results: list[dict[str, Any]] = []

    for index, example in enumerate(examples, start=1):
        record: dict[str, Any] = {
            "id": example.id,
            "issue": example.issue,
            "buggy_code": example.buggy_code,
            "expected_fixed_code": example.fixed_code,
        }
        try:
            candidates = generator.generate(
                issue=example.issue,
                buggy_code=example.buggy_code,
                fault_location=example.fault_location,
                num_candidates=num_candidates,
            )
            outcomes = [run_tests(example, candidate, timeout) for candidate in candidates]
            num_correct = sum(o.passed for o in outcomes)
            record.update(
                {
                    "candidates": [
                        {"code": c, **o.as_dict()} for c, o in zip(candidates, outcomes)
                    ],
                    "generated_code": candidates[0] if candidates else "",
                    "num_candidates": len(candidates),
                    "num_correct": num_correct,
                    "solved": num_correct > 0,
                    "error": None,
                }
            )
            status = "SOLVED" if num_correct else outcomes[0].status if outcomes else "no output"
        except Exception as exc:  # a bad problem must not abort the run
            record.update(
                {
                    "candidates": [],
                    "generated_code": "",
                    "num_candidates": 0,
                    "num_correct": 0,
                    "solved": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            status = "ERROR"

        solved_so_far = sum(r["solved"] for r in results) + int(record["solved"])
        print(f"[eval] ({index}/{len(examples)}) {example.id}: {status}  "
              f"[running: {solved_so_far}/{index}]")
        results.append(record)

    return {"summary": summarize(results, num_candidates), "results": results}


def summarize(results: list[dict[str, Any]], num_candidates: int) -> dict[str, Any]:
    """Aggregate per-problem outcomes into pass@k and failure-mode counts."""
    total = len(results)
    summary: dict[str, Any] = {
        "total_problems": total,
        "solved": sum(r["solved"] for r in results),
        "errors": sum(1 for r in results if r["error"]),
        "num_candidates_per_problem": num_candidates,
    }
    if total:
        summary["pass@1"] = round(
            sum(pass_at_k(r["num_candidates"], r["num_correct"], 1) for r in results if r["num_candidates"]) / total,
            4,
        )
        # pass@k is only meaningful with more samples than k.
        if num_candidates >= 5:
            summary["pass@5"] = round(
                sum(pass_at_k(r["num_candidates"], r["num_correct"], 5) for r in results if r["num_candidates"] >= 5) / total,
                4,
            )

    # Why the failures failed — distinguishes "wrong fix" from "broken output".
    modes: dict[str, int] = {}
    for record in results:
        for candidate in record["candidates"]:
            modes[candidate["status"]] = modes.get(candidate["status"], 0) + 1
    summary["candidate_status_counts"] = modes
    return summary


def print_report(report: dict[str, Any], show_failures: int = 3) -> None:
    print("\n" + "=" * 70)
    print("HUMANEVALFIX — FUNCTIONAL CORRECTNESS")
    print("=" * 70)
    for key, value in report["summary"].items():
        print(f"  {key:>28}: {value}")

    failures = [r for r in report["results"] if not r["solved"]][:show_failures]
    if failures:
        print("\n" + "=" * 70)
        print(f"SAMPLE FAILURES (first {len(failures)})")
        print("=" * 70)
        for record in failures:
            detail = record["error"] or (
                record["candidates"][0]["detail"] if record["candidates"] else "no candidates"
            )
            print(f"\n--- {record['id']} — {detail} ---")
            print(record["generated_code"][:600] or "(empty)")


def main() -> int:
    parser = argparse.ArgumentParser(description="HumanEvalFix functional evaluation")
    parser.add_argument("--self-test", action="store_true",
                        help="Validate the harness against reference solutions (no model).")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N problems")
    parser.add_argument("--candidates", type=int, default=1, help="Candidates per problem")
    parser.add_argument("--timeout", type=float, default=15.0, help="Per-test timeout in seconds")
    parser.add_argument("--no-adapter", action="store_true", help="Evaluate the base model")
    parser.add_argument("--bug-hints", action="store_true",
                        help="Include bug_type/failure_symptoms in the issue (easier; non-standard)")
    parser.add_argument("--output", default=None, help="Where to write the JSON report")
    args = parser.parse_args()

    if args.self_test:
        return self_test(limit=args.limit, timeout=args.timeout)

    from src.inference.generate_patch import PatchGenerator

    config = load_config()
    examples, _ = load_humanevalfix(limit=args.limit, include_bug_hints=args.bug_hints)

    generator = PatchGenerator.from_config(config, use_adapter=not args.no_adapter)
    report = evaluate(generator, examples, args.candidates, args.timeout)
    report["summary"]["adapter"] = "none (base model)" if args.no_adapter else str(
        config.generation.adapter_dir
    )
    print_report(report)

    tag = "base" if args.no_adapter else "adapter"
    output_path = Path(args.output or config.data.processed_dir / f"humanevalfix_{tag}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[humanevalfix] report written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
