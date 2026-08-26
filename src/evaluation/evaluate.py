"""Evaluate generated patches against the held-out bug-fix examples.

Deliberately simple for milestone 1 — no Defects4J-style test execution yet.
Three cheap signals are reported per example:

  * syntax_valid  : does the generated patch parse as Python at all?
  * exact_match   : identical to the reference fix after whitespace normalization.
  * ast_match     : structurally identical ignoring formatting/comments
                    (catches correct fixes that merely differ in layout).

Run:
    python -m src.evaluation.evaluate
    python -m src.evaluation.evaluate --limit 3 --candidates 3 --no-adapter
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from src.dataset.loader import RepairExample
from src.dataset.preprocess import prepare_datasets
from src.training.config import load_config

# PatchGenerator is imported lazily inside main() so that the scoring helpers
# below stay importable (and unit-testable) on machines without torch.


def normalize_code(code: str) -> str:
    """Strip trailing whitespace and blank lines for a fair textual comparison."""
    lines = [line.rstrip() for line in code.strip().splitlines()]
    return "\n".join(line for line in lines if line.strip())


def is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def ast_equivalent(generated: str, reference: str) -> bool:
    """Compare parse trees, so formatting and comments do not count as differences."""
    try:
        return ast.dump(ast.parse(generated)) == ast.dump(ast.parse(reference))
    except SyntaxError:
        return False


def score_candidates(candidates: list[str], reference: str) -> dict[str, Any]:
    """Best-of-N scoring: an example counts as matched if ANY candidate matches."""
    normalized_reference = normalize_code(reference)
    per_candidate = [
        {
            "candidate": candidate,
            "syntax_valid": is_syntax_valid(candidate),
            "exact_match": normalize_code(candidate) == normalized_reference,
            "ast_match": ast_equivalent(candidate, reference),
        }
        for candidate in candidates
    ]
    return {
        "candidates": per_candidate,
        "any_syntax_valid": any(c["syntax_valid"] for c in per_candidate),
        "any_exact_match": any(c["exact_match"] for c in per_candidate),
        "any_ast_match": any(c["ast_match"] for c in per_candidate),
    }


def evaluate_examples(
    generator: "PatchGenerator",
    examples: list[RepairExample],
    num_candidates: int | None = None,
) -> dict[str, Any]:
    """Generate and score patches for every example, tolerating per-example errors."""
    results: list[dict[str, Any]] = []
    errors = 0

    for index, example in enumerate(examples, start=1):
        print(f"[evaluate] ({index}/{len(examples)}) {example.id}")
        record: dict[str, Any] = {
            "id": example.id,
            "issue": example.issue,
            "buggy_code": example.buggy_code,
            "expected_fixed_code": example.fixed_code,
            "fault_location": (
                example.fault_location.as_dict() if example.fault_location else None
            ),
        }
        try:
            candidates = generator.generate(
                issue=example.issue,
                buggy_code=example.buggy_code,
                fault_location=example.fault_location,
                extras=example.extras,
                num_candidates=num_candidates,
            )
            record.update(score_candidates(candidates, example.fixed_code))
            record["generated_code"] = candidates[0] if candidates else ""
            record["error"] = None
        except Exception as exc:  # keep going; one bad example must not kill the run
            errors += 1
            record.update(
                {
                    "candidates": [],
                    "generated_code": "",
                    "any_syntax_valid": False,
                    "any_exact_match": False,
                    "any_ast_match": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[evaluate]   error: {record['error']}")
        results.append(record)

    total = len(results)
    summary = {
        "total_examples": total,
        "errors": errors,
        "syntax_valid": sum(r["any_syntax_valid"] for r in results),
        "exact_matches": sum(r["any_exact_match"] for r in results),
        "ast_matches": sum(r["any_ast_match"] for r in results),
    }
    # Rates as a convenience; guard against an empty eval set.
    if total:
        summary["exact_match_rate"] = round(summary["exact_matches"] / total, 4)
        summary["ast_match_rate"] = round(summary["ast_matches"] / total, 4)
        summary["syntax_valid_rate"] = round(summary["syntax_valid"] / total, 4)

    return {"summary": summary, "results": results}


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    for key, value in summary.items():
        print(f"  {key:>20}: {value}")

    print("\n" + "=" * 70)
    print("PER-EXAMPLE OUTPUTS")
    print("=" * 70)
    for record in report["results"]:
        flags = []
        if record["any_exact_match"]:
            flags.append("EXACT")
        elif record["any_ast_match"]:
            flags.append("AST")
        elif record["any_syntax_valid"]:
            flags.append("syntax-ok")
        if record["error"]:
            flags.append("ERROR")
        print(f"\n--- {record['id']} [{', '.join(flags) or 'no match'}] ---")
        print(f"issue    : {record['issue']}")
        print(f"expected :\n{record['expected_fixed_code']}")
        print(f"generated:\n{record['generated_code'] or record['error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the repair model")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples")
    parser.add_argument("--candidates", type=int, default=None, help="Candidate patches per bug")
    parser.add_argument(
        "--no-adapter", action="store_true", help="Evaluate the base model (useful as a baseline)"
    )
    parser.add_argument("--output", default=None, help="Where to write the JSON report")
    args = parser.parse_args()

    from src.inference.generate_patch import PatchGenerator

    config = load_config()
    _, _, train_examples, eval_examples = prepare_datasets(config.data)

    # Fall back to training examples only if there is no holdout at all.
    examples = eval_examples or train_examples
    if not eval_examples:
        print("[evaluate] no holdout split available; evaluating on training examples instead")
    if args.limit:
        examples = examples[: args.limit]

    generator = PatchGenerator.from_config(config, use_adapter=not args.no_adapter)
    report = evaluate_examples(generator, examples, num_candidates=args.candidates)
    print_report(report)

    output_path = Path(
        args.output or config.data.processed_dir / "evaluation_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[evaluate] report written to {output_path}")


if __name__ == "__main__":
    main()
