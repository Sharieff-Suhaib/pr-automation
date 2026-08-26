"""Load bug-fix examples from JSONL and normalize them into RepairExample objects.

The on-disk schema is intentionally forgiving: only `buggy_code` and `fixed_code`
are required. Everything else (issue text, fault location, tests, metadata) is
optional and defaults to something the prompt builder can handle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

# Accepted aliases so we can point the loader at third-party datasets later
# without rewriting every record first.
_BUGGY_KEYS = ("buggy_code", "buggy", "source", "input", "before")
_FIXED_KEYS = ("fixed_code", "fixed", "target", "output", "after", "patch")
_ISSUE_KEYS = ("issue", "bug_description", "description", "problem_statement", "message")


@dataclass
class FaultLocation:
    """1-indexed, inclusive line range pointing at the suspicious code."""

    start_line: int
    end_line: int

    def as_dict(self) -> dict[str, int]:
        return {"start_line": self.start_line, "end_line": self.end_line}


@dataclass
class RepairExample:
    """One normalized bug-fix pair."""

    id: str
    buggy_code: str
    fixed_code: str
    issue: str | None = None
    fault_location: FaultLocation | None = None
    # Free-form slots reserved for later Agent-SWE stages (similar bugs,
    # repository context, relevant tests, recommended tools).
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "issue": self.issue,
            "buggy_code": self.buggy_code,
            "fixed_code": self.fixed_code,
            "fault_location": self.fault_location.as_dict() if self.fault_location else None,
            "extras": self.extras,
        }


class DatasetError(Exception):
    """Raised when a dataset file is missing or contains no usable examples."""


def _first_present(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _parse_fault_location(raw: Any, buggy_code: str) -> FaultLocation | None:
    """Accept {"start_line": x, "end_line": y}, [x, y], or a bare int.

    Returns None when the value is absent or cannot be interpreted, and clamps
    the range to the actual number of lines in the buggy code so a malformed
    annotation never produces a nonsensical prompt.
    """
    if raw is None:
        return None

    start = end = None
    if isinstance(raw, dict):
        start = raw.get("start_line", raw.get("start"))
        end = raw.get("end_line", raw.get("end", start))
    elif isinstance(raw, (list, tuple)) and len(raw) >= 1:
        start = raw[0]
        end = raw[1] if len(raw) > 1 else raw[0]
    elif isinstance(raw, int):
        start = end = raw

    try:
        start, end = int(start), int(end)
    except (TypeError, ValueError):
        return None

    total_lines = len(buggy_code.splitlines()) or 1
    start = max(1, min(start, total_lines))
    end = max(start, min(end, total_lines))
    return FaultLocation(start_line=start, end_line=end)


def parse_record(record: dict[str, Any], fallback_id: str) -> RepairExample | None:
    """Convert one raw JSON record into a RepairExample, or None if unusable."""
    buggy = _first_present(record, _BUGGY_KEYS)
    fixed = _first_present(record, _FIXED_KEYS)
    if buggy is None or fixed is None:
        return None

    buggy, fixed = str(buggy), str(fixed)
    issue = _first_present(record, _ISSUE_KEYS)

    # Anything we do not recognize is preserved for future pipeline stages.
    known = set(_BUGGY_KEYS + _FIXED_KEYS + _ISSUE_KEYS) | {"id", "fault_location"}
    extras = {k: v for k, v in record.items() if k not in known}

    return RepairExample(
        id=str(record.get("id") or fallback_id),
        buggy_code=buggy,
        fixed_code=fixed,
        issue=str(issue) if issue else None,
        fault_location=_parse_fault_location(record.get("fault_location"), buggy),
        extras=extras,
    )


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file, skipping blank lines."""
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc


def load_examples(path: Path | str, verbose: bool = True) -> list[RepairExample]:
    """Load and normalize every usable example from a JSONL file.

    Records missing `buggy_code`/`fixed_code` are skipped with a warning rather
    than aborting the run — real-world datasets are rarely complete.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(
            f"Dataset file not found: {path}\n"
            f"Set AGENT_SWE_TRAIN_FILE to a JSONL file, or use the bundled "
            f"data/raw/sample_bugs.jsonl."
        )

    examples: list[RepairExample] = []
    skipped: list[str] = []
    for index, record in enumerate(iter_jsonl(path)):
        example = parse_record(record, fallback_id=f"{path.stem}_{index:04d}")
        if example is None:
            skipped.append(str(record.get("id", f"line {index + 1}")))
            continue
        examples.append(example)

    if verbose and skipped:
        print(
            f"[loader] skipped {len(skipped)} record(s) missing buggy/fixed code: "
            f"{', '.join(skipped[:5])}{' ...' if len(skipped) > 5 else ''}"
        )
    if not examples:
        raise DatasetError(f"No usable examples found in {path}.")
    if verbose:
        print(f"[loader] loaded {len(examples)} example(s) from {path}")
    return examples


def train_eval_split(
    examples: list[RepairExample], eval_split: float, seed: int
) -> tuple[list[RepairExample], list[RepairExample]]:
    """Deterministic shuffle-and-split holdout.

    Guarantees at least one eval example (when more than one exists) and never
    leaves the training set empty.
    """
    import random

    if eval_split <= 0 or len(examples) < 2:
        return list(examples), []

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    n_eval = max(1, round(len(shuffled) * eval_split))
    n_eval = min(n_eval, len(shuffled) - 1)  # keep at least one training example
    return shuffled[n_eval:], shuffled[:n_eval]