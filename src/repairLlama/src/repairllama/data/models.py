"""The intermediate format the dataset pipeline speaks, and its report types.

Everything upstream (Megadiff, Defects4J, a GitHub PR scraper, a hand-written
JSONL file) is normalised into :class:`BugFixPair` before any processing
happens.  That is the only contract the rest of the pipeline depends on::

    {
      "bug_id": "megadiff-1a2b3c",          # required, unique
      "buggy_code": "public int max(...",   # required, the buggy unit
      "fixed_code": "public int max(...",   # required, the same unit fixed
      "suspicious_start": 4,                # optional, 1-based inclusive
      "suspicious_end": 4,                  # optional, 1-based inclusive
      "file_path": "src/main/java/Foo.java",# optional
      "project": "commons-lang",            # optional, used for split grouping
      "metadata": {"commit": "..."}         # optional, carried through
    }

``buggy_code`` and ``fixed_code`` are the *same unit* before and after the
fix — normally one Java method, optionally with its enclosing class.  When the
suspicious region is absent it is derived from the line diff
(:meth:`BugFixPair.changed_span`), which is "perfect localization": exactly
what the reference fix touched.

The pipeline turns each pair into a :class:`TrainingRecord` whose JSONL row
carries ``input`` (the IR4 prompt) and ``output`` (the OR2 replacement).
"""

from __future__ import annotations

import difflib
import statistics
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "DataError",
    "BugFixPair",
    "TrainingRecord",
    "RejectionReason",
    "Rejection",
    "StageReport",
    "TokenStats",
    "DatasetReport",
]


class DataError(ValueError):
    """Raised when a record cannot be interpreted as a bug/fix pair."""


# --------------------------------------------------------------------------- #
# the intermediate format
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BugFixPair:
    """One buggy/fixed unit of Java source.

    Line numbers are 1-based and inclusive, matching
    :mod:`repairllama.representation`.
    """

    bug_id: str
    buggy_code: str
    fixed_code: str
    suspicious_start: Optional[int] = None
    suspicious_end: Optional[int] = None
    file_path: Optional[str] = None
    project: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.bug_id:
            raise DataError("bug_id must not be empty")
        if not isinstance(self.buggy_code, str) or not isinstance(self.fixed_code, str):
            raise DataError(f"{self.bug_id}: buggy_code and fixed_code must be strings")
        if (self.suspicious_start is None) != (self.suspicious_end is None):
            raise DataError(
                f"{self.bug_id}: suspicious_start and suspicious_end must be given "
                "together or not at all"
            )

    # -- views -------------------------------------------------------------- #
    @property
    def buggy_lines(self) -> List[str]:
        return self.buggy_code.splitlines()

    @property
    def fixed_lines(self) -> List[str]:
        return self.fixed_code.splitlines()

    @property
    def has_region(self) -> bool:
        return self.suspicious_start is not None and self.suspicious_end is not None

    @property
    def group_key(self) -> str:
        """Key used to keep related examples in the same split."""
        return self.project or self.file_path or self.bug_id

    @property
    def is_identical(self) -> bool:
        return self.buggy_code == self.fixed_code

    # -- derivation --------------------------------------------------------- #
    def changed_span(self) -> Optional[Tuple[int, int]]:
        """The 1-based inclusive buggy lines the fix touches, or None.

        This is perfect localization: the smallest region covering every line
        the reference diff changed.  A pure insertion is anchored on the line
        it follows, so the region is never empty.  Returns ``None`` when the
        two versions are identical.
        """
        matcher = difflib.SequenceMatcher(
            a=self.buggy_lines, b=self.fixed_lines, autojunk=False
        )
        changed = [op for op in matcher.get_opcodes() if op[0] != "equal"]
        if not changed:
            return None
        lower = min(op[1] for op in changed)
        upper = max(op[2] for op in changed)
        if upper <= lower:  # pure insertion(s): anchor on the preceding line
            anchor = max(1, lower)
            return (anchor, anchor)
        return (lower + 1, upper)

    def with_region(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> "BugFixPair":
        """Return a copy carrying a suspicious region.

        With no arguments the region is derived from the diff; an existing
        region is kept as-is.
        """
        if start is None and end is None:
            if self.has_region:
                return self
            span = self.changed_span()
            if span is None:
                raise DataError(
                    f"{self.bug_id}: cannot derive a region — buggy and fixed "
                    "code are identical"
                )
            start, end = span
        if start is None or end is None:
            raise DataError(f"{self.bug_id}: start and end must be given together")
        return BugFixPair(
            bug_id=self.bug_id,
            buggy_code=self.buggy_code,
            fixed_code=self.fixed_code,
            suspicious_start=start,
            suspicious_end=end,
            file_path=self.file_path,
            project=self.project,
            metadata=dict(self.metadata),
        )

    # -- serialisation ------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "bug_id": self.bug_id,
            "buggy_code": self.buggy_code,
            "fixed_code": self.fixed_code,
        }
        if self.has_region:
            payload["suspicious_start"] = self.suspicious_start
            payload["suspicious_end"] = self.suspicious_end
        if self.file_path:
            payload["file_path"] = self.file_path
        if self.project:
            payload["project"] = self.project
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BugFixPair":
        missing = [key for key in ("bug_id", "buggy_code", "fixed_code") if not data.get(key)]
        if missing:
            raise DataError(f"record is missing required field(s): {missing}")
        return cls(
            bug_id=str(data["bug_id"]),
            buggy_code=data["buggy_code"],
            fixed_code=data["fixed_code"],
            suspicious_start=data.get("suspicious_start"),
            suspicious_end=data.get("suspicious_end"),
            file_path=data.get("file_path"),
            project=data.get("project"),
            metadata=dict(data.get("metadata") or {}),
        )


# --------------------------------------------------------------------------- #
# the training record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainingRecord:
    """One training example: an IR4 prompt and its OR2 replacement.

    ``to_json_dict`` emits ``input`` and ``output`` first; with
    ``minimal=True`` it emits nothing else.
    """

    input: str
    output: str
    bug_id: str
    suspicious_start: int
    suspicious_end: int
    file_path: Optional[str] = None
    project: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    split: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> Optional[int]:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens

    @property
    def group_key(self) -> str:
        return self.project or self.file_path or self.bug_id

    def with_tokens(self, input_tokens: int, output_tokens: int) -> "TrainingRecord":
        return replace_record(self, input_tokens=input_tokens, output_tokens=output_tokens)

    def with_split(self, split: str) -> "TrainingRecord":
        return replace_record(self, split=split)

    def to_json_dict(self, minimal: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"input": self.input, "output": self.output}
        if minimal:
            return payload
        payload.update(
            {
                "bug_id": self.bug_id,
                "suspicious_start": self.suspicious_start,
                "suspicious_end": self.suspicious_end,
            }
        )
        for key in ("file_path", "project", "input_tokens", "output_tokens", "split"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


def replace_record(record: TrainingRecord, **changes: Any) -> TrainingRecord:
    """``dataclasses.replace`` for :class:`TrainingRecord` (keeps it importable)."""
    from dataclasses import replace as _replace

    return _replace(record, **changes)


# --------------------------------------------------------------------------- #
# rejections and reporting
# --------------------------------------------------------------------------- #
class RejectionReason(str, Enum):
    """Why a sample left the pipeline.  The report counts these by name."""

    MISSING_FIELD = "missing_field"
    EMPTY_SOURCE = "empty_source"
    IDENTICAL = "identical"
    NO_CHANGE = "no_change"
    INVALID_REGION = "invalid_region"
    DIFF_TOO_SMALL = "diff_too_small"
    DIFF_TOO_LARGE = "diff_too_large"
    NOT_SINGLE_FUNCTION = "not_single_function"
    TEST_ONLY = "test_only"
    DUPLICATE = "duplicate"
    INPUT_TOO_LONG = "input_too_long"
    OUTPUT_TOO_LONG = "output_too_long"
    REPRESENTATION_ERROR = "representation_error"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class Rejection:
    """A dropped sample, with enough detail to audit the decision."""

    bug_id: str
    reason: RejectionReason
    detail: str = ""
    stage: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bug_id": self.bug_id,
            "reason": self.reason.value,
            "detail": self.detail,
            "stage": self.stage,
        }


@dataclass(frozen=True)
class StageReport:
    """How many samples a stage received, kept and dropped."""

    name: str
    received: int
    kept: int
    removed_by_reason: Dict[str, int] = field(default_factory=dict)

    @property
    def removed(self) -> int:
        return self.received - self.kept

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "received": self.received,
            "kept": self.kept,
            "removed": self.removed,
            "removed_by_reason": dict(self.removed_by_reason),
        }


@dataclass(frozen=True)
class TokenStats:
    """Distribution of a token count over the dataset."""

    count: int
    total: int
    minimum: int
    maximum: int
    mean: float
    median: float
    p90: int
    p95: int
    p99: int
    histogram: Dict[str, int] = field(default_factory=dict)

    @staticmethod
    def _percentile(values: Sequence[int], fraction: float) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return ordered[index]

    @classmethod
    def from_values(
        cls, values: Sequence[int], buckets: Sequence[int] = (64, 128, 256, 512, 1024, 2048)
    ) -> "TokenStats":
        values = list(values)
        if not values:
            return cls(0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, {})
        histogram: Dict[str, int] = {}
        edges = list(buckets)
        previous = 0
        for edge in edges:
            histogram[f"{previous}-{edge}"] = sum(
                1 for value in values if previous < value <= edge
            )
            previous = edge
        histogram[f">{previous}"] = sum(1 for value in values if value > previous)
        return cls(
            count=len(values),
            total=sum(values),
            minimum=min(values),
            maximum=max(values),
            mean=round(statistics.fmean(values), 2),
            median=statistics.median(values),
            p90=cls._percentile(values, 0.90),
            p95=cls._percentile(values, 0.95),
            p99=cls._percentile(values, 0.99),
            histogram=histogram,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "total": self.total,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "median": self.median,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "histogram": dict(self.histogram),
        }


@dataclass
class DatasetReport:
    """The end-to-end account of one dataset build."""

    dataset_name: str = ""
    raw_samples: int = 0
    remaining: int = 0
    duplicates: int = 0
    stages: List[StageReport] = field(default_factory=list)
    removed_by_reason: Dict[str, int] = field(default_factory=dict)
    token_stats: Dict[str, TokenStats] = field(default_factory=dict)
    split_counts: Dict[str, int] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    generated_at: Optional[str] = None

    @property
    def removed(self) -> int:
        return self.raw_samples - self.remaining

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "generated_at": self.generated_at,
            "raw_samples": self.raw_samples,
            "removed": self.removed,
            "duplicates": self.duplicates,
            "remaining": self.remaining,
            "removed_by_reason": dict(self.removed_by_reason),
            "stages": [stage.to_dict() for stage in self.stages],
            "token_stats": {name: stats.to_dict() for name, stats in self.token_stats.items()},
            "splits": dict(self.split_counts),
            "settings": dict(self.settings),
        }

    def render(self) -> str:
        """A human-readable report — what the CLI prints after a build."""
        width = 66
        lines = [
            "=" * width,
            f"dataset report: {self.dataset_name or '(unnamed)'}",
            "=" * width,
            f"raw samples      {self.raw_samples:>8}",
            f"removed          {self.removed:>8}",
            f"  of which dupes {self.duplicates:>8}",
            f"remaining        {self.remaining:>8}",
        ]

        if self.removed_by_reason:
            lines += ["", "removed by reason"]
            for reason, count in sorted(
                self.removed_by_reason.items(), key=lambda item: (-item[1], item[0])
            ):
                lines.append(f"  {reason:<24} {count:>8}")

        if self.stages:
            lines += ["", "stages"]
            for stage in self.stages:
                lines.append(
                    f"  {stage.name:<24} {stage.received:>6} -> {stage.kept:>6}"
                    f"  (-{stage.removed})"
                )

        for name, stats in self.token_stats.items():
            lines += ["", f"token length: {name}"]
            if not stats.count:
                lines.append("  (no samples measured)")
                continue
            lines.append(
                f"  min {stats.minimum}  median {stats.median}  mean {stats.mean}  "
                f"p90 {stats.p90}  p95 {stats.p95}  p99 {stats.p99}  max {stats.maximum}"
            )
            for bucket, count in stats.histogram.items():
                bar = "#" * min(40, count) if count else ""
                lines.append(f"    {bucket:<12} {count:>6} {bar}")

        if self.split_counts:
            lines += ["", "splits"]
            total = sum(self.split_counts.values()) or 1
            for split, count in self.split_counts.items():
                lines.append(f"  {split:<12} {count:>8}  ({count / total:.1%})")

        if self.settings:
            lines += ["", "settings"]
            for key, value in self.settings.items():
                lines.append(f"  {key:<24} {value}")

        lines.append("=" * width)
        return "\n".join(lines)

    def save(self, path: str | Path, also_text: bool = True) -> Path:
        """Write ``report.json`` (and a sibling ``.txt``) and return the path."""
        import json

        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, default=str) + "\n", encoding="utf-8"
        )
        if also_text:
            target.with_suffix(".txt").write_text(self.render() + "\n", encoding="utf-8")
        return target
