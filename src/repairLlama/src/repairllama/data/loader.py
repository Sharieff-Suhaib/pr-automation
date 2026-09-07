"""Loading bug/fix pairs from disk, without assuming any corpus layout.

Megadiff, Defects4J exports and home-grown scrapes all store bug/fix pairs
differently, so the loader is configured rather than hard-coded.  Three
built-in formats cover the shapes these corpora actually take:

``jsonl`` / ``json``
    One record per line (or one JSON array).  :class:`FieldMap` maps the
    corpus's own key names onto the canonical ones, so a file using
    ``{"id": ..., "before": ..., "after": ...}`` loads with::

        FieldMap(bug_id="id", buggy_code="before", fixed_code="after")

    Dotted paths reach into nested objects (``"commit.sha"``), and each
    canonical field also accepts a list of candidate keys.

``directory``
    A tree of per-bug directories holding the two versions as files, which is
    how Megadiff-style corpora are usually unpacked::

        <root>/<bug>/buggy.java
        <root>/<bug>/fixed.java
        <root>/<bug>/metadata.json      # optional, merged in

    :class:`DirectoryLayout` configures the file names (or glob patterns), the
    recursion depth, where the id and project come from, and whether an
    optional metadata file is read.

``pairs`` (in-memory)
    ``load_pairs`` also accepts an iterable of dicts, for tests and for
    callers that already have records in hand.

Custom corpora register their own reader with :func:`register_loader` instead
of patching this module.

Nothing here downloads anything: a loader only ever reads local paths.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from repairllama.data.models import BugFixPair, DataError
from repairllama.utils.logging import get_logger

__all__ = [
    "FieldMap",
    "DirectoryLayout",
    "LoaderOptions",
    "load_pairs",
    "load_jsonl",
    "load_json",
    "load_directory",
    "load_records",
    "register_loader",
    "available_formats",
]

log = get_logger("data.loader")

CANONICAL_FIELDS = (
    "bug_id",
    "buggy_code",
    "fixed_code",
    "suspicious_start",
    "suspicious_end",
    "file_path",
    "project",
)


@dataclass(frozen=True)
class FieldMap:
    """Maps a corpus's key names onto the canonical ones.

    Each value is a key name, a dotted path (``"commit.sha"``), or a list of
    candidates tried in order.  Defaults are the canonical names plus the
    aliases these corpora commonly use.
    """

    bug_id: Union[str, Sequence[str]] = ("bug_id", "id", "identifier", "name")
    buggy_code: Union[str, Sequence[str]] = ("buggy_code", "buggy", "before", "input")
    fixed_code: Union[str, Sequence[str]] = ("fixed_code", "fixed", "after", "output")
    suspicious_start: Union[str, Sequence[str]] = ("suspicious_start", "start_line", "start")
    suspicious_end: Union[str, Sequence[str]] = ("suspicious_end", "end_line", "end")
    file_path: Union[str, Sequence[str]] = ("file_path", "path", "file")
    project: Union[str, Sequence[str]] = ("project", "repo", "repository")
    metadata: Union[str, Sequence[str]] = ("metadata", "meta")
    keep_unmapped_as_metadata: bool = True

    def candidates(self, field_name: str) -> List[str]:
        value = getattr(self, field_name)
        return [value] if isinstance(value, str) else list(value)

    def lookup(self, record: Mapping[str, Any], field_name: str) -> Optional[Any]:
        """First candidate key present in ``record``, or None."""
        for candidate in self.candidates(field_name):
            found, value = _dotted_get(record, candidate)
            if found and value is not None:
                return value
        return None

    def mapped_keys(self) -> set:
        """Top-level keys the map consumes (used to build the metadata rest)."""
        keys = set()
        for name in (*CANONICAL_FIELDS, "metadata"):
            for candidate in self.candidates(name):
                keys.add(candidate.split(".", 1)[0])
        return keys


def _dotted_get(record: Mapping[str, Any], path: str) -> Tuple[bool, Any]:
    cursor: Any = record
    for part in path.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return (False, None)
        cursor = cursor[part]
    return (True, cursor)


def _as_line_number(value: Any, field_name: str, bug_id: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise DataError(f"{bug_id}: {field_name} must be a line number, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise DataError(f"{bug_id}: {field_name} must be a line number, got {value!r}")


@dataclass(frozen=True)
class DirectoryLayout:
    """Where the two versions of a bug live inside a directory tree."""

    buggy_name: str = "buggy.java"
    fixed_name: str = "fixed.java"
    metadata_name: Optional[str] = "metadata.json"
    recursive: bool = True
    # Where the bug id comes from: the containing directory, or the file stem.
    id_from: str = "parent"  # parent | relative_path | stem
    # Optional regex over the path relative to the root; group 1 is the project.
    project_pattern: Optional[str] = None
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        if self.id_from not in {"parent", "relative_path", "stem"}:
            raise DataError(f"unknown id_from: {self.id_from!r}")
        if not self.buggy_name or not self.fixed_name:
            raise DataError("buggy_name and fixed_name must not be empty")


@dataclass(frozen=True)
class LoaderOptions:
    """How to read a corpus."""

    format: str = "jsonl"  # jsonl | json | directory | a registered name
    field_map: FieldMap = field(default_factory=FieldMap)
    layout: DirectoryLayout = field(default_factory=DirectoryLayout)
    encoding: str = "utf-8"
    limit: int = 0  # 0 == no cap
    skip_invalid: bool = False
    infer_region: bool = True  # derive the region from the diff when absent


# --------------------------------------------------------------------------- #
# record -> BugFixPair
# --------------------------------------------------------------------------- #
def record_to_pair(
    record: Mapping[str, Any], options: Optional[LoaderOptions] = None
) -> BugFixPair:
    """Map one raw record onto a :class:`BugFixPair`."""
    options = options or LoaderOptions()
    field_map = options.field_map

    bug_id = field_map.lookup(record, "bug_id")
    buggy = field_map.lookup(record, "buggy_code")
    fixed = field_map.lookup(record, "fixed_code")
    missing = [
        name
        for name, value in (
            ("bug_id", bug_id),
            ("buggy_code", buggy),
            ("fixed_code", fixed),
        )
        if value is None
    ]
    if missing:
        raise DataError(
            f"record is missing required field(s) {missing}; "
            f"available keys: {sorted(record)}"
        )
    bug_id = str(bug_id)

    metadata: Dict[str, Any] = {}
    explicit = field_map.lookup(record, "metadata")
    if isinstance(explicit, Mapping):
        metadata.update(explicit)
    if field_map.keep_unmapped_as_metadata:
        consumed = field_map.mapped_keys()
        metadata.update(
            {key: value for key, value in record.items() if key not in consumed}
        )

    pair = BugFixPair(
        bug_id=bug_id,
        buggy_code=str(buggy),
        fixed_code=str(fixed),
        suspicious_start=_as_line_number(
            field_map.lookup(record, "suspicious_start"), "suspicious_start", bug_id
        ),
        suspicious_end=_as_line_number(
            field_map.lookup(record, "suspicious_end"), "suspicious_end", bug_id
        ),
        file_path=_optional_str(field_map.lookup(record, "file_path")),
        project=_optional_str(field_map.lookup(record, "project")),
        metadata=metadata,
    )
    if options.infer_region and not pair.has_region and not pair.is_identical:
        pair = pair.with_region()
    return pair


def _optional_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


# --------------------------------------------------------------------------- #
# format readers
# --------------------------------------------------------------------------- #
def _read_json_lines(path: Path, encoding: str) -> Iterator[Mapping[str, Any]]:
    with path.open("r", encoding=encoding) as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{line_no}: invalid JSON — {exc}") from exc
            if not isinstance(record, Mapping):
                raise DataError(f"{path}:{line_no}: expected a JSON object")
            yield record


def load_jsonl(path: Path, options: LoaderOptions) -> Iterator[Mapping[str, Any]]:
    """One JSON object per line."""
    yield from _read_json_lines(path, options.encoding)


def load_json(path: Path, options: LoaderOptions) -> Iterator[Mapping[str, Any]]:
    """A JSON array of objects, or an object whose values are the records."""
    payload = json.loads(path.read_text(encoding=options.encoding))
    if isinstance(payload, Mapping):
        payload = list(payload.values())
    if not isinstance(payload, list):
        raise DataError(f"{path}: expected a JSON array or object of records")
    for index, record in enumerate(payload):
        if not isinstance(record, Mapping):
            raise DataError(f"{path}[{index}]: expected a JSON object")
        yield record


def load_directory(path: Path, options: LoaderOptions) -> Iterator[Mapping[str, Any]]:
    """A tree of per-bug directories holding the buggy and fixed versions."""
    layout = options.layout
    pattern = ("**/" if layout.recursive else "") + layout.buggy_name
    project_re = re.compile(layout.project_pattern) if layout.project_pattern else None

    for buggy_path in sorted(path.glob(pattern)):
        if not buggy_path.is_file():
            continue
        fixed_path = _fixed_path_for(buggy_path, layout)
        if fixed_path is None or not fixed_path.is_file():
            log.debug("no fixed counterpart for %s; skipping", buggy_path)
            continue

        relative = buggy_path.relative_to(path)
        record: Dict[str, Any] = {
            "bug_id": _bug_id_for(relative, buggy_path, layout),
            "buggy_code": buggy_path.read_text(encoding=layout.encoding),
            "fixed_code": fixed_path.read_text(encoding=layout.encoding),
            "file_path": str(relative),
        }
        if project_re is not None:
            match = project_re.search(str(relative))
            if match:
                record["project"] = match.group(1) if match.groups() else match.group(0)
        if layout.metadata_name:
            meta_path = buggy_path.parent / layout.metadata_name
            if meta_path.is_file():
                extra = json.loads(meta_path.read_text(encoding=layout.encoding))
                if isinstance(extra, Mapping):
                    # Explicit metadata wins over path-derived defaults.
                    record.update(
                        {k: v for k, v in extra.items() if k not in {"buggy_code", "fixed_code"}}
                    )
        yield record


def _fixed_path_for(buggy_path: Path, layout: DirectoryLayout) -> Optional[Path]:
    """Locate the fixed file next to ``buggy_path``.

    Handles both fixed names (``buggy.java`` -> ``fixed.java``) and glob-style
    names (``*.buggy.java`` -> ``*.fixed.java``) by substituting the part of
    the pattern that differs.
    """
    if "*" not in layout.buggy_name:
        return buggy_path.parent / layout.fixed_name
    prefix, _, suffix = layout.buggy_name.partition("*")
    stem = buggy_path.name[len(prefix) : len(buggy_path.name) - len(suffix)]
    return buggy_path.parent / layout.fixed_name.replace("*", stem)


def _bug_id_for(relative: Path, buggy_path: Path, layout: DirectoryLayout) -> str:
    if layout.id_from == "relative_path":
        return str(relative)
    if layout.id_from == "stem":
        return buggy_path.stem
    parent = relative.parent
    return str(parent) if str(parent) not in {"", "."} else buggy_path.stem


Reader = Callable[[Path, LoaderOptions], Iterable[Mapping[str, Any]]]

_READERS: Dict[str, Reader] = {
    "jsonl": load_jsonl,
    "json": load_json,
    "directory": load_directory,
}


def register_loader(name: str, reader: Reader) -> None:
    """Register a reader for a corpus layout this module does not know."""
    _READERS[name] = reader


def available_formats() -> List[str]:
    return sorted(_READERS)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def load_records(
    source: Union[str, Path, Iterable[Mapping[str, Any]]],
    options: Optional[LoaderOptions] = None,
) -> Iterator[Mapping[str, Any]]:
    """Yield raw records from a path (per ``options.format``) or an iterable."""
    options = options or LoaderOptions()
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.exists():
            raise DataError(f"dataset source not found: {path}")
        reader = _READERS.get(options.format)
        if reader is None:
            raise DataError(
                f"unknown loader format {options.format!r}; "
                f"available: {available_formats()}"
            )
        if options.format == "directory" and not path.is_dir():
            raise DataError(f"format 'directory' needs a directory, got {path}")
        yield from reader(path, options)
    else:
        yield from source


def load_pairs(
    source: Union[str, Path, Iterable[Mapping[str, Any]]],
    options: Optional[LoaderOptions] = None,
) -> Iterator[BugFixPair]:
    """Load a corpus and normalise it into :class:`BugFixPair` objects.

    With ``options.skip_invalid`` a malformed record is logged and skipped;
    otherwise it raises :class:`~repairllama.data.models.DataError`.
    ``options.limit`` caps how many pairs are produced.
    """
    options = options or LoaderOptions()
    produced = 0
    for index, record in enumerate(load_records(source, options)):
        try:
            pair = record_to_pair(record, options)
        except DataError as exc:
            if not options.skip_invalid:
                raise
            log.warning("skipping record %d: %s", index, exc)
            continue
        yield pair
        produced += 1
        if options.limit and produced >= options.limit:
            log.debug("stopping after %d records (limit)", produced)
            return
