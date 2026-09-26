"""Small JSON / JSONL read-write helpers used across the pipeline stages."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

__all__ = ["read_json", "write_json", "read_jsonl", "write_jsonl", "count_lines"]


def read_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | os.PathLike[str], payload: Any, indent: int = 2) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False, default=str)
        handle.write("\n")
    return target


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[Dict[str, Any]]:
    """Stream a JSONL file, skipping blank lines."""
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON — {exc}") from exc


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Dict[str, Any]]) -> int:
    """Write ``rows`` as JSONL; returns the number of records written."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            written += 1
    return written


def count_lines(path: str | os.PathLike[str]) -> int:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def load_lines(path: str | os.PathLike[str]) -> List[str]:
    return Path(path).expanduser().read_text(encoding="utf-8").splitlines()
