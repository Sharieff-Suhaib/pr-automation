"""Checkpoints during training, and the adapter artifact at the end of it.

Two different things are written, and keeping them apart matters:

*Checkpoints* (``outputs/training/<run>/checkpoint-<step>/``) are the
machinery for resuming — adapter weights plus optimizer, scheduler and RNG
state.  They are large, disposable, and rotated by ``save_total_limit``.

*The artifact* (``adapters/java-repair/``) is the deliverable: adapter weights,
the tokenizer, and metadata saying what produced them.  No optimizer state, no
base checkpoint, nothing merged.  It is what inference and evaluation load, and
what you would publish.

Resuming
--------
``resolve_resume`` accepts ``true``/``auto`` (take the newest checkpoint in the
run directory), an explicit path, or nothing.  A path that does not exist is an
error rather than a silent fresh start — resuming the wrong run, or silently
not resuming at all, wastes far more than it saves.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from repairllama.utils.logging import get_logger

__all__ = [
    "CheckpointError",
    "CHECKPOINT_PREFIX",
    "TRAINING_SUMMARY_FILE",
    "list_checkpoints",
    "find_last_checkpoint",
    "resolve_resume",
    "save_training_artifact",
    "write_training_summary",
    "read_training_summary",
]

log = get_logger("training.checkpoint")

CHECKPOINT_PREFIX = "checkpoint"
TRAINING_SUMMARY_FILE = "training_summary.json"
_CHECKPOINT_RE = re.compile(rf"^{CHECKPOINT_PREFIX}-(\d+)$")


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be found, read or written."""


# --------------------------------------------------------------------------- #
# finding checkpoints
# --------------------------------------------------------------------------- #
def list_checkpoints(output_dir: Union[str, Path]) -> List[Path]:
    """Checkpoint directories under ``output_dir``, oldest step first."""
    root = Path(output_dir).expanduser()
    if not root.is_dir():
        return []
    found = []
    for entry in root.iterdir():
        match = _CHECKPOINT_RE.match(entry.name)
        if entry.is_dir() and match:
            found.append((int(match.group(1)), entry))
    return [path for _, path in sorted(found)]


def find_last_checkpoint(output_dir: Union[str, Path]) -> Optional[Path]:
    """The highest-numbered checkpoint under ``output_dir``, if any."""
    checkpoints = list_checkpoints(output_dir)
    return checkpoints[-1] if checkpoints else None


def resolve_resume(
    requested: Optional[str], output_dir: Union[str, Path]
) -> Optional[str]:
    """Turn a ``resume_from_checkpoint`` setting into a path, or None.

    ``""``/``None``/``false`` start fresh; ``auto``/``true``/``last`` take the
    newest checkpoint in ``output_dir`` (starting fresh if there is none); any
    other value is a path, which must exist.
    """
    if requested is None:
        return None
    value = str(requested).strip()
    if not value or value.lower() in {"false", "no", "none"}:
        return None

    if value.lower() in {"auto", "true", "yes", "last", "latest"}:
        last = find_last_checkpoint(output_dir)
        if last is None:
            log.info("resume requested, but %s holds no checkpoint; starting fresh", output_dir)
            return None
        log.info("resuming from the newest checkpoint: %s", last)
        return str(last)

    path = Path(value).expanduser()
    if not path.is_dir():
        raise CheckpointError(
            f"cannot resume: {path} is not a directory.\n"
            f"Use training.resume_from_checkpoint: auto to take the newest "
            f"checkpoint under {output_dir}, or point it at a checkpoint-<step> "
            "directory."
        )
    if not (path / "adapter_model.safetensors").exists() and not (
        path / "adapter_model.bin"
    ).exists():
        log.warning(
            "%s has no adapter_model file; resuming may fail if it is not a "
            "PEFT checkpoint",
            path,
        )
    log.info("resuming from %s", path)
    return str(path)


# --------------------------------------------------------------------------- #
# the final artifact
# --------------------------------------------------------------------------- #
def save_training_artifact(
    model: Any,
    tokenizer: Any,
    adapters_dir: Union[str, Path],
    *,
    language: str = "java",
    base_model: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    summary: Optional[Dict[str, Any]] = None,
    guard: Any = None,
) -> Path:
    """Write the final adapter artifact: weights, tokenizer, metadata, summary.

    Only the adapter is saved — the base checkpoint is never copied or merged
    (see :mod:`repairllama.model.lora`).  Returns the directory written.
    """
    from repairllama.model.lora import save_adapter

    target = save_adapter(
        model,
        adapters_dir,
        language=language,
        base_model=base_model,
        metadata=metadata,
        guard=guard,
    )

    if tokenizer is not None:
        # The tokenizer travels with the adapter: an adapter trained after
        # <FILL_ME> was added to the vocabulary is unusable with the base
        # tokenizer alone.
        tokenizer.save_pretrained(str(target))
        log.info("saved the tokenizer alongside the adapter (%d tokens)", len(tokenizer))

    if summary is not None:
        write_training_summary(target / TRAINING_SUMMARY_FILE, summary)
    return target


def write_training_summary(path: Union[str, Path], summary: Dict[str, Any]) -> Path:
    """Write ``training_summary.json``, creating parents as needed."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    payload.setdefault(
        "written_at", datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    target.write_text(
        json.dumps(payload, indent=2, default=str, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    log.info("wrote the training summary to %s", target)
    return target


def read_training_summary(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a training summary written by :func:`write_training_summary`."""
    target = Path(path).expanduser()
    if target.is_dir():
        target = target / TRAINING_SUMMARY_FILE
    if not target.is_file():
        raise CheckpointError(f"no training summary at {target}")
    return json.loads(target.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# run directories
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunPaths:
    """Where one training run writes its checkpoints and its artifact."""

    run_dir: Path
    adapters_dir: Path
    language: str = "java"

    @property
    def artifact_dir(self) -> Path:
        from repairllama.model.lora import profile_for

        return self.adapters_dir / profile_for(self.language).directory

    def prepare(self) -> "RunPaths":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        return self

    def to_dict(self) -> Dict[str, str]:
        return {
            "run_dir": str(self.run_dir),
            "adapters_dir": str(self.adapters_dir),
            "artifact_dir": str(self.artifact_dir),
        }


def run_directory(
    output_dir: Union[str, Path], experiment_name: str, timestamped: bool = False
) -> Path:
    """``<output_dir>/training/<experiment>`` — where checkpoints accumulate."""
    root = Path(output_dir).expanduser() / "training"
    name = experiment_name or "run"
    if timestamped:
        name = f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return root / name
