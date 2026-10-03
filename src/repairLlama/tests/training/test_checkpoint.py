"""Checkpoint discovery, resume resolution, and the final artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repairllama.training.checkpoint import (
    TRAINING_SUMMARY_FILE,
    CheckpointError,
    RunPaths,
    find_last_checkpoint,
    list_checkpoints,
    read_training_summary,
    resolve_resume,
    run_directory,
    save_training_artifact,
    write_training_summary,
)


def _checkpoint(root: Path, step: int) -> Path:
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_model.safetensors").write_bytes(b"")
    return path


# --------------------------------------------------------------------------- #
# finding checkpoints
# --------------------------------------------------------------------------- #
def test_checkpoints_are_ordered_numerically(tmp_path: Path) -> None:
    """checkpoint-100 is newer than checkpoint-20, which sorts wrong as text."""
    for step in (20, 5, 100):
        _checkpoint(tmp_path, step)
    assert [path.name for path in list_checkpoints(tmp_path)] == [
        "checkpoint-5",
        "checkpoint-20",
        "checkpoint-100",
    ]
    assert find_last_checkpoint(tmp_path).name == "checkpoint-100"


def test_no_checkpoints_is_not_an_error(tmp_path: Path) -> None:
    assert list_checkpoints(tmp_path) == []
    assert find_last_checkpoint(tmp_path) is None
    assert list_checkpoints(tmp_path / "absent") == []


def test_unrelated_directories_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir()
    (tmp_path / "checkpoint-notanumber").mkdir()
    _checkpoint(tmp_path, 3)
    assert [path.name for path in list_checkpoints(tmp_path)] == ["checkpoint-3"]


# --------------------------------------------------------------------------- #
# resuming
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", ["", None, "false", "no", "none"])
def test_no_resume_requested(value, tmp_path: Path) -> None:
    assert resolve_resume(value, tmp_path) is None


@pytest.mark.parametrize("value", ["auto", "true", "last", "latest", "AUTO"])
def test_auto_takes_the_newest_checkpoint(value: str, tmp_path: Path) -> None:
    _checkpoint(tmp_path, 10)
    newest = _checkpoint(tmp_path, 20)
    assert resolve_resume(value, tmp_path) == str(newest)


def test_auto_with_nothing_to_resume_starts_fresh(tmp_path: Path) -> None:
    assert resolve_resume("auto", tmp_path) is None


def test_an_explicit_path_is_used(tmp_path: Path) -> None:
    target = _checkpoint(tmp_path, 7)
    assert resolve_resume(str(target), tmp_path) == str(target)


def test_a_missing_resume_path_is_an_error(tmp_path: Path) -> None:
    """Silently starting over would waste far more than it saves."""
    with pytest.raises(CheckpointError, match="cannot resume"):
        resolve_resume(str(tmp_path / "checkpoint-999"), tmp_path)


def test_a_resume_path_without_adapter_files_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    empty = tmp_path / "checkpoint-1"
    empty.mkdir()
    with caplog.at_level("WARNING", logger="repairllama.training.checkpoint"):
        assert resolve_resume(str(empty), tmp_path) == str(empty)


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #
def test_summary_round_trip(tmp_path: Path) -> None:
    path = write_training_summary(tmp_path / "nested" / TRAINING_SUMMARY_FILE, {"a": 1})
    assert path.is_file()
    payload = read_training_summary(path)
    assert payload["a"] == 1
    assert "written_at" in payload


def test_summary_can_be_read_from_a_directory(tmp_path: Path) -> None:
    write_training_summary(tmp_path / TRAINING_SUMMARY_FILE, {"a": 2})
    assert read_training_summary(tmp_path)["a"] == 2


def test_a_missing_summary_is_reported(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="no training summary"):
        read_training_summary(tmp_path)


def test_summary_handles_non_serialisable_values(tmp_path: Path) -> None:
    path = write_training_summary(tmp_path / "s.json", {"path": tmp_path})
    assert json.loads(path.read_text())["path"] == str(tmp_path)


# --------------------------------------------------------------------------- #
# run paths
# --------------------------------------------------------------------------- #
def test_run_directory_layout(tmp_path: Path) -> None:
    assert run_directory(tmp_path, "java-repair") == tmp_path / "training" / "java-repair"


def test_timestamped_run_directory(tmp_path: Path) -> None:
    path = run_directory(tmp_path, "exp", timestamped=True)
    assert path.name.startswith("exp-")


def test_run_paths_expose_the_artifact_directory(tmp_path: Path) -> None:
    paths = RunPaths(run_dir=tmp_path / "run", adapters_dir=tmp_path / "adapters").prepare()
    assert paths.artifact_dir == tmp_path / "adapters" / "java-repair"
    assert paths.run_dir.is_dir()
    assert "artifact_dir" in paths.to_dict()


# --------------------------------------------------------------------------- #
# the artifact
# --------------------------------------------------------------------------- #
def test_artifact_holds_adapter_tokenizer_and_summary(
    tiny_checkpoint: Path, tiny_tokenizer, tmp_path: Path
) -> None:
    pytest.importorskip("peft")
    from repairllama.model.loader import LoadOptions, load_base_model
    from repairllama.model.lora import attach_java_repair_adapter

    loaded = load_base_model(
        LoadOptions(base_model=str(tiny_checkpoint), device="cpu", dtype="float32")
    )
    model = attach_java_repair_adapter(loaded.model)

    target = save_training_artifact(
        model,
        loaded.tokenizer,
        tmp_path / "adapters",
        base_model=str(tiny_checkpoint),
        summary={"experiment": "test"},
    )

    names = {entry.name for entry in target.iterdir()}
    assert "adapter_model.safetensors" in names
    assert "adapter_config.json" in names
    assert TRAINING_SUMMARY_FILE in names
    assert any(name.startswith("tokenizer") for name in names)
    # the base checkpoint is never part of the artifact
    assert "model.safetensors" not in names
    assert "config.json" not in names


def test_artifact_summary_is_readable(
    tiny_checkpoint: Path, tiny_tokenizer, tmp_path: Path
) -> None:
    pytest.importorskip("peft")
    from repairllama.model.loader import LoadOptions, load_base_model
    from repairllama.model.lora import attach_java_repair_adapter

    loaded = load_base_model(
        LoadOptions(base_model=str(tiny_checkpoint), device="cpu", dtype="float32")
    )
    model = attach_java_repair_adapter(loaded.model)
    target = save_training_artifact(
        model, loaded.tokenizer, tmp_path / "adapters", summary={"loss": 1.5}
    )
    assert read_training_summary(target)["loss"] == 1.5
