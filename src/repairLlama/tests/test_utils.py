"""Utilities: logging setup, IO round-trips, paths and seeding."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from repairllama.config import RepairConfig
from repairllama.utils import io as rio
from repairllama.utils.logging import (
    ROOT_LOGGER_NAME,
    JsonFormatter,
    configure_from_config,
    configure_logging,
    get_logger,
)
from repairllama.utils.paths import ensure_dir, project_root
from repairllama.utils.seed import set_seed


def test_get_logger_is_namespaced() -> None:
    assert get_logger("data").name == f"{ROOT_LOGGER_NAME}.data"
    assert get_logger().name == ROOT_LOGGER_NAME
    assert get_logger("repairllama.cli").name == "repairllama.cli"


def test_configure_logging_writes_a_file(tmp_path: Path) -> None:
    configure_logging(level="DEBUG", log_dir=tmp_path, file_name="t.log", force=True)
    get_logger("test").debug("hello from the test")
    logging.getLogger(ROOT_LOGGER_NAME).handlers[-1].flush()
    assert "hello from the test" in (tmp_path / "t.log").read_text(encoding="utf-8")


def test_configure_logging_is_idempotent(tmp_path: Path) -> None:
    configure_logging(level="INFO", log_dir=tmp_path, force=True)
    before = len(logging.getLogger(ROOT_LOGGER_NAME).handlers)
    configure_logging(level="INFO", log_dir=tmp_path)  # no force: ignored
    assert len(logging.getLogger(ROOT_LOGGER_NAME).handlers) == before


def test_json_formatter_emits_one_object() -> None:
    record = logging.LogRecord(
        "repairllama.test", logging.INFO, __file__, 1, "patched %s", ("Foo.java",), None
    )
    record.bug_id = "Chart-1"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "patched Foo.java"
    assert payload["level"] == "INFO"
    assert payload["bug_id"] == "Chart-1"


def test_configure_from_config(tmp_path: Path) -> None:
    cfg = RepairConfig.from_dict(
        {"paths": {"project_root": str(tmp_path)}, "logging": {"format": "json"}}
    )
    configure_from_config(cfg)
    assert (tmp_path / "outputs" / "logs").is_dir()


def test_jsonl_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "rows.jsonl"
    rows = [{"bug_id": f"B-{i}", "buggy": "int x = 1;"} for i in range(3)]
    assert rio.write_jsonl(target, rows) == 3
    assert list(rio.read_jsonl(target)) == rows
    assert rio.count_lines(target) == 3


def test_json_round_trip(tmp_path: Path) -> None:
    target = rio.write_json(tmp_path / "cfg.json", {"a": [1, 2]})
    assert rio.read_json(target) == {"a": [1, 2]}


def test_project_root_contains_pyproject() -> None:
    assert (project_root() / "pyproject.toml").is_file()


def test_ensure_dir(tmp_path: Path) -> None:
    created = ensure_dir(tmp_path / "a" / "b")
    assert created.is_dir()
    assert ensure_dir(created) == created  # idempotent


def test_set_seed_is_reproducible() -> None:
    import random

    seeded = set_seed(1234)
    assert "python" in seeded
    first = [random.random() for _ in range(5)]
    set_seed(1234)
    assert [random.random() for _ in range(5)] == first
