"""Logging setup shared by every stage of the pipeline.

One call to :func:`configure_logging` at process start (the CLI does this)
installs a console handler and, optionally, a rotating file handler.  Library
code should only ever call :func:`get_logger`.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = [
    "ROOT_LOGGER_NAME",
    "configure_logging",
    "get_logger",
    "log_section",
    "JsonFormatter",
]

ROOT_LOGGER_NAME = "repairllama"

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[1;31m",
}
_RESET = "\033[0m"

# Attributes LogRecord always carries; anything else was passed via `extra=`.
_STANDARD_ATTRS = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}

_configured = False


class ColorFormatter(logging.Formatter):
    """Console formatter that tints the level name when the stream is a TTY."""

    def __init__(self, fmt: str = _DEFAULT_FORMAT, datefmt: str = _DATE_FORMAT,
                 color: bool = True) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        if not self.color:
            return super().format(record)
        original = record.levelname
        tint = _LEVEL_COLORS.get(original)
        if tint:
            record.levelname = f"{tint}{original}{_RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original


class JsonFormatter(logging.Formatter):
    """One JSON object per line — convenient for training-run log shipping."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _supports_color(stream: Any) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def configure_logging(
    level: str = "INFO",
    log_dir: Optional[str | os.PathLike[str]] = None,
    file_name: str = "repairllama.log",
    fmt: str = "text",
    color: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Install handlers on the ``repairllama`` logger and return it.

    Repeat calls are ignored unless ``force`` is set, so importing a module
    twice never doubles up log lines.
    """
    global _configured
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    if _configured and not force:
        return logger

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    console = logging.StreamHandler(stream=sys.stderr)
    if fmt == "json":
        console.setFormatter(JsonFormatter())
    else:
        console.setFormatter(
            ColorFormatter(color=color and _supports_color(sys.stderr))
        )
    logger.addHandler(console)

    if log_dir:
        directory = Path(log_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            directory / file_name,
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            JsonFormatter() if fmt == "json"
            else logging.Formatter(_DEFAULT_FORMAT, _DATE_FORMAT)
        )
        logger.addHandler(file_handler)

    _configured = True
    return logger


def configure_from_config(cfg: Any) -> logging.Logger:
    """Configure logging from a :class:`repairllama.config.RepairConfig`."""
    log_dir = cfg.paths.resolve("log_dir") if cfg.logging.log_to_file else None
    return configure_logging(
        level=cfg.logging.level,
        log_dir=log_dir,
        file_name=cfg.logging.file_name,
        fmt=cfg.logging.format,
        color=cfg.logging.color,
        force=True,
    )


def get_logger(name: str = "") -> logging.Logger:
    """Return a child of the package logger, e.g. ``get_logger("data")``."""
    if not name or name == ROOT_LOGGER_NAME:
        return logging.getLogger(ROOT_LOGGER_NAME)
    if name.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    # Module names like "repairllama.data.loader" arrive as __name__.
    suffix = name.split(f"{ROOT_LOGGER_NAME}.", 1)[-1]
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{suffix}")


def log_section(logger: logging.Logger, title: str, width: int = 72) -> None:
    """Emit a visually separated banner — used between pipeline stages."""
    logger.info("=" * width)
    logger.info(title)
    logger.info("=" * width)
