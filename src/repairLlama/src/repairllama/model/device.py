"""Device and dtype resolution, plus the memory reporting the loader logs.

Two decisions have to be made before a checkpoint is touched — *where* it goes
and *in what precision* — and both must be explicit, because getting them
wrong is the difference between a model that loads in 13 GB and one that does
not load at all.

    resolve_device("auto")   -> "cuda:0" | "mps" | "cpu"
    resolve_dtype("auto", d) -> torch.bfloat16 | torch.float16 | torch.float32

``auto`` picks the best available option; naming a device explicitly is
honoured or refused with a message that says why (asking for ``cuda`` on a
machine without a GPU is a mistake worth surfacing, not something to silently
downgrade).  Explicit *dtypes* are always honoured — a deliberate
``float32`` on CUDA is a legitimate debugging choice — but combinations that
are slow or unsupported get a warning.

``torch`` is imported inside the functions that need it, so this module can be
imported (and its error messages read) in an environment with no ML stack.
"""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from repairllama.utils.logging import get_logger

__all__ = [
    "DeviceError",
    "DeviceSpec",
    "DTYPE_NAMES",
    "torch_available",
    "available_devices",
    "resolve_device",
    "resolve_dtype",
    "resolve_device_spec",
    "describe_device",
    "memory_snapshot",
    "format_bytes",
    "format_count",
    "log_device_summary",
]

log = get_logger("model.device")

DTYPE_NAMES = ("auto", "float32", "float16", "bfloat16")
_DEVICE_NAMES = ("auto", "cpu", "cuda", "mps")


class DeviceError(RuntimeError):
    """Raised when a requested device or dtype cannot be honoured."""


def torch_available() -> bool:
    """True when torch can be imported, without raising if it cannot."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _require_torch(action: str):
    try:
        import torch
    except ImportError as exc:
        raise DeviceError(
            f"torch is required to {action}. Install the model stack with "
            "`pip install -e \".[train]\"` (or `pip install -r requirements.txt`)."
        ) from exc
    return torch


# --------------------------------------------------------------------------- #
# the resolved pair
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DeviceSpec:
    """A resolved device/dtype pair, plus how it was arrived at."""

    device: str  # "cpu", "mps", "cuda:0"
    dtype_name: str  # "float32" | "float16" | "bfloat16"
    requested_device: str = "auto"
    requested_dtype: str = "auto"
    notes: Tuple[str, ...] = ()

    @property
    def device_type(self) -> str:
        return self.device.split(":", 1)[0]

    @property
    def index(self) -> Optional[int]:
        _, _, suffix = self.device.partition(":")
        return int(suffix) if suffix.isdigit() else None

    @property
    def torch_dtype(self):
        """The :class:`torch.dtype` this spec names."""
        torch = _require_torch("resolve a dtype")
        return getattr(torch, self.dtype_name)

    @property
    def is_auto_device(self) -> bool:
        return self.requested_device == "auto"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "dtype": self.dtype_name,
            "requested_device": self.requested_device,
            "requested_dtype": self.requested_dtype,
            "notes": list(self.notes),
        }

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.device} / {self.dtype_name}"


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
def available_devices() -> List[str]:
    """Device names usable in this process, best first."""
    if not torch_available():
        return []
    import torch

    devices: List[str] = []
    if torch.cuda.is_available():
        devices.extend(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def resolve_device(preference: str = "auto") -> str:
    """Resolve a device preference to a concrete device string.

    ``auto`` prefers CUDA, then Apple MPS, then CPU.  A named device that is
    not available raises :class:`DeviceError` rather than falling back, so a
    training run cannot quietly end up on the CPU.
    """
    if preference not in _DEVICE_NAMES and not preference.startswith("cuda:"):
        raise DeviceError(
            f"unknown device {preference!r}; expected one of {list(_DEVICE_NAMES)} "
            "or 'cuda:<index>'"
        )
    torch = _require_torch("select a device")

    if preference == "auto":
        devices = available_devices()
        return devices[0] if devices else "cpu"

    if preference == "cpu":
        return "cpu"

    if preference == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise DeviceError(
                "device 'mps' was requested but Apple Metal is not available "
                f"(platform: {platform.platform()}). Use device: cpu, or 'auto'."
            )
        return "mps"

    # cuda / cuda:<index>
    if not torch.cuda.is_available():
        raise DeviceError(
            "device 'cuda' was requested but torch reports no CUDA device "
            f"(torch {torch.__version__}, CUDA build: {torch.version.cuda}). "
            "Use device: cpu, or 'auto' to pick what is available."
        )
    if preference == "cuda":
        return "cuda:0"
    index = int(preference.split(":", 1)[1])
    count = torch.cuda.device_count()
    if index >= count:
        raise DeviceError(
            f"device {preference!r} was requested but only {count} CUDA device(s) "
            "are visible (indices 0.." + str(max(count - 1, 0)) + ")"
        )
    return preference


def _supports_bfloat16(device: str) -> bool:
    torch = _require_torch("check bfloat16 support")
    if device.startswith("cuda"):
        checker = getattr(torch.cuda, "is_bf16_supported", None)
        return bool(checker()) if checker else False
    if device == "mps":
        return False  # bf16 support on Metal is patchy; prefer fp16 there
    return True  # CPU bf16 works, though it is slow


def resolve_dtype(name: str, device: str) -> Tuple[str, List[str]]:
    """Resolve a dtype name for ``device``; returns the name and any warnings.

    ``auto`` means: bfloat16 on CUDA when supported, float16 on CUDA without
    it and on MPS, float32 on CPU.  An explicit dtype is always honoured — the
    notes explain the cost when the combination is a poor one.
    """
    if name not in DTYPE_NAMES:
        raise DeviceError(f"unknown dtype {name!r}; expected one of {list(DTYPE_NAMES)}")
    notes: List[str] = []

    if name == "auto":
        if device.startswith("cuda"):
            resolved = "bfloat16" if _supports_bfloat16(device) else "float16"
        elif device == "mps":
            resolved = "float16"
        else:
            resolved = "float32"
        notes.append(f"dtype 'auto' resolved to {resolved} for {device}")
        return resolved, notes

    if name == "float16" and device == "cpu":
        notes.append(
            "float16 on CPU is slow and only partially supported; float32 or "
            "bfloat16 is usually the better choice"
        )
    if name == "bfloat16" and device.startswith("cuda") and not _supports_bfloat16(device):
        notes.append(
            "this GPU does not report bfloat16 support; expect emulation or errors"
        )
    if name == "bfloat16" and device == "mps":
        notes.append("bfloat16 support on Apple Metal is incomplete; float16 is safer")
    if name == "float32":
        notes.append("float32 doubles the memory a 16-bit load would need")
    return name, notes


def resolve_device_spec(device: str = "auto", dtype: str = "auto") -> DeviceSpec:
    """Resolve both halves at once, keeping what was asked for."""
    resolved_device = resolve_device(device)
    resolved_dtype, notes = resolve_dtype(dtype, resolved_device)
    if device == "auto":
        notes.insert(0, f"device 'auto' resolved to {resolved_device}")
    return DeviceSpec(
        device=resolved_device,
        dtype_name=resolved_dtype,
        requested_device=device,
        requested_dtype=dtype,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def format_bytes(value: Optional[float]) -> str:
    """Human-readable byte count (``13.5 GiB``)."""
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TiB"  # pragma: no cover - unreachable


def format_count(value: int) -> str:
    """Human-readable parameter count (``6.7B``, ``40.0M``)."""
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"{value / threshold:.1f}{suffix}"
    return str(value)


def describe_device(device: str) -> Dict[str, Any]:
    """Static facts about a device: name, capability, total memory."""
    info: Dict[str, Any] = {"device": device, "type": device.split(":", 1)[0]}
    if not torch_available():
        info["torch"] = None
        return info

    import torch

    info["torch"] = torch.__version__
    kind = info["type"]
    if kind == "cuda":
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        properties = torch.cuda.get_device_properties(index)
        info.update(
            {
                "name": properties.name,
                "capability": f"{properties.major}.{properties.minor}",
                "total_memory": properties.total_memory,
                "multi_processor_count": properties.multi_processor_count,
                "cuda": torch.version.cuda,
            }
        )
    elif kind == "mps":
        info.update({"name": f"Apple Metal ({platform.machine()})"})
        recommended = getattr(torch.mps, "recommended_max_memory", None)
        if recommended is not None:
            try:
                info["total_memory"] = recommended()
            except Exception:  # noqa: BLE001 - informational only
                pass
    else:
        info.update({"name": platform.processor() or platform.machine()})
        usage = shutil.disk_usage("/")
        info["disk_free"] = usage.free
        try:
            import os

            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            info["total_memory"] = pages * page_size
        except (ValueError, AttributeError, OSError):  # pragma: no cover - platform
            pass
    return info


def memory_snapshot(device: str) -> Dict[str, Any]:
    """Current memory use on ``device``, as far as torch can report it."""
    snapshot: Dict[str, Any] = {"device": device}
    if not torch_available():
        return snapshot

    import torch

    kind = device.split(":", 1)[0]
    if kind == "cuda":
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        free, total = torch.cuda.mem_get_info(index)
        snapshot.update(
            {
                "allocated": torch.cuda.memory_allocated(index),
                "reserved": torch.cuda.memory_reserved(index),
                "peak_allocated": torch.cuda.max_memory_allocated(index),
                "free": free,
                "total": total,
            }
        )
    elif kind == "mps" and hasattr(torch, "mps"):
        getter = getattr(torch.mps, "current_allocated_memory", None)
        if getter is not None:
            snapshot["allocated"] = getter()
        driver = getattr(torch.mps, "driver_allocated_memory", None)
        if driver is not None:
            snapshot["reserved"] = driver()
    return snapshot


def log_device_summary(spec: DeviceSpec, logger=None) -> None:
    """Log the resolved device, its capabilities and its current memory use."""
    logger = logger or log
    info = describe_device(spec.device)
    logger.info(
        "device: %s (%s) | dtype: %s",
        spec.device,
        info.get("name", "unknown"),
        spec.dtype_name,
    )
    if info.get("capability"):
        logger.info("compute capability: %s | CUDA %s", info["capability"], info.get("cuda"))
    if info.get("total_memory"):
        logger.info("device memory: %s total", format_bytes(info["total_memory"]))

    snapshot = memory_snapshot(spec.device)
    if "allocated" in snapshot:
        logger.info(
            "memory in use: %s allocated, %s reserved%s",
            format_bytes(snapshot.get("allocated")),
            format_bytes(snapshot.get("reserved")),
            (
                f", {format_bytes(snapshot['free'])} free"
                if snapshot.get("free") is not None
                else ""
            ),
        )
    for note in spec.notes:
        logger.info("note: %s", note)
