"""Device and dtype resolution, and the memory reporting helpers."""

from __future__ import annotations

import pytest

from repairllama.model.device import (
    DeviceError,
    DeviceSpec,
    available_devices,
    describe_device,
    format_bytes,
    format_count,
    memory_snapshot,
    resolve_device,
    resolve_device_spec,
    resolve_dtype,
    torch_available,
)

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------------- #
# devices
# --------------------------------------------------------------------------- #
def test_torch_is_detected() -> None:
    assert torch_available() is True


def test_auto_picks_an_available_device() -> None:
    assert resolve_device("auto") in available_devices()


def test_cpu_is_always_available() -> None:
    assert resolve_device("cpu") == "cpu"
    assert "cpu" in available_devices()


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(DeviceError, match="unknown device"):
        resolve_device("tpu")


def test_unavailable_device_is_refused_not_downgraded() -> None:
    """Asking for a GPU that is not there must fail loudly."""
    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA")
    with pytest.raises(DeviceError, match="no CUDA device"):
        resolve_device("cuda")


def test_cuda_index_is_validated() -> None:
    if not torch.cuda.is_available():
        with pytest.raises(DeviceError):
            resolve_device("cuda:3")
    else:
        with pytest.raises(DeviceError, match="only"):
            resolve_device(f"cuda:{torch.cuda.device_count() + 5}")


def test_mps_is_refused_when_unavailable() -> None:
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        assert resolve_device("mps") == "mps"
    else:
        with pytest.raises(DeviceError, match="Metal"):
            resolve_device("mps")


# --------------------------------------------------------------------------- #
# dtypes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["float32", "float16", "bfloat16"])
def test_explicit_dtypes_are_honoured(name: str) -> None:
    resolved, _ = resolve_dtype(name, "cpu")
    assert resolved == name


def test_unknown_dtype_is_rejected() -> None:
    with pytest.raises(DeviceError, match="unknown dtype"):
        resolve_dtype("float8", "cpu")


def test_auto_dtype_on_cpu_is_float32() -> None:
    resolved, notes = resolve_dtype("auto", "cpu")
    assert resolved == "float32"
    assert any("auto" in note for note in notes)


def test_auto_dtype_on_mps_is_float16() -> None:
    assert resolve_dtype("auto", "mps")[0] == "float16"


def test_auto_dtype_on_cuda_is_16_bit() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    assert resolve_dtype("auto", "cuda:0")[0] in {"bfloat16", "float16"}


def test_float16_on_cpu_warns_but_is_allowed() -> None:
    resolved, notes = resolve_dtype("float16", "cpu")
    assert resolved == "float16"
    assert any("slow" in note for note in notes)


def test_float32_notes_the_memory_cost() -> None:
    assert any("memory" in note for note in resolve_dtype("float32", "cpu")[1])


# --------------------------------------------------------------------------- #
# the resolved pair
# --------------------------------------------------------------------------- #
def test_spec_records_what_was_requested() -> None:
    spec = resolve_device_spec("cpu", "float16")
    assert spec.device == "cpu"
    assert spec.dtype_name == "float16"
    assert spec.requested_device == "cpu"
    assert spec.requested_dtype == "float16"
    assert spec.torch_dtype is torch.float16


def test_auto_spec_explains_itself() -> None:
    spec = resolve_device_spec("auto", "auto")
    assert spec.is_auto_device
    assert any("resolved to" in note for note in spec.notes)


def test_spec_device_type_and_index() -> None:
    spec = DeviceSpec(device="cuda:2", dtype_name="float16")
    assert spec.device_type == "cuda"
    assert spec.index == 2
    assert DeviceSpec(device="cpu", dtype_name="float32").index is None


def test_spec_serialises() -> None:
    payload = resolve_device_spec("cpu", "bfloat16").to_dict()
    assert payload["device"] == "cpu"
    assert payload["dtype"] == "bfloat16"


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "unknown"), (512, "512 B"), (2048, "2.0 KiB"), (13 * 1024**3, "13.0 GiB")],
)
def test_format_bytes(value, expected: str) -> None:
    assert format_bytes(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(42, "42"), (1500, "1.5K"), (6_700_000_000, "6.7B"), (40_000_000, "40.0M")],
)
def test_format_count(value: int, expected: str) -> None:
    assert format_count(value) == expected


def test_describe_device_reports_torch_and_a_name() -> None:
    info = describe_device("cpu")
    assert info["type"] == "cpu"
    assert info["torch"] == torch.__version__
    assert "name" in info


def test_memory_snapshot_is_safe_on_any_device() -> None:
    snapshot = memory_snapshot(resolve_device("auto"))
    assert "device" in snapshot


def test_log_device_summary_does_not_raise(caplog: pytest.LogCaptureFixture) -> None:
    from repairllama.model.device import log_device_summary

    with caplog.at_level("INFO", logger="repairllama.model.device"):
        log_device_summary(resolve_device_spec("cpu", "float32"))
    assert any("device:" in record.message for record in caplog.records)
