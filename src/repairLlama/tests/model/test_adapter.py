"""LoRA settings, the base-weight guard, and the four adapter operations.

The tests that need PEFT are skipped when it is not installed; everything
that guards the base model works on plain torch and always runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tiny_checkpoint as tc

from repairllama.config import RepairConfig
from repairllama.model.adapter import (
    AdapterError,
    BaseWeightGuard,
    BaseWeightsModified,
    LoRASettings,
    adapter_state,
    attach_lora,
    is_adapter_parameter,
    load_adapter,
    save_adapter,
    trainable_parameter_names,
    unload_adapter,
)

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

def _has_peft() -> bool:
    try:
        import peft  # noqa: F401
    except ImportError:
        return False
    return True


requires_peft = pytest.mark.skipif(not _has_peft(), reason="peft is not installed")


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #
def test_settings_from_config() -> None:
    cfg = RepairConfig.default().model.lora
    settings = LoRASettings.from_config(cfg)
    assert settings.r == cfg.r
    assert settings.alpha == cfg.alpha
    assert list(settings.target_modules) == cfg.target_modules


@pytest.mark.parametrize(
    "kwargs",
    [{"r": 0}, {"alpha": 0}, {"dropout": 1.0}, {"dropout": -0.1}, {"target_modules": ()}],
)
def test_invalid_settings_are_rejected(kwargs: dict) -> None:
    with pytest.raises(AdapterError):
        LoRASettings(**kwargs)


def test_settings_serialise() -> None:
    payload = LoRASettings().to_dict()
    assert payload["r"] == 16
    assert "q_proj" in payload["target_modules"]


# --------------------------------------------------------------------------- #
# telling adapter parameters from base ones
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight", True),
        ("base_model.model.layers.0.self_attn.q_proj.base_layer.weight", False),
        ("model.embed_tokens.weight", False),
        ("modules_to_save.default.weight", True),
    ],
)
def test_adapter_parameters_are_recognised(name: str, expected: bool) -> None:
    assert is_adapter_parameter(name) is expected


def test_trainable_names_are_empty_for_a_frozen_model(tiny_model) -> None:
    assert trainable_parameter_names(tiny_model.model) == []


# --------------------------------------------------------------------------- #
# the base-weight guard
# --------------------------------------------------------------------------- #
def test_guard_passes_when_nothing_changes(tiny_model) -> None:
    guard = BaseWeightGuard.capture(tiny_model.model)
    guard.verify(tiny_model.model)
    assert guard.changed(tiny_model.model) == []


def test_guard_detects_a_modified_weight(tiny_model) -> None:
    model = tiny_model.model
    guard = BaseWeightGuard.capture(model, sample=0)  # every tensor
    with torch.no_grad():
        next(iter(model.parameters())).add_(1.0)
    changed = guard.changed(model)
    assert changed
    with pytest.raises(BaseWeightsModified, match="changed"):
        guard.verify(model, context="the test")


def test_guard_samples_large_models_deterministically(tiny_model) -> None:
    first = BaseWeightGuard.capture(tiny_model.model, sample=5)
    second = BaseWeightGuard.capture(tiny_model.model, sample=5)
    assert list(first.fingerprints) == list(second.fingerprints)
    assert first.fingerprints == second.fingerprints
    assert len(first.fingerprints) <= 6  # sample plus both endpoints


def test_guard_records_the_parameter_total(tiny_model) -> None:
    guard = BaseWeightGuard.capture(tiny_model.model)
    assert guard.total_parameters == tiny_model.parameter_counts().total


def test_guard_needs_base_parameters() -> None:
    class Empty(torch.nn.Module):
        pass

    with pytest.raises(AdapterError, match="no base parameters"):
        BaseWeightGuard.capture(Empty())


# --------------------------------------------------------------------------- #
# operations that work without peft
# --------------------------------------------------------------------------- #
def test_unloading_a_model_without_an_adapter_returns_it(tiny_model) -> None:
    assert unload_adapter(tiny_model.model) is tiny_model.model


def test_adapter_state_of_a_plain_model(tiny_model) -> None:
    state = adapter_state(tiny_model.model)
    assert state["is_peft"] is False
    assert state["adapters"] == []
    assert state["parameters"]["trainable"] == 0


def test_saving_requires_a_peft_model(tiny_model, tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="expects a PEFT-wrapped model"):
        save_adapter(tiny_model.model, tmp_path / "adapter")


def test_loading_a_missing_adapter_directory(tiny_model, tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="adapter directory not found"):
        load_adapter(tiny_model.model, tmp_path / "absent")


def test_loading_a_directory_that_is_not_an_adapter(tiny_model, tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AdapterError, match="adapter_config.json"):
        load_adapter(tiny_model.model, tmp_path)


@pytest.mark.skipif(_has_peft(), reason="peft is installed")
def test_missing_peft_explains_how_to_install_it(tiny_model) -> None:
    with pytest.raises(AdapterError, match="peft is required"):
        attach_lora(tiny_model.model, LoRASettings(target_modules=("q_proj",)))


# --------------------------------------------------------------------------- #
# operations that need peft
# --------------------------------------------------------------------------- #
@requires_peft
def test_attach_lora_makes_only_the_adapter_trainable(tiny_model) -> None:
    model = attach_lora(tiny_model.model, LoRASettings(r=4, target_modules=("q_proj",)))
    trainable = trainable_parameter_names(model)
    assert trainable
    assert all(is_adapter_parameter(name) for name in trainable)


@requires_peft
def test_attach_lora_leaves_the_base_weights_untouched(tiny_model) -> None:
    guard = BaseWeightGuard.capture(tiny_model.model, sample=0)
    model = attach_lora(tiny_model.model, LoRASettings(r=4, target_modules=("q_proj",)))
    guard.verify(model, context="attaching LoRA")


@requires_peft
def test_attach_lora_trains_a_small_fraction(tiny_model) -> None:
    from repairllama.model.loader import count_parameters

    model = attach_lora(tiny_model.model, LoRASettings(r=4, target_modules=("q_proj",)))
    counts = count_parameters(model)
    assert 0 < counts.trainable < counts.total * 0.5


@requires_peft
def test_target_modules_that_match_nothing_are_reported(tiny_model) -> None:
    with pytest.raises(AdapterError):
        attach_lora(tiny_model.model, LoRASettings(target_modules=("not_a_module",)))


@requires_peft
def test_unload_restores_the_base_model(tiny_model) -> None:
    from repairllama.model.loader import count_parameters

    guard = BaseWeightGuard.capture(tiny_model.model, sample=0)
    baseline = count_parameters(tiny_model.model).total

    model = attach_lora(tiny_model.model, LoRASettings(r=4, target_modules=("q_proj",)))
    base = unload_adapter(model)

    assert count_parameters(base).total == baseline
    guard.verify(base, context="unloading the adapter")


@requires_peft
def test_save_and_load_round_trip(tiny_model, tmp_path: Path) -> None:
    model = attach_lora(tiny_model.model, LoRASettings(r=4, target_modules=("q_proj",)))
    saved = save_adapter(model, tmp_path / "adapter")
    assert (saved / "default" / "adapter_config.json").is_file() or (
        saved / "adapter_config.json"
    ).is_file()


@requires_peft
def test_load_adapter_from_disk(tiny_checkpoint: Path, tmp_path: Path) -> None:
    from repairllama.model.loader import LoadOptions, count_parameters, load_base_model

    adapter_dir = tc.build_adapter(tmp_path / "adapter", tiny_checkpoint)
    loaded = load_base_model(
        LoadOptions(base_model=str(tiny_checkpoint), device="cpu", dtype="float32")
    )
    guard = BaseWeightGuard.capture(loaded.model, sample=0)

    model = load_adapter(loaded.model, adapter_dir)
    assert adapter_state(model)["is_peft"] is True
    guard.verify(model, context="loading an adapter")
    # loaded adapters are inference-only unless asked otherwise
    assert count_parameters(model).trainable == 0


@requires_peft
def test_loaded_adapter_can_be_trainable(tiny_checkpoint: Path, tmp_path: Path) -> None:
    from repairllama.model.loader import LoadOptions, count_parameters, load_base_model

    adapter_dir = tc.build_adapter(tmp_path / "adapter", tiny_checkpoint)
    loaded = load_base_model(
        LoadOptions(base_model=str(tiny_checkpoint), device="cpu", dtype="float32")
    )
    model = load_adapter(loaded.model, adapter_dir, is_trainable=True)
    assert count_parameters(model).trainable > 0
    assert all(is_adapter_parameter(name) for name in trainable_parameter_names(model))


@requires_peft
def test_adapter_state_reports_the_lora_settings(tiny_model) -> None:
    model = attach_lora(tiny_model.model, LoRASettings(r=8, target_modules=("q_proj",)))
    state = adapter_state(model)
    assert state["is_peft"] is True
    assert state["lora"]["r"] == 8
    assert "q_proj" in state["lora"]["target_modules"]
