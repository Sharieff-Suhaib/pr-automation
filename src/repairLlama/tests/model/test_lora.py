"""The Java repair LoRA adapter: the paper's config, its assertions, its layout.

These are integration tests against a real PEFT model — the properties being
checked (base frozen, only LoRA trainable, nothing merged) are properties of
the wrapped model, and a mock would only test the mock.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repairllama.config import RepairConfig
from repairllama.model.adapter import (
    AdapterError,
    BaseWeightGuard,
    LoRASettings,
    is_adapter_parameter,
    trainable_parameter_names,
    unload_adapter,
)
from repairllama.model.loader import LoadOptions, count_parameters, load_base_model
from repairllama.model.lora import (
    ADAPTER_METADATA_FILE,
    ADAPTER_PROFILES,
    JAVA_REPAIR,
    PLANNED_LANGUAGES,
    LoRAAssertionError,
    adapted_modules,
    adapter_directory,
    assert_java_repair_lora,
    attach_java_repair_adapter,
    available_languages,
    check_java_repair_lora,
    create_lora_config,
    load_java_repair_adapter,
    print_trainable_parameters,
    profile_for,
    save_adapter,
)

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
peft = pytest.importorskip("peft")


@pytest.fixture()
def adapted(tiny_model):
    """A tiny model with the Java repair adapter attached."""
    return attach_java_repair_adapter(tiny_model.model)


# --------------------------------------------------------------------------- #
# the paper's configuration
# --------------------------------------------------------------------------- #
def test_profile_matches_the_paper() -> None:
    settings = JAVA_REPAIR.settings
    assert settings.r == 8
    assert settings.alpha == 16
    assert settings.dropout == 0.05
    assert list(settings.target_modules) == ["q_proj", "v_proj"]
    assert settings.bias == "none"
    assert settings.task_type == "CAUSAL_LM"


def test_the_shipped_config_matches_the_paper(shipped_config: RepairConfig) -> None:
    """configs/java_repair.yaml must not drift from the paper's values."""
    lora = shipped_config.model.lora
    assert (lora.r, lora.alpha, lora.dropout) == (8, 16, 0.05)
    assert lora.target_modules == ["q_proj", "v_proj"]


def test_config_defaults_match_the_profile() -> None:
    settings = LoRASettings.from_config(RepairConfig.default().model.lora)
    assert settings == JAVA_REPAIR.settings


def test_create_lora_config_uses_the_paper_values() -> None:
    config = create_lora_config()
    assert config.r == 8
    assert config.lora_alpha == 16
    assert config.lora_dropout == 0.05
    assert sorted(config.target_modules) == ["q_proj", "v_proj"]
    assert config.task_type == "CAUSAL_LM"
    assert config.inference_mode is False


def test_create_lora_config_accepts_overrides() -> None:
    config = create_lora_config(r=16, target_modules=["q_proj"])
    assert config.r == 16
    assert list(config.target_modules) == ["q_proj"]


def test_create_lora_config_validates_overrides() -> None:
    with pytest.raises(AdapterError):
        create_lora_config(r=0)


def test_create_lora_config_for_inference() -> None:
    assert create_lora_config(inference_mode=True).inference_mode is True


# --------------------------------------------------------------------------- #
# attaching
# --------------------------------------------------------------------------- #
def test_attach_returns_a_peft_model(adapted) -> None:
    assert hasattr(adapted, "peft_config")
    assert "java-repair" in adapted.peft_config


def test_attach_adapts_exactly_q_proj_and_v_proj(adapted) -> None:
    assert adapted_modules(adapted) == ("q_proj", "v_proj")


def test_attached_lora_layers_exist_on_q_and_v_only(adapted) -> None:
    adapted_names = {
        name.split(".self_attn.")[1].split(".lora_")[0]
        for name in trainable_parameter_names(adapted)
        if ".self_attn." in name
    }
    assert adapted_names == {"q_proj", "v_proj"}


def test_attach_runs_the_assertions(tiny_model) -> None:
    """A target that matches nothing must fail at attach time, not in training."""
    with pytest.raises(AdapterError):
        attach_java_repair_adapter(
            tiny_model.model, LoRASettings(target_modules=("not_a_module",))
        )


def test_attach_verifies_the_base_weights_when_given_a_guard(tiny_model) -> None:
    guard = BaseWeightGuard.capture(tiny_model.model, sample=0)
    model = attach_java_repair_adapter(tiny_model.model, guard=guard)
    guard.verify(model, context="the test")


def test_attach_uses_the_java_repair_adapter_name(adapted) -> None:
    assert sorted(adapted.peft_config) == ["java-repair"]


# --------------------------------------------------------------------------- #
# the four assertions
# --------------------------------------------------------------------------- #
def test_all_four_checks_pass(adapted) -> None:
    checks = check_java_repair_lora(adapted)
    assert len(checks) == 4
    assert all(check.ok for check in checks), [c.to_dict() for c in checks if not c.ok]


def test_assertion_1_base_parameters_are_frozen(adapted) -> None:
    leaked = [
        name
        for name, parameter in adapted.named_parameters()
        if parameter.requires_grad and not is_adapter_parameter(name)
    ]
    assert leaked == []


def test_assertion_2_lora_parameters_are_trainable(adapted) -> None:
    trainable = trainable_parameter_names(adapted)
    assert trainable
    assert all(is_adapter_parameter(name) for name in trainable)
    assert count_parameters(adapted).trainable > 0


def test_assertion_3_target_modules(adapted) -> None:
    check = next(c for c in check_java_repair_lora(adapted) if "target modules" in c.name)
    assert check.ok
    assert "q_proj" in check.detail and "v_proj" in check.detail


def test_assertion_4_trainable_is_dramatically_smaller(adapted) -> None:
    counts = count_parameters(adapted)
    assert counts.trainable < counts.total * 0.05
    assert counts.total / counts.trainable > 20


def test_assert_raises_when_the_base_is_trainable(adapted) -> None:
    for name, parameter in adapted.named_parameters():
        if not is_adapter_parameter(name):
            parameter.requires_grad_(True)
            break
    with pytest.raises(LoRAAssertionError, match="base parameters are frozen"):
        assert_java_repair_lora(adapted)


def test_assert_raises_when_nothing_is_trainable(adapted) -> None:
    for parameter in adapted.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(LoRAAssertionError, match="LoRA parameters are trainable"):
        assert_java_repair_lora(adapted)


def test_assert_raises_on_the_wrong_target_modules(tiny_model) -> None:
    model = attach_java_repair_adapter(
        tiny_model.model, LoRASettings(r=8, target_modules=("q_proj", "k_proj")), verify=False
    )
    with pytest.raises(LoRAAssertionError, match="target modules"):
        assert_java_repair_lora(model)


def test_assert_raises_when_too_much_is_trainable(adapted) -> None:
    with pytest.raises(LoRAAssertionError, match="small fraction"):
        assert_java_repair_lora(adapted, max_trainable_fraction=0.0001)


def test_assert_returns_the_checks_on_success(adapted) -> None:
    checks = assert_java_repair_lora(adapted)
    assert [check.ok for check in checks] == [True] * 4


def test_a_plain_model_fails_the_checks(tiny_model) -> None:
    checks = check_java_repair_lora(tiny_model.model)
    assert not all(check.ok for check in checks)
    with pytest.raises(LoRAAssertionError):
        assert_java_repair_lora(tiny_model.model)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def test_print_trainable_parameters(adapted) -> None:
    lines: list[str] = []
    summary = print_trainable_parameters(adapted, printer=lines.append)

    counts = count_parameters(adapted)
    assert summary.total == counts.total
    assert summary.trainable == counts.trainable
    assert summary.frozen == counts.total - counts.trainable
    assert summary.adapter_parameters == counts.trainable
    assert summary.adapted_modules == ("q_proj", "v_proj")
    assert summary.ratio > 20

    assert any("trainable params:" in line for line in lines)
    assert any("q_proj" in line for line in lines)


def test_summary_serialises(adapted) -> None:
    payload = print_trainable_parameters(adapted).to_dict()
    assert payload["trainable"] > 0
    assert payload["adapted_modules"] == ["q_proj", "v_proj"]


def test_summary_render_reads_like_pefts(adapted) -> None:
    text = print_trainable_parameters(adapted).render()
    assert "trainable params:" in text
    assert "all params:" in text
    assert "trainable%:" in text


# --------------------------------------------------------------------------- #
# saving — adapter only, never merged
# --------------------------------------------------------------------------- #
def test_save_writes_to_the_java_repair_directory(adapted, tmp_path: Path) -> None:
    written = save_adapter(adapted, tmp_path / "adapters")
    assert written == tmp_path / "adapters" / "java-repair"
    assert (written / "adapter_config.json").is_file()
    assert (written / "adapter_model.safetensors").is_file()


def test_save_does_not_write_the_base_model(adapted, tmp_path: Path) -> None:
    """The base checkpoint must never be copied into an adapter directory."""
    written = save_adapter(adapted, tmp_path / "adapters")
    names = {entry.name for entry in written.iterdir()}
    assert "config.json" not in names
    assert "model.safetensors" not in names
    assert not any(name.startswith("pytorch_model") for name in names)


def test_saved_adapter_is_small(adapted, tmp_path: Path) -> None:
    written = save_adapter(adapted, tmp_path / "adapters")
    weights = (written / "adapter_model.safetensors").stat().st_size
    total_bytes = sum(p.numel() * p.element_size() for p in adapted.parameters())
    assert weights < total_bytes * 0.2


def test_save_records_metadata(adapted, tmp_path: Path) -> None:
    written = save_adapter(adapted, tmp_path / "adapters", base_model="codellama/x")
    payload = json.loads((written / ADAPTER_METADATA_FILE).read_text(encoding="utf-8"))
    assert payload["language"] == "java"
    assert payload["directory"] == "java-repair"
    assert payload["base_model"] == "codellama/x"
    assert payload["merged_into_base"] is False
    assert payload["lora"]["r"] == 8
    assert payload["lora"]["target_modules"] == ["q_proj", "v_proj"]
    assert payload["trainable_parameters"] > 0


def test_save_accepts_extra_metadata(adapted, tmp_path: Path) -> None:
    written = save_adapter(
        adapted, tmp_path / "adapters", metadata={"epochs": 3, "dataset": "megadiff"}
    )
    payload = json.loads((written / ADAPTER_METADATA_FILE).read_text(encoding="utf-8"))
    assert payload["metadata"]["epochs"] == 3


def test_save_verifies_the_base_is_unmerged(tiny_model, tmp_path: Path) -> None:
    guard = BaseWeightGuard.capture(tiny_model.model, sample=0)
    model = attach_java_repair_adapter(tiny_model.model, guard=guard)
    written = save_adapter(model, tmp_path / "adapters", guard=guard)
    assert written.is_dir()


def test_save_refuses_a_model_without_an_adapter(tiny_model, tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="expects a PEFT-wrapped model"):
        save_adapter(tiny_model.model, tmp_path / "adapters")


def test_base_weights_are_unchanged_across_the_whole_lifecycle(
    tiny_model, tmp_path: Path
) -> None:
    """attach -> save -> unload must leave the base model bit-for-bit identical."""
    guard = BaseWeightGuard.capture(tiny_model.model, sample=0)
    model = attach_java_repair_adapter(tiny_model.model, guard=guard)
    save_adapter(model, tmp_path / "adapters")
    guard.verify(model, context="saving")
    base = unload_adapter(model)
    guard.verify(base, context="unloading")
    assert not hasattr(base, "peft_config")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def test_save_and_load_round_trip(tiny_checkpoint: Path, tmp_path: Path) -> None:
    def fresh():
        return load_base_model(
            LoadOptions(base_model=str(tiny_checkpoint), device="cpu", dtype="float32")
        ).model

    trained = attach_java_repair_adapter(fresh())
    save_adapter(trained, tmp_path / "adapters")

    reloaded = load_java_repair_adapter(fresh(), tmp_path / "adapters")
    assert sorted(reloaded.peft_config) == ["java-repair"]
    assert adapted_modules(reloaded) == ("q_proj", "v_proj")


def test_loaded_adapter_weights_match_what_was_saved(
    tiny_checkpoint: Path, tmp_path: Path
) -> None:
    def fresh():
        return load_base_model(
            LoadOptions(base_model=str(tiny_checkpoint), device="cpu", dtype="float32")
        ).model

    trained = attach_java_repair_adapter(fresh())
    with torch.no_grad():  # stand in for training: change the adapter only
        for name, parameter in trained.named_parameters():
            if "lora_B" in name:
                parameter.add_(0.5)
    expected = {
        name: parameter.detach().clone()
        for name, parameter in trained.named_parameters()
        if "lora_B" in name
    }
    save_adapter(trained, tmp_path / "adapters")

    reloaded = load_java_repair_adapter(fresh(), tmp_path / "adapters")
    actual = {name: p for name, p in reloaded.named_parameters() if "lora_B" in name}
    assert set(actual) == set(expected)
    for name, tensor in expected.items():
        assert torch.allclose(actual[name], tensor)


def test_loaded_adapter_is_inference_only_by_default(
    tiny_checkpoint: Path, tmp_path: Path
) -> None:
    def fresh():
        return load_base_model(
            LoadOptions(base_model=str(tiny_checkpoint), device="cpu", dtype="float32")
        ).model

    save_adapter(attach_java_repair_adapter(fresh()), tmp_path / "adapters")
    reloaded = load_java_repair_adapter(fresh(), tmp_path / "adapters")
    assert count_parameters(reloaded).trainable == 0

    trainable = load_java_repair_adapter(
        fresh(), tmp_path / "adapters", is_trainable=True
    )
    assert count_parameters(trainable).trainable > 0


def test_loading_needs_a_directory_or_a_path(tiny_model) -> None:
    with pytest.raises(AdapterError, match="adapters_dir or an explicit path"):
        load_java_repair_adapter(tiny_model.model)


def test_loading_a_missing_adapter_is_reported(tiny_model, tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="adapter directory not found"):
        load_java_repair_adapter(tiny_model.model, tmp_path / "empty")


def test_loading_from_an_explicit_path(tiny_checkpoint: Path, tmp_path: Path) -> None:
    def fresh():
        return load_base_model(
            LoadOptions(base_model=str(tiny_checkpoint), device="cpu", dtype="float32")
        ).model

    written = save_adapter(attach_java_repair_adapter(fresh()), tmp_path / "adapters")
    reloaded = load_java_repair_adapter(fresh(), path=written)
    assert sorted(reloaded.peft_config) == ["java-repair"]


# --------------------------------------------------------------------------- #
# the per-language layout
# --------------------------------------------------------------------------- #
def test_java_is_the_only_implemented_language() -> None:
    assert available_languages() == ["java"]
    assert set(ADAPTER_PROFILES) == {"java"}


def test_adapter_directory_layout(tmp_path: Path) -> None:
    assert adapter_directory(tmp_path, "java") == tmp_path / "java-repair"


def test_adapter_directory_can_create(tmp_path: Path) -> None:
    path = adapter_directory(tmp_path / "adapters", "java", create=True)
    assert path.is_dir()


def test_planned_languages_are_reserved_but_not_implemented() -> None:
    assert PLANNED_LANGUAGES == {"python": "python-repair", "cpp": "cpp-repair"}
    for language in PLANNED_LANGUAGES:
        with pytest.raises(AdapterError, match="not implemented|no adapter is implemented"):
            profile_for(language)


def test_an_unknown_language_lists_what_exists() -> None:
    with pytest.raises(AdapterError) as info:
        profile_for("rust")
    message = str(info.value)
    assert "java" in message and "python" in message


def test_a_planned_language_names_its_reserved_directory() -> None:
    with pytest.raises(AdapterError, match="python-repair"):
        profile_for("python")


def test_profiles_are_language_specific() -> None:
    assert profile_for("java") is JAVA_REPAIR
    assert profile_for("JAVA") is JAVA_REPAIR
    assert JAVA_REPAIR.adapter_name == "java-repair"
    assert JAVA_REPAIR.to_dict()["lora"]["r"] == 8


def test_two_adapters_can_share_one_base(tiny_checkpoint: Path, tmp_path: Path) -> None:
    """The layout's premise: one frozen base, several named adapters."""
    from repairllama.model.adapter import load_adapter

    base = load_base_model(
        LoadOptions(base_model=str(tiny_checkpoint), device="cpu", dtype="float32")
    ).model
    guard = BaseWeightGuard.capture(base, sample=0)

    model = attach_java_repair_adapter(base)
    written = save_adapter(model, tmp_path / "adapters")
    model = load_adapter(model, written, adapter_name="second-repair")

    assert sorted(model.peft_config) == ["java-repair", "second-repair"]
    guard.verify(model, context="loading a second adapter")
