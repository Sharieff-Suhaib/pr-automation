"""Configuration schema: defaults, YAML round-trip, overrides and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from repairllama.config import ConfigError, RepairConfig, load_config


def test_defaults_are_valid(default_config: RepairConfig) -> None:
    assert default_config.experiment_name == "java-repair"
    assert default_config.data.language == "java"
    assert default_config.model.lora.enabled is True


def test_shipped_config_loads(shipped_config: RepairConfig) -> None:
    assert shipped_config.model.base_model
    assert shipped_config.representation.input_format in {"ir1", "ir2", "ir3", "ir4"}
    assert shipped_config.evaluation.build_tool == "maven"


def test_shipped_config_matches_dataclass_defaults(shipped_config: RepairConfig) -> None:
    """The YAML file should stay in sync with the schema's own defaults."""
    defaults = RepairConfig.default().to_dict()
    shipped = shipped_config.to_dict()
    defaults.pop("paths")
    shipped.pop("paths")
    assert shipped == defaults


def test_yaml_round_trip(tmp_path: Path, default_config: RepairConfig) -> None:
    target = tmp_path / "round_trip.yaml"
    default_config.to_yaml(target)
    reloaded = RepairConfig.from_yaml(target)
    assert reloaded.to_dict()["training"] == default_config.to_dict()["training"]


def test_dotted_override(default_config: RepairConfig) -> None:
    updated = default_config.override({"training.epochs": 7, "model.lora.r": 32})
    assert updated.training.epochs == 7
    assert updated.model.lora.r == 32
    assert default_config.training.epochs == 2, "override must not mutate the original"


def test_unknown_override_key_rejected(default_config: RepairConfig) -> None:
    with pytest.raises(ConfigError):
        default_config.override({"training.nope": 1})


def test_unknown_yaml_key_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("training:\n  epochs: 2\n  mystery: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mystery"):
        RepairConfig.from_yaml(bad)


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        RepairConfig.from_yaml(tmp_path / "absent.yaml")


@pytest.mark.parametrize(
    "overrides",
    [
        {"data.train_split": 0.5},  # splits no longer sum to 1
        {"representation.input_format": "ir9"},
        {"localization.strategy": "magic"},
        {"training.learning_rate": 0},
        {"inference.top_p": 2.0},
        {"patching.strategy": "rewrite_everything"},
        {"evaluation.build_tool": "ant"},
        {"logging.level": "LOUD"},
        {"model.lora.dropout": 1.5},
    ],
)
def test_invalid_values_rejected(default_config: RepairConfig, overrides: dict) -> None:
    with pytest.raises(ConfigError):
        default_config.override(overrides)


def test_greedy_multi_candidate_combination_rejected(default_config: RepairConfig) -> None:
    with pytest.raises(ConfigError, match="greedy"):
        default_config.override({"inference.do_sample": False})


def test_logging_level_is_normalised() -> None:
    cfg = RepairConfig.from_dict({"logging": {"level": "debug"}})
    assert cfg.logging.level == "DEBUG"


def test_paths_resolve_against_project_root(tmp_path: Path) -> None:
    cfg = RepairConfig.from_dict({"paths": {"project_root": str(tmp_path)}})
    assert cfg.paths.resolve("splits_dir") == (tmp_path / "data/splits").resolve()
    with pytest.raises(ConfigError):
        cfg.paths.resolve("not_a_path_key")


def test_load_config_without_path_uses_defaults() -> None:
    cfg = load_config(None, {"experiment_name": "smoke"})
    assert cfg.experiment_name == "smoke"


def test_effective_batch_size(default_config: RepairConfig) -> None:
    training = default_config.training
    assert training.effective_batch_size == (
        training.batch_size * training.gradient_accumulation_steps
    )


def test_tokenizer_defaults_to_base_model(default_config: RepairConfig) -> None:
    assert default_config.model.tokenizer_name == default_config.model.base_model
