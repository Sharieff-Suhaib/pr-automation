"""Checkpoint resolution, loading, parameter counting — and no downloads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repairllama.config import RepairConfig
from repairllama.model.loader import (
    LoadOptions,
    ModelError,
    ModelNotAvailableError,
    count_parameters,
    describe_missing_model,
    freeze_parameters,
    load_base_model,
    resolve_model_source,
)

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


# --------------------------------------------------------------------------- #
# finding a checkpoint
# --------------------------------------------------------------------------- #
def test_a_local_directory_is_used_directly(tiny_checkpoint: Path) -> None:
    source = resolve_model_source(str(tiny_checkpoint))
    assert source.origin == "local_path"
    assert source.is_local
    assert source.reference == str(tiny_checkpoint)


def test_a_bare_name_is_found_in_the_models_dir(tiny_checkpoint: Path) -> None:
    source = resolve_model_source("tiny", models_dir=str(tiny_checkpoint.parent))
    assert source.origin == "models_dir"
    assert source.reference == str(tiny_checkpoint)


def test_a_repo_id_is_found_in_the_models_dir_by_basename(tiny_checkpoint: Path) -> None:
    """`org/tiny` should find `<models_dir>/tiny`, the usual unpack layout."""
    source = resolve_model_source("someorg/tiny", models_dir=str(tiny_checkpoint.parent))
    assert source.origin == "models_dir"


def test_a_directory_without_weights_is_not_a_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ModelNotAvailableError):
        resolve_model_source(str(tmp_path))


def test_missing_checkpoints_are_not_downloaded() -> None:
    with pytest.raises(ModelNotAvailableError) as info:
        resolve_model_source("codellama/CodeLlama-7b-hf", models_dir="/nonexistent")
    message = str(info.value)
    assert "downloading is disabled" in message
    assert "huggingface-cli download" in message
    assert "model.base_model" in message
    assert "model.allow_download" in message


def test_the_error_lists_where_it_looked() -> None:
    with pytest.raises(ModelNotAvailableError) as info:
        resolve_model_source("some/model", models_dir="/models")
    message = str(info.value)
    assert "/models/model" in message
    assert "HuggingFace cache" in message


def test_allow_download_passes_the_reference_through() -> None:
    source = resolve_model_source("some/model", allow_download=True)
    assert source.origin == "remote"
    assert source.reference == "some/model"
    assert not source.is_local


def test_an_empty_reference_is_a_configuration_error() -> None:
    with pytest.raises(ModelError, match="base_model is empty"):
        resolve_model_source("")


def test_describe_missing_model_names_the_target_directory() -> None:
    message = describe_missing_model("org/Model-7b", models_dir="/project/models")
    assert "/project/models/Model-7b" in message


# --------------------------------------------------------------------------- #
# options
# --------------------------------------------------------------------------- #
def test_options_from_config() -> None:
    cfg = RepairConfig.default()
    options = LoadOptions.from_config(cfg)
    assert options.base_model == cfg.model.base_model
    assert options.dtype == cfg.model.dtype
    assert options.allow_download is False
    assert options.local_files_only is True
    assert cfg.representation.fill_token in options.extra_tokens


def test_tokenizer_defaults_to_the_base_model() -> None:
    assert LoadOptions(base_model="x").tokenizer_source == "x"
    assert LoadOptions(base_model="x", tokenizer="y").tokenizer_source == "y"


def test_local_files_only_tracks_allow_download() -> None:
    assert LoadOptions(allow_download=True).local_files_only is False


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def test_load_returns_a_model_and_tokenizer(tiny_model) -> None:
    assert tiny_model.model is not None
    assert tiny_model.tokenizer is not None
    assert tiny_model.source.is_local


def test_loaded_model_is_frozen(tiny_model) -> None:
    """Requirement: base weights are never trainable."""
    counts = tiny_model.parameter_counts()
    assert counts.total > 0
    assert counts.trainable == 0
    assert counts.frozen == counts.total
    assert all(not p.requires_grad for p in tiny_model.model.parameters())


def test_freezing_can_be_disabled_explicitly(tiny_checkpoint: Path) -> None:
    loaded = load_base_model(
        LoadOptions(
            base_model=str(tiny_checkpoint), device="cpu", dtype="float32", freeze_base=False
        )
    )
    assert loaded.parameter_counts().trainable > 0


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_every_dtype_loads(tiny_checkpoint: Path, dtype: str) -> None:
    loaded = load_base_model(
        LoadOptions(base_model=str(tiny_checkpoint), device="cpu", dtype=dtype)
    )
    assert loaded.actual_dtype() == dtype
    assert loaded.spec.dtype_name == dtype


def test_dtype_changes_the_memory_footprint(tiny_checkpoint: Path) -> None:
    wide = load_base_model(
        LoadOptions(base_model=str(tiny_checkpoint), device="cpu", dtype="float32")
    )
    narrow = load_base_model(
        LoadOptions(base_model=str(tiny_checkpoint), device="cpu", dtype="float16")
    )
    assert narrow.memory_footprint() < wide.memory_footprint()


def test_model_lands_on_the_requested_device(tiny_model) -> None:
    assert tiny_model.actual_device().startswith("cpu")
    assert tiny_model.spec.device == "cpu"


def test_model_can_be_loaded_onto_the_default_device(tiny_checkpoint: Path) -> None:
    loaded = load_base_model(LoadOptions(base_model=str(tiny_checkpoint)))
    assert loaded.actual_device().split(":")[0] == loaded.spec.device_type


def test_keyword_overrides_are_applied(tiny_checkpoint: Path) -> None:
    loaded = load_base_model(
        LoadOptions(base_model=str(tiny_checkpoint)), device="cpu", dtype="float32"
    )
    assert loaded.spec.device == "cpu"


def test_tokenizer_special_tokens_resize_the_embeddings(tiny_model) -> None:
    """<FILL_ME> is added to the vocabulary, so embeddings must grow to match."""
    embeddings = tiny_model.model.get_input_embeddings().weight.shape[0]
    assert embeddings >= len(tiny_model.tokenizer)


def test_model_can_be_loaded_without_a_tokenizer(tiny_checkpoint: Path) -> None:
    loaded = load_base_model(
        LoadOptions(base_model=str(tiny_checkpoint), device="cpu", dtype="float32"),
        load_tokenizer_too=False,
    )
    assert loaded.tokenizer is None


def test_loading_a_missing_checkpoint_explains_how_to_configure_it(tmp_path: Path) -> None:
    with pytest.raises(ModelNotAvailableError, match="downloading is disabled"):
        load_base_model(LoadOptions(base_model=str(tmp_path / "nope"), device="cpu"))


def test_four_bit_needs_cuda(tiny_checkpoint: Path) -> None:
    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA")
    with pytest.raises(ModelError, match="load_in_4bit requires a CUDA device"):
        load_base_model(
            LoadOptions(base_model=str(tiny_checkpoint), device="cpu", load_in_4bit=True)
        )


def test_a_corrupt_checkpoint_reports_the_path(tmp_path: Path) -> None:
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    (broken / "model.safetensors").write_bytes(b"not a real tensor file")
    with pytest.raises(ModelError) as info:
        load_base_model(
            LoadOptions(base_model=str(broken), device="cpu"), load_tokenizer_too=False
        )
    assert str(broken) in str(info.value)


# --------------------------------------------------------------------------- #
# describing what was loaded
# --------------------------------------------------------------------------- #
def test_describe_covers_device_dtype_and_counts(tiny_model) -> None:
    description = tiny_model.describe()
    assert description["actual_device"].startswith("cpu")
    assert description["actual_dtype"] == "float32"
    assert description["parameters"]["total"] > 0
    assert description["parameters"]["trainable"] == 0
    assert description["is_peft"] is False
    assert description["model_class"] == "LlamaForCausalLM"


def test_memory_footprint_is_reported(tiny_model) -> None:
    footprint = tiny_model.memory_footprint()
    counts = tiny_model.parameter_counts()
    assert footprint >= counts.total * 4 * 0.9  # float32, allowing for buffers


def test_log_summary_reports_parameters(tiny_model, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO", logger="repairllama.model.loader"):
        tiny_model.log_summary()
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "parameters" in text
    assert "footprint" in text


# --------------------------------------------------------------------------- #
# counting
# --------------------------------------------------------------------------- #
def test_count_parameters_splits_trainable_and_frozen(tiny_model) -> None:
    model = tiny_model.model
    counts = count_parameters(model)
    assert counts.trainable == 0

    for parameter in model.parameters():
        parameter.requires_grad_(True)
        break
    after = count_parameters(model)
    assert after.trainable > 0
    assert after.trainable + after.frozen == after.total


def test_count_parameters_groups_by_dtype(tiny_model) -> None:
    counts = tiny_model.parameter_counts()
    assert set(counts.by_dtype) == {"float32"}
    assert sum(counts.by_dtype.values()) == counts.total


def test_trainable_fraction_and_render(tiny_model) -> None:
    counts = tiny_model.parameter_counts()
    assert counts.trainable_fraction == 0.0
    assert "trainable" in counts.render()


def test_freeze_parameters_returns_the_count(tiny_model) -> None:
    model = tiny_model.model
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    frozen = freeze_parameters(model)
    assert frozen == count_parameters(model).total
    assert count_parameters(model).trainable == 0
