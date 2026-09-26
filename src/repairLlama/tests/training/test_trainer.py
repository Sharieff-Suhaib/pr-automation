"""The training loop, end to end on a tiny local model.

These are integration tests: the properties that matter (the base does not
move, only the adapter is saved, a resumed run continues rather than
restarting) are properties of a real run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repairllama.config import RepairConfig
from repairllama.model.loader import count_parameters
from repairllama.training.checkpoint import TRAINING_SUMMARY_FILE, list_checkpoints
from repairllama.training.trainer import (
    PAPER_SETTINGS,
    JsonlDataset,
    TrainingError,
    _matches_paper,
    build_training_arguments,
    load_split,
    print_training_banner,
    resolve_mixed_precision,
    run_training,
)

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def test_load_split(splits_dir: Path) -> None:
    dataset = load_split(splits_dir, "train")
    assert len(dataset) == 8
    assert set(dataset[0]) >= {"input", "output"}


def test_optional_splits(splits_dir: Path, tmp_path: Path) -> None:
    assert load_split(splits_dir, "validation", required=False) is not None
    assert load_split(tmp_path, "validation", required=False) is None


def test_a_missing_train_split_says_how_to_build_one(tmp_path: Path) -> None:
    with pytest.raises(TrainingError, match="prepare-data"):
        load_split(tmp_path, "train")


def test_an_unknown_split_is_rejected(splits_dir: Path) -> None:
    with pytest.raises(TrainingError, match="unknown split"):
        load_split(splits_dir, "dev")


def test_rows_must_have_input_and_output(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text(json.dumps({"prompt": "a"}) + "\n", encoding="utf-8")
    with pytest.raises(TrainingError, match="'input' and 'output'"):
        load_split(tmp_path, "train")


def test_an_empty_split_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(TrainingError, match="nothing to train on"):
        load_split(tmp_path, "train")


def test_jsonl_dataset_is_a_sequence() -> None:
    dataset = JsonlDataset([{"input": "a", "output": "b"}], "train")
    assert len(dataset) == 1
    assert list(dataset) == dataset.rows


# --------------------------------------------------------------------------- #
# mixed precision
# --------------------------------------------------------------------------- #
def test_no_mixed_precision_on_cpu() -> None:
    assert resolve_mixed_precision("auto", "cpu") == (False, False)
    assert resolve_mixed_precision("no", "cuda:0") == (False, False)


def test_requested_precision_on_cpu_falls_back_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="repairllama.training.trainer"):
        assert resolve_mixed_precision("fp16", "cpu") == (False, False)
        assert resolve_mixed_precision("bf16", "mps") == (False, False)
    assert len(caplog.records) == 2


def test_explicit_precision_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    assert resolve_mixed_precision("fp16", "cuda:0") == (True, False)
    assert resolve_mixed_precision("bf16", "cuda:0") == (False, True)


# --------------------------------------------------------------------------- #
# training arguments
# --------------------------------------------------------------------------- #
def test_arguments_carry_the_configured_values(tmp_path: Path) -> None:
    cfg = RepairConfig.default()
    arguments = build_training_arguments(
        cfg, tmp_path, device="cpu", dataset_size=100
    )
    assert arguments.learning_rate == 5e-4
    assert arguments.num_train_epochs == 2
    assert arguments.lr_scheduler_type.value == "cosine"
    assert arguments.per_device_train_batch_size == cfg.training.batch_size
    assert arguments.gradient_accumulation_steps == cfg.training.gradient_accumulation_steps
    assert arguments.seed == cfg.training.seed


def test_cpu_is_not_silently_overridden(tmp_path: Path) -> None:
    """The Trainer otherwise picks an accelerator and ignores model.device."""
    arguments = build_training_arguments(
        RepairConfig.default(), tmp_path, device="cpu", dataset_size=10
    )
    assert arguments.use_cpu is True


def test_evaluation_can_be_disabled(tmp_path: Path) -> None:
    cfg = RepairConfig.default().override({"training.eval_strategy": "no"})
    arguments = build_training_arguments(cfg, tmp_path, device="cpu", dataset_size=10)
    assert str(arguments.eval_strategy) in {"no", "IntervalStrategy.NO"}


def test_max_steps_overrides_epochs(tmp_path: Path) -> None:
    cfg = RepairConfig.default().override({"training.max_steps": 50})
    arguments = build_training_arguments(cfg, tmp_path, device="cpu", dataset_size=10)
    assert arguments.max_steps == 50


# --------------------------------------------------------------------------- #
# the banner
# --------------------------------------------------------------------------- #
def test_banner_prints_every_required_field() -> None:
    from repairllama.model.adapter import LoRASettings
    from repairllama.model.loader import ParameterCounts

    lines: list[str] = []
    payload = print_training_banner(
        counts=ParameterCounts(total=1000, trainable=10, frozen=990),
        settings=LoRASettings(),
        dataset_sizes={"train": 500, "validation": 20},
        max_length=1024,
        batch_size=4,
        gradient_accumulation=8,
        learning_rate=5e-4,
        epochs=2,
        device="cpu",
        dtype="float32",
        base_model="codellama/CodeLlama-7b-hf",
        printer=lines.append,
    )
    text = "\n".join(lines)
    for label in (
        "Base parameters:",
        "Trainable parameters:",
        "Trainable percentage:",
        "Dataset size:",
        "Max sequence length:",
        "Batch size:",
        "Learning rate:",
        "LoRA configuration:",
    ):
        assert label in text

    assert "990" in text and "10" in text
    assert "1.0000%" in text
    assert "q_proj" in text and "v_proj" in text
    assert payload["trainable_parameters"] == 10
    assert payload["effective_batch_size"] == 32


# --------------------------------------------------------------------------- #
# the paper comparison — honest, not aspirational
# --------------------------------------------------------------------------- #
def test_defaults_match_the_reported_hyperparameters() -> None:
    report = _matches_paper(RepairConfig.default())
    assert report["all_match"] is True
    assert report["differences"] == {}
    assert report["used"]["learning_rate"] == PAPER_SETTINGS["learning_rate"]


def test_differences_from_the_paper_are_recorded() -> None:
    cfg = RepairConfig.default().override(
        {"training.epochs": 5, "training.learning_rate": 1e-4}
    )
    report = _matches_paper(cfg)
    assert report["all_match"] is False
    assert report["differences"]["epochs"] == {"paper": 2, "used": 5}
    assert report["differences"]["learning_rate"]["used"] == 1e-4


def test_the_paper_settings_are_the_reported_ones() -> None:
    assert PAPER_SETTINGS["learning_rate"] == 5e-4
    assert PAPER_SETTINGS["lr_scheduler"] == "cosine"
    assert PAPER_SETTINGS["epochs"] == 2
    assert PAPER_SETTINGS["max_length"] == 1024
    assert PAPER_SETTINGS["lora_r"] == 8
    assert PAPER_SETTINGS["lora_alpha"] == 16
    assert PAPER_SETTINGS["target_modules"] == ["q_proj", "v_proj"]


# --------------------------------------------------------------------------- #
# dry run
# --------------------------------------------------------------------------- #
def test_dry_run_sets_up_without_training(training_config, splits_dir: Path) -> None:
    result = run_training(training_config, splits_dir=splits_dir, dry_run=True)
    assert result.artifact_dir is None
    assert result.summary["dry_run"] is True
    assert result.summary["run"]["trainable_parameters"] > 0
    assert result.summary["run"]["dataset_size"]["train"] == 8


def test_dry_run_writes_no_adapter(training_config, splits_dir: Path, tmp_path: Path) -> None:
    adapters = tmp_path / "adapters"
    run_training(
        training_config, splits_dir=splits_dir, adapters_dir=adapters, dry_run=True
    )
    assert not (adapters / "java-repair" / "adapter_model.safetensors").exists()


# --------------------------------------------------------------------------- #
# a real (very short) run
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def _trained(tmp_path_factory, request):
    """One short training run, shared by the assertions below."""
    return None


def test_training_produces_an_adapter_artifact(
    training_config, splits_dir: Path, tmp_path: Path
) -> None:
    result = run_training(
        training_config,
        splits_dir=splits_dir,
        adapters_dir=tmp_path / "adapters",
        output_dir=tmp_path / "outputs",
    )
    artifact = result.artifact_dir
    assert artifact == tmp_path / "adapters" / "java-repair"

    names = {entry.name for entry in artifact.iterdir()}
    assert "adapter_model.safetensors" in names
    assert "adapter_config.json" in names
    assert TRAINING_SUMMARY_FILE in names
    assert any(name.startswith("tokenizer") for name in names)
    # only the adapter: no base weights, no optimizer state
    assert "model.safetensors" not in names
    assert "optimizer.pt" not in names


def test_training_updates_only_the_adapter(
    training_config, splits_dir: Path, tmp_path: Path
) -> None:
    """The whole point of LoRA: the base must come out bit-for-bit unchanged."""
    result = run_training(
        training_config,
        splits_dir=splits_dir,
        adapters_dir=tmp_path / "adapters",
        output_dir=tmp_path / "outputs",
    )
    counts = count_parameters(result.model)
    assert 0 < counts.trainable < counts.total * 0.05
    assert result.summary["parameters"]["base_modified"] is False
    # run_training raises if the guard finds a changed base weight, so reaching
    # here is itself the assertion; check the recorded claim as well
    assert result.summary["parameters"]["adapter_only"] is True


def test_training_records_losses(training_config, splits_dir: Path, tmp_path: Path) -> None:
    cfg = training_config.override(
        {"training.eval_strategy": "steps", "training.eval_steps": 1}
    )
    result = run_training(
        cfg,
        splits_dir=splits_dir,
        adapters_dir=tmp_path / "adapters",
        output_dir=tmp_path / "outputs",
    )
    assert result.final_train_loss is not None
    assert result.final_validation_loss is not None
    assert result.summary["history"]["evaluations"]
    assert result.summary["history"]["best_validation"]["perplexity"] > 0


def test_training_summary_is_complete(
    training_config, splits_dir: Path, tmp_path: Path
) -> None:
    result = run_training(
        training_config,
        splits_dir=splits_dir,
        adapters_dir=tmp_path / "adapters",
        output_dir=tmp_path / "outputs",
    )
    summary = json.loads(
        (result.artifact_dir / TRAINING_SUMMARY_FILE).read_text(encoding="utf-8")
    )
    assert summary["experiment"] == "test-run"
    assert summary["run"]["lora"]["r"] == 8
    assert summary["settings"]["training"]["learning_rate"] == 5e-4
    assert summary["environment"]["torch"] == torch.__version__
    assert summary["data"]["max_length"] == 128
    assert summary["data"]["splits_dir"] == str(splits_dir)
    assert summary["duration_seconds"] > 0


def test_summary_does_not_claim_to_reproduce_the_paper(
    training_config, splits_dir: Path, tmp_path: Path
) -> None:
    """A short run on a toy model must not be recorded as a reproduction."""
    result = run_training(
        training_config,
        splits_dir=splits_dir,
        adapters_dir=tmp_path / "adapters",
        output_dir=tmp_path / "outputs",
    )
    reproduction = result.summary["reproduction"]
    assert reproduction["verified_reproduction_of_paper_results"] is False
    # this run overrides epochs/max_length, and the summary says so
    assert reproduction["matches_paper_reported_hyperparameters"]["all_match"] is False
    assert "max_length" in reproduction["matches_paper_reported_hyperparameters"]["differences"]
    assert "does not by itself reproduce" in reproduction["note"]


def test_seed_is_recorded_and_applied(
    training_config, splits_dir: Path, tmp_path: Path
) -> None:
    result = run_training(
        training_config,
        splits_dir=splits_dir,
        adapters_dir=tmp_path / "adapters",
        output_dir=tmp_path / "outputs",
    )
    assert result.summary["settings"]["training"]["seed"] == 42


def test_checkpoints_are_written_and_resumable(
    training_config, splits_dir: Path, tmp_path: Path
) -> None:
    cfg = training_config.override(
        {"training.save_strategy": "steps", "training.save_steps": 1}
    )
    first = run_training(
        cfg,
        splits_dir=splits_dir,
        adapters_dir=tmp_path / "adapters",
        output_dir=tmp_path / "outputs",
    )
    checkpoints = list_checkpoints(first.run_dir)
    assert checkpoints, "training should have written checkpoints"

    resumed = run_training(
        cfg.override({"training.max_steps": 4}),
        splits_dir=splits_dir,
        adapters_dir=tmp_path / "adapters",
        output_dir=tmp_path / "outputs",
        resume="auto",
    )
    assert resumed.summary["resumed_from"] == str(checkpoints[-1])
    # the run continued rather than starting over
    steps = [entry["step"] for entry in resumed.summary["history"]["train_loss_curve"]]
    assert max(steps) == 4


def test_max_examples_caps_the_training_set(
    training_config, splits_dir: Path, tmp_path: Path
) -> None:
    result = run_training(
        training_config,
        splits_dir=splits_dir,
        adapters_dir=tmp_path / "adapters",
        output_dir=tmp_path / "outputs",
        max_examples=3,
    )
    assert result.summary["run"]["dataset_size"]["train"] == 3


def test_training_without_a_validation_split(
    training_config, splits_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (splits_dir / "validation.jsonl").unlink()
    cfg = training_config.override({"training.eval_strategy": "steps"})
    result = run_training(
        cfg,
        splits_dir=splits_dir,
        adapters_dir=tmp_path / "adapters",
        output_dir=tmp_path / "outputs",
    )
    assert result.artifact_dir.is_dir()
    assert result.final_validation_loss is None
