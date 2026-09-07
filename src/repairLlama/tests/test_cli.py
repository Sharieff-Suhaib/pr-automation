"""CLI wiring: parsing, exit codes and the commands that already work."""

from __future__ import annotations

from pathlib import Path

import pytest

from repairllama import cli


def _run(argv: list[str]) -> int:
    return cli.main(argv)


def test_help_lists_every_stage(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _run(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for command in ["config", "init", "prepare-data", "localize", "train", "infer",
                    "patch", "evaluate"]:
        assert command in out


def test_no_command_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run([]) == cli.EXIT_USAGE
    assert "usage:" in capsys.readouterr().out


def test_config_show(config_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["config", "show", "-c", str(config_path), "--no-log-file"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "experiment_name:" in out
    assert "lora:" in out


def test_config_validate(config_path: Path) -> None:
    assert _run(["config", "validate", "-c", str(config_path), "--no-log-file"]) == cli.EXIT_OK


def test_config_rejects_bad_override(config_path: Path) -> None:
    code = _run(
        ["config", "validate", "-c", str(config_path), "--no-log-file",
         "--set", "training.epochs=-1"]
    )
    assert code == cli.EXIT_USAGE


def test_malformed_override_rejected(config_path: Path) -> None:
    code = _run(["config", "show", "-c", str(config_path), "--no-log-file", "--set", "epochs"])
    assert code == cli.EXIT_USAGE


def test_override_is_yaml_typed() -> None:
    parsed = cli._parse_overrides(["training.epochs=5", "model.lora.enabled=false"])
    assert parsed == {"training.epochs": 5, "model.lora.enabled": False}


def test_init_creates_directories(tmp_path: Path, config_path: Path) -> None:
    code = _run(
        ["init", "-c", str(config_path), "--no-log-file",
         "--set", f"paths.project_root={tmp_path}"]
    )
    assert code == cli.EXIT_OK
    for relative in ["data/raw", "data/processed", "data/splits", "models",
                     "adapters/java-repair", "outputs"]:
        assert (tmp_path / relative).is_dir()


def test_init_dry_run_creates_nothing(tmp_path: Path, config_path: Path) -> None:
    code = _run(
        ["init", "-c", str(config_path), "--no-log-file", "--dry-run",
         "--set", f"paths.project_root={tmp_path}"]
    )
    assert code == cli.EXIT_OK
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["localize"],
        ["train"],
        ["infer"],
        ["patch"],
        ["evaluate"],
    ],
)
def test_unimplemented_stages_exit_cleanly(argv: list[str], config_path: Path) -> None:
    """Stages must validate config and report cleanly, not raise a traceback."""
    code = _run([*argv, "-c", str(config_path), "--no-log-file"])
    assert code == cli.EXIT_NOT_IMPLEMENTED


# --------------------------------------------------------------------------- #
# prepare-data
# --------------------------------------------------------------------------- #
def _corpus(path: Path, count: int = 12) -> Path:
    """A small JSONL corpus of one-line Java bugs spread over projects."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            buggy = (
                f"public class Variant{index} {{\n"
                f"    public int value(int a) {{\n"
                f"        return a - {index};\n"
                "    }\n}\n"
            )
            handle.write(
                json.dumps(
                    {
                        "bug_id": f"bug-{index:03d}",
                        "buggy_code": buggy,
                        "fixed_code": buggy.replace(
                            f"return a - {index};", f"return a + {index};"
                        ),
                        "project": f"project-{index}",
                    }
                )
                + "\n"
            )
    return path


def test_prepare_data_writes_splits_and_a_report(
    tmp_path: Path, config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    source = _corpus(tmp_path / "raw" / "corpus.jsonl")
    out = tmp_path / "splits"
    code = _run(
        ["prepare-data", "-c", str(config_path), "--no-log-file",
         "--input", str(source), "--output", str(out)]
    )
    assert code == cli.EXIT_OK

    assert "dataset report" in capsys.readouterr().out
    for name in ("train", "validation", "test"):
        assert (out / f"{name}.jsonl").is_file()

    rows = [
        json.loads(line)
        for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows
    assert "<FILL_ME>" in rows[0]["input"]
    assert rows[0]["output"]

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["raw_samples"] == 12
    assert report["remaining"] == sum(report["splits"].values())


def test_prepare_data_minimal_rows(tmp_path: Path, config_path: Path) -> None:
    import json

    source = _corpus(tmp_path / "raw" / "corpus.jsonl", count=6)
    out = tmp_path / "splits"
    code = _run(
        ["prepare-data", "-c", str(config_path), "--no-log-file", "--minimal",
         "--input", str(source), "--output", str(out)]
    )
    assert code == cli.EXIT_OK
    row = json.loads((out / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert set(row) == {"input", "output"}


def test_prepare_data_dry_run_writes_nothing(tmp_path: Path, config_path: Path) -> None:
    source = _corpus(tmp_path / "raw" / "corpus.jsonl", count=4)
    out = tmp_path / "splits"
    code = _run(
        ["prepare-data", "-c", str(config_path), "--no-log-file", "--dry-run",
         "--input", str(source), "--output", str(out)]
    )
    assert code == cli.EXIT_OK
    assert not out.exists()


def test_prepare_data_limit_is_applied(
    tmp_path: Path, config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _corpus(tmp_path / "raw" / "corpus.jsonl", count=12)
    code = _run(
        ["prepare-data", "-c", str(config_path), "--no-log-file", "--dry-run",
         "--limit", "3", "--input", str(source)]
    )
    assert code == cli.EXIT_OK
    assert "raw samples             3" in capsys.readouterr().out


def test_prepare_data_reports_a_missing_corpus(tmp_path: Path, config_path: Path) -> None:
    code = _run(
        ["prepare-data", "-c", str(config_path), "--no-log-file",
         "--input", str(tmp_path / "absent.jsonl")]
    )
    assert code == cli.EXIT_USAGE


# --------------------------------------------------------------------------- #
# model-check (dry run)
# --------------------------------------------------------------------------- #
def test_model_check_is_listed(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        _run(["--help"])
    assert "model-check" in capsys.readouterr().out


def test_model_check_without_a_checkpoint_explains_how_to_configure_it(
    config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No downloads: the failure must tell the user how to point at a model."""
    code = _run(
        ["model-check", "-c", str(config_path), "--no-log-file",
         "--model", str(tmp_path / "not-a-model"), "--device", "cpu"]
    )
    assert code == cli.EXIT_MODEL_UNAVAILABLE
    message = capsys.readouterr().err
    assert "downloading is disabled" in message
    assert "huggingface-cli download" in message
    assert "model.base_model" in message


def test_model_check_rejects_an_unavailable_device(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA")
    code = _run(["model-check", "-c", str(config_path), "--no-log-file", "--device", "cuda"])
    assert code == cli.EXIT_USAGE
    assert "CUDA" in capsys.readouterr().err


def _tiny_checkpoint(tmp_path: Path) -> Path:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "model"))
    import tiny_checkpoint as tc

    return tc.build_checkpoint(tmp_path / "tiny")


def test_model_check_reports_parameters_and_verifies(
    config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = _tiny_checkpoint(tmp_path)
    code = _run(
        ["model-check", "-c", str(config_path), "--no-log-file",
         "--model", str(checkpoint), "--device", "cpu", "--dtype", "float32"]
    )
    assert code == cli.EXIT_OK
    text = capsys.readouterr().err
    assert "total parameters" in text
    assert "trainable parameters" in text
    assert "memory footprint" in text
    assert "[PASS] model is on the resolved device" in text
    assert "[PASS] model is in the resolved dtype" in text
    assert "[PASS] base weights are frozen" in text
    assert "check(s) passed" in text


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_model_check_honours_each_dtype(
    config_path: Path, tmp_path: Path, dtype: str, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = _tiny_checkpoint(tmp_path)
    code = _run(
        ["model-check", "-c", str(config_path), "--no-log-file",
         "--model", str(checkpoint), "--device", "cpu", "--dtype", dtype]
    )
    assert code == cli.EXIT_OK
    assert f"dtype: {dtype}" in capsys.readouterr().err


def test_model_check_tokenizer_only_skips_the_weights(
    config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = _tiny_checkpoint(tmp_path)
    code = _run(
        ["model-check", "-c", str(config_path), "--no-log-file", "--tokenizer-only",
         "--model", str(checkpoint), "--device", "cpu"]
    )
    assert code == cli.EXIT_OK
    text = capsys.readouterr().err
    assert "total parameters" not in text
    assert "[PASS] tokenizer has a pad token" in text


def test_model_check_no_load_resolves_without_reading_weights(
    config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = _tiny_checkpoint(tmp_path)
    code = _run(
        ["model-check", "-c", str(config_path), "--no-log-file", "--no-load",
         "--model", str(checkpoint), "--device", "cpu"]
    )
    assert code == cli.EXIT_OK
    text = capsys.readouterr().err
    assert "checkpoint:" in text
    assert "total parameters" not in text


def test_model_check_attach_lora_reports_trainable_parameters(
    config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("peft")
    checkpoint = _tiny_checkpoint(tmp_path)
    code = _run(
        ["model-check", "-c", str(config_path), "--no-log-file", "--attach-lora",
         "--model", str(checkpoint), "--device", "cpu", "--dtype", "float32"]
    )
    assert code == cli.EXIT_OK
    text = capsys.readouterr().err
    assert "trainable after adapter" in text
    assert "[PASS] base weights unchanged by the adapter" in text


def test_model_check_reports_a_bad_adapter_path(
    config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = _tiny_checkpoint(tmp_path)
    code = _run(
        ["model-check", "-c", str(config_path), "--no-log-file",
         "--model", str(checkpoint), "--device", "cpu",
         "--adapter", str(tmp_path / "no-adapter")]
    )
    assert code == cli.EXIT_USAGE
    assert "adapter directory not found" in capsys.readouterr().err
