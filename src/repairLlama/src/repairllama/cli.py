"""Command line entry points for the RepairLLaMA-Java pipeline.

    repairllama --help
    repairllama config show --config configs/java_repair.yaml
    repairllama init
    repairllama prepare-data --set data.max_examples=500

``config``, ``init``, ``prepare-data``, ``model-check`` and ``train`` do real
work today.  The remaining
pipeline stages resolve their configuration, log the plan they would execute
and then exit with :data:`EXIT_NOT_IMPLEMENTED` until that phase lands, which
keeps the argument surface stable while the stages are filled in one at a
time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from repairllama import __version__
from repairllama.config import (
    ConfigError,
    RepairConfig,
    load_config,
    parse_override_value,
)
from repairllama.utils.logging import configure_from_config, get_logger, log_section
from repairllama.utils.paths import default_config_path
from repairllama.utils.seed import set_seed

__all__ = ["main", "build_parser", "StageNotImplemented"]

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_NOT_IMPLEMENTED = 2
EXIT_MODEL_UNAVAILABLE = 3

log = get_logger("cli")


class StageNotImplemented(NotImplementedError):
    """Raised by a pipeline stage that is scheduled for a later phase."""


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def _parse_overrides(pairs: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Turn ``["training.epochs=5", "model.device=cpu"]`` into a dict.

    Values are parsed as YAML scalars, so ``5`` becomes an int and ``true`` a
    bool, matching what the same key would mean in the config file — except
    that ``no``/``yes``/``on``/``off`` stay strings, since several settings
    take them as values.
    """
    overrides: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ConfigError(f"--set expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        key = key.strip()
        if not key:
            raise ConfigError(f"--set expects key=value, got {pair!r}")
        overrides[key] = parse_override_value(raw)
    return overrides


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        metavar="PATH",
        default=None,
        help="YAML config file (default: configs/java_repair.yaml if present)",
    )
    parser.add_argument(
        "-s",
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set training.epochs=5 (repeatable)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="override logging.level for this run",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="log to the console only",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repairllama",
        description="Fine-tuned LLM program repair for Java.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-V", "--version", action="version", version=f"repairllama {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # -- config ------------------------------------------------------------- #
    p_config = subparsers.add_parser(
        "config", help="show or validate the resolved configuration"
    )
    _add_common_args(p_config)
    p_config.add_argument(
        "action",
        choices=["show", "validate", "path"],
        nargs="?",
        default="show",
        help="show the merged config, validate it, or print the file used",
    )
    p_config.set_defaults(func=cmd_config)

    # -- init --------------------------------------------------------------- #
    p_init = subparsers.add_parser(
        "init", help="create the data/model/output directories from the config"
    )
    _add_common_args(p_init)
    p_init.add_argument(
        "--dry-run", action="store_true", help="print directories without creating them"
    )
    p_init.set_defaults(func=cmd_init)

    # -- pipeline stages ---------------------------------------------------- #
    p_data = subparsers.add_parser(
        "prepare-data", help="build bug/fix pairs and write train/val/test splits"
    )
    _add_common_args(p_data)
    p_data.add_argument(
        "--input",
        default=None,
        help="corpus path (default: <raw_data_dir>/<dataset_name>.jsonl)",
    )
    p_data.add_argument(
        "--format",
        choices=["jsonl", "json", "directory"],
        default=None,
        help="override data.loader_format",
    )
    p_data.add_argument("--output", default=None, help="override the splits directory")
    p_data.add_argument("--limit", type=int, default=None, help="cap the example count")
    p_data.add_argument(
        "--minimal",
        action="store_true",
        help='emit only {"input", "output"} in the JSONL rows',
    )
    p_data.add_argument(
        "--dry-run", action="store_true", help="build and report without writing files"
    )
    p_data.set_defaults(func=cmd_prepare_data)

    p_model = subparsers.add_parser(
        "model-check",
        help="dry run: load the tokenizer/model, print parameter counts, verify "
        "device and dtype, then exit",
    )
    _add_common_args(p_model)
    p_model.add_argument(
        "--model", default=None, help="checkpoint path or id (overrides model.base_model)"
    )
    p_model.add_argument(
        "--device", default=None, help="auto | cpu | cuda | cuda:<n> | mps"
    )
    p_model.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default=None
    )
    p_model.add_argument(
        "--adapter", default=None, help="LoRA adapter directory to load after the base"
    )
    p_model.add_argument(
        "--attach-lora",
        action="store_true",
        help="attach a fresh LoRA adapter and report trainable parameters",
    )
    p_model.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="check the tokenizer without loading model weights",
    )
    p_model.add_argument(
        "--no-load",
        action="store_true",
        help="resolve the device, dtype and checkpoint path without loading anything",
    )
    p_model.set_defaults(func=cmd_model_check)

    p_localize = subparsers.add_parser(
        "localize", help="run fault localization over a project or dataset"
    )
    _add_common_args(p_localize)
    p_localize.add_argument("--project", default=None, help="path to a Java project")
    p_localize.set_defaults(func=cmd_localize)

    p_train = subparsers.add_parser(
        "train", help="fine-tune the Java repair LoRA adapter"
    )
    _add_common_args(p_train)
    p_train.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="resume: 'auto' for the newest checkpoint, or a checkpoint path",
    )
    p_train.add_argument("--splits-dir", default=None, help="where the JSONL splits live")
    p_train.add_argument("--adapters-dir", default=None, help="parent of java-repair/")
    p_train.add_argument(
        "--max-examples", type=int, default=0, help="cap training rows (0 = all)"
    )
    p_train.add_argument(
        "--dry-run",
        action="store_true",
        help="set everything up and print the banner without training",
    )
    p_train.set_defaults(func=cmd_train)

    p_infer = subparsers.add_parser("infer", help="generate candidate patches for a bug")
    _add_common_args(p_infer)
    p_infer.add_argument("--input", default=None, help="bug JSONL or single buggy file")
    p_infer.add_argument("--output", default=None, help="where to write candidates")
    p_infer.add_argument("-n", "--num-candidates", type=int, default=None)
    p_infer.set_defaults(func=cmd_infer)

    p_patch = subparsers.add_parser("patch", help="apply candidate patches to source files")
    _add_common_args(p_patch)
    p_patch.add_argument("--candidates", default=None, help="candidate JSONL from `infer`")
    p_patch.add_argument("--project", default=None, help="path to the Java project")
    p_patch.set_defaults(func=cmd_patch)

    p_eval = subparsers.add_parser("evaluate", help="compile and test candidate patches")
    _add_common_args(p_eval)
    p_eval.add_argument("--candidates", default=None, help="candidate JSONL to score")
    p_eval.add_argument("--report", default=None, help="where to write the JSON report")
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


# --------------------------------------------------------------------------- #
# config bootstrap
# --------------------------------------------------------------------------- #
def _resolve_config(args: argparse.Namespace) -> RepairConfig:
    path = args.config
    if path is None:
        shipped = default_config_path()
        path = str(shipped) if shipped.is_file() else None

    overrides = _parse_overrides(getattr(args, "overrides", None))
    if getattr(args, "log_level", None):
        overrides["logging.level"] = args.log_level
    if getattr(args, "no_log_file", False):
        overrides["logging.log_to_file"] = False

    cfg = load_config(path, overrides)
    configure_from_config(cfg)
    if path:
        log.debug("loaded config from %s", path)
    else:
        log.debug("no config file found; using built-in defaults")
    return cfg


def _stage_banner(cfg: RepairConfig, title: str, details: Dict[str, Any]) -> None:
    log_section(log, f"{title} — experiment '{cfg.experiment_name}'")
    for key, value in details.items():
        log.info("%-22s %s", key + ":", value)


def _not_implemented(stage: str, phase: str) -> int:
    log.warning("`%s` is not implemented yet (%s).", stage, phase)
    log.warning("Configuration and paths above are validated and ready for it.")
    return EXIT_NOT_IMPLEMENTED


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_config(args: argparse.Namespace) -> int:
    cfg = _resolve_config(args)
    if args.action == "path":
        shipped = args.config or default_config_path()
        print(shipped)
        return EXIT_OK
    if args.action == "validate":
        log.info("configuration is valid")
        return EXIT_OK
    print(cfg.to_yaml(), end="")
    return EXIT_OK


def cmd_init(args: argparse.Namespace) -> int:
    cfg = _resolve_config(args)
    keys = [
        "raw_data_dir",
        "processed_data_dir",
        "splits_dir",
        "models_dir",
        "adapters_dir",
        "output_dir",
        "log_dir",
    ]
    # One adapter directory per implemented language, so the layout the
    # adapters registry documents exists from the start.
    from repairllama.model.lora import ADAPTER_PROFILES

    log_section(log, "initialising project directories")
    directories = [cfg.paths.resolve(key) for key in keys]
    adapters_root = cfg.paths.resolve("adapters_dir")
    directories.extend(
        adapters_root / profile.directory for profile in ADAPTER_PROFILES.values()
    )

    for directory in directories:
        if args.dry_run:
            log.info("would create %s", directory)
        else:
            directory.mkdir(parents=True, exist_ok=True)
            log.info("ready %s", directory)
    return EXIT_OK


def cmd_prepare_data(args: argparse.Namespace) -> int:
    from repairllama.data import DatasetBuilder, DataError, LoaderOptions

    cfg = _resolve_config(args)
    if args.limit is not None:
        cfg = cfg.override({"data.max_examples": args.limit})
    if args.format:
        cfg = cfg.override({"data.loader_format": args.format})
    set_seed(cfg.data.shuffle_seed)

    source = (
        Path(args.input)
        if args.input
        else cfg.paths.resolve("raw_data_dir") / f"{cfg.data.dataset_name}.jsonl"
    )
    splits_dir = Path(args.output) if args.output else cfg.paths.resolve("splits_dir")

    _stage_banner(
        cfg,
        "prepare-data",
        {
            "dataset": cfg.data.dataset_name,
            "source": source,
            "format": cfg.data.loader_format,
            "max examples": cfg.data.max_examples or "all",
            "splits": f"{cfg.data.train_split}/{cfg.data.val_split}/{cfg.data.test_split}",
            "splits dir": splits_dir,
            "seed": cfg.data.shuffle_seed,
        },
    )

    builder = DatasetBuilder.from_config(cfg)
    try:
        result = builder.build_from_source(
            source,
            LoaderOptions(format=cfg.data.loader_format, skip_invalid=True),
        )
    except DataError as exc:
        log.error("could not build the dataset: %s", exc)
        return EXIT_USAGE

    print(result.report.render())
    if args.dry_run:
        log.info("dry run: no files written")
        return EXIT_OK
    if not result.total:
        log.warning("no records survived the pipeline; nothing was written")
        return EXIT_USAGE

    written = result.write(
        splits_dir, minimal=args.minimal or cfg.data.minimal_records
    )
    log.info("dataset written to %s", written["train"].parent)
    return EXIT_OK


def cmd_model_check(args: argparse.Namespace) -> int:
    """Load the model layer and report what it got — a dry run, no training."""
    from repairllama.model import (
        AdapterError,
        BaseWeightGuard,
        DeviceError,
        LoadOptions,
        LoRASettings,
        ModelError,
        ModelNotAvailableError,
        adapter_state,
        attach_lora,
        count_parameters,
        describe_device,
        format_bytes,
        format_count,
        load_adapter,
        load_base_model,
        load_tokenizer,
        resolve_device_spec,
        resolve_model_source,
        torch_available,
    )

    cfg = _resolve_config(args)
    overrides: Dict[str, Any] = {}
    if args.model:
        overrides["model.base_model"] = args.model
    if args.device:
        overrides["model.device"] = args.device
    if args.dtype:
        overrides["model.dtype"] = args.dtype
    if args.adapter:
        overrides["model.adapter_path"] = args.adapter
    if overrides:
        cfg = cfg.override(overrides)

    log_section(log, f"model check — experiment '{cfg.experiment_name}'")
    checks: List[Tuple[str, bool, str]] = []

    if not torch_available():
        log.error(
            "torch is not installed, so no model can be loaded. Install the model "
            'stack with `pip install -e ".[train]"`.'
        )
        return EXIT_MODEL_UNAVAILABLE

    # -- device and dtype --------------------------------------------------- #
    try:
        spec = resolve_device_spec(cfg.model.device, cfg.model.dtype)
    except DeviceError as exc:
        log.error("%s", exc)
        return EXIT_USAGE
    info = describe_device(spec.device)
    log.info("torch %s", info.get("torch"))
    log.info(
        "device: %s (%s) | dtype: %s (requested %s / %s)",
        spec.device,
        info.get("name", "unknown"),
        spec.dtype_name,
        cfg.model.device,
        cfg.model.dtype,
    )
    if info.get("total_memory"):
        log.info("device memory: %s", format_bytes(info["total_memory"]))
    for note in spec.notes:
        log.info("note: %s", note)

    # -- checkpoint resolution ---------------------------------------------- #
    options = LoadOptions.from_config(cfg)
    try:
        source = resolve_model_source(
            options.base_model, options.models_dir, options.allow_download
        )
    except ModelNotAvailableError as exc:
        log.error("%s", exc)
        return EXIT_MODEL_UNAVAILABLE
    except ModelError as exc:
        log.error("%s", exc)
        return EXIT_USAGE
    log.info("checkpoint: %s (%s)", source.reference, source.origin)

    if args.no_load:
        log.info("--no-load: stopping before any weights are read")
        return EXIT_OK

    # -- tokenizer and model ------------------------------------------------- #
    # In tokenizer-only mode the tokenizer is loaded here; otherwise
    # load_base_model loads it alongside the weights, so it is only read once.
    from repairllama.model import log_tokenizer_summary

    loaded = None
    try:
        if args.tokenizer_only:
            tokenizer, changes = load_tokenizer(
                options.tokenizer or source.reference,
                revision=options.revision,
                local_files_only=options.local_files_only,
                trust_remote_code=options.trust_remote_code,
                padding_side=options.padding_side,
                extra_tokens=options.extra_tokens,
            )
            log_tokenizer_summary(tokenizer, changes)
        else:
            loaded = load_base_model(options)
            tokenizer = loaded.tokenizer
            log_tokenizer_summary(tokenizer, loaded.tokenizer_changes)
    except ModelNotAvailableError as exc:
        log.error("%s", exc)
        return EXIT_MODEL_UNAVAILABLE
    except ModelError as exc:
        log.error("%s", exc)
        return EXIT_USAGE

    checks.append(
        ("tokenizer has a pad token", tokenizer.pad_token is not None, str(tokenizer.pad_token))
    )
    fill_token = cfg.representation.fill_token
    encoded = tokenizer.encode(fill_token, add_special_tokens=False)
    checks.append(
        (
            f"{fill_token} is a single token",
            len(encoded) == 1,
            f"{len(encoded)} token(s)",
        )
    )

    if loaded is None:  # --tokenizer-only
        return _report_checks(checks)

    counts = loaded.parameter_counts()
    log_section(log, "parameters")
    log.info("total parameters:     %14s (%s)", f"{counts.total:,}", format_count(counts.total))
    log.info("trainable parameters: %14s (%.4f%%)", f"{counts.trainable:,}",
             counts.trainable_fraction * 100)
    log.info("frozen parameters:    %14s", f"{counts.frozen:,}")
    for dtype_name, count in sorted(counts.by_dtype.items()):
        log.info("  %-10s %14s", dtype_name, f"{count:,}")
    log.info("memory footprint: %s", format_bytes(loaded.memory_footprint()))

    checks.append(
        (
            "model is on the resolved device",
            loaded.actual_device().split(":")[0] == spec.device_type,
            f"{loaded.actual_device()} vs {spec.device}",
        )
    )
    checks.append(
        (
            "model is in the resolved dtype",
            loaded.actual_dtype() == spec.dtype_name,
            f"{loaded.actual_dtype()} vs {spec.dtype_name}",
        )
    )
    checks.append(
        (
            "base weights are frozen",
            counts.trainable == 0,
            f"{counts.trainable:,} trainable",
        )
    )
    checks.append(
        (
            "embeddings match the tokenizer",
            loaded.model.get_input_embeddings().weight.shape[0] >= len(tokenizer),
            f"{loaded.model.get_input_embeddings().weight.shape[0]} vs {len(tokenizer)}",
        )
    )

    # -- adapters ------------------------------------------------------------ #
    guard = BaseWeightGuard.capture(loaded.model)
    model = loaded.model

    if cfg.model.adapter_path:
        log_section(log, "adapter")
        try:
            model = load_adapter(model, cfg.model.adapter_path)
        except AdapterError as exc:
            log.error("%s", exc)
            return EXIT_USAGE
    elif args.attach_lora:
        from repairllama.model import attach_java_repair_adapter, check_java_repair_lora

        log_section(log, "LoRA (fresh Java repair adapter, not trained)")
        settings = LoRASettings.from_config(cfg.model.lora)
        try:
            model = attach_java_repair_adapter(model, settings, guard=guard)
        except AdapterError as exc:
            log.error("%s", exc)
            return EXIT_USAGE
        checks.extend(
            (check.name, check.ok, check.detail)
            for check in check_java_repair_lora(model, settings)
        )

    if model is not loaded.model:
        adapted = count_parameters(model)
        state = adapter_state(model)
        log.info("adapters: %s (active: %s)", state["adapters"], state["active_adapter"])
        log.info(
            "trainable after adapter: %s of %s (%.4f%%)",
            f"{adapted.trainable:,}",
            f"{adapted.total:,}",
            adapted.trainable_fraction * 100,
        )
        checks.append(
            ("adapter added trainable parameters", adapted.trainable > 0,
             f"{adapted.trainable:,} trainable")
        )
        changed = guard.changed(model)
        checks.append(
            ("base weights unchanged by the adapter", not changed, ", ".join(changed[:3]) or "ok")
        )

    log.info("dry run complete — nothing was trained and no weights were written")
    return _report_checks(checks)


def _report_checks(checks: List[Tuple[str, bool, str]]) -> int:
    """Print the verification table; non-zero exit if anything failed."""
    log_section(log, "verification")
    for name, ok, detail in checks:
        log.info("[%s] %-38s %s", "PASS" if ok else "FAIL", name, detail)
    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        log.error("%d check(s) failed: %s", len(failed), ", ".join(failed))
        return EXIT_USAGE
    log.info("all %d check(s) passed", len(checks))
    return EXIT_OK


def cmd_localize(args: argparse.Namespace) -> int:
    cfg = _resolve_config(args)
    _stage_banner(
        cfg,
        "localize",
        {
            "strategy": cfg.localization.strategy,
            "granularity": cfg.localization.granularity,
            "max candidates": cfg.localization.max_candidates,
            "project": args.project or "(from dataset)",
        },
    )
    return _not_implemented("localize", "repairllama.localization")


def cmd_train(args: argparse.Namespace) -> int:
    from repairllama.model.loader import ModelError, ModelNotAvailableError
    from repairllama.training import TrainingError, run_training

    cfg = _resolve_config(args)
    set_seed(cfg.training.seed)
    try:
        result = run_training(
            cfg,
            splits_dir=args.splits_dir,
            adapters_dir=args.adapters_dir,
            resume=args.resume,
            dry_run=args.dry_run,
            max_examples=args.max_examples,
        )
    except ModelNotAvailableError as exc:
        log.error("%s", exc)
        return EXIT_MODEL_UNAVAILABLE
    except (TrainingError, ModelError) as exc:
        log.error("%s", exc)
        return EXIT_USAGE
    except KeyboardInterrupt:
        log.warning("interrupted; resume with --resume auto")
        return 130

    if args.dry_run:
        log.info("dry run complete — nothing was trained")
        return EXIT_OK
    log.info("adapter written to %s", result.artifact_dir)
    return EXIT_OK


def cmd_infer(args: argparse.Namespace) -> int:
    cfg = _resolve_config(args)
    if args.num_candidates is not None:
        cfg = cfg.override({"inference.num_candidates": args.num_candidates})
    _stage_banner(
        cfg,
        "infer",
        {
            "base model": cfg.model.base_model,
            "adapter": cfg.model.adapter_path or "(none)",
            "input format": cfg.representation.input_format,
            "output format": cfg.representation.output_format,
            "candidates": cfg.inference.num_candidates,
            "temperature": cfg.inference.temperature,
            "input": args.input or "(unset)",
            "output": args.output or cfg.paths.resolve("output_dir") / "candidates.jsonl",
        },
    )
    return _not_implemented("infer", "repairllama.inference")


def cmd_patch(args: argparse.Namespace) -> int:
    cfg = _resolve_config(args)
    _stage_banner(
        cfg,
        "patch",
        {
            "strategy": cfg.patching.strategy,
            "validate syntax": cfg.patching.validate_syntax,
            "backup originals": cfg.patching.backup_originals,
            "candidates": args.candidates or "(unset)",
            "project": args.project or "(unset)",
        },
    )
    return _not_implemented("patch", "repairllama.patching")


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = _resolve_config(args)
    _stage_banner(
        cfg,
        "evaluate",
        {
            "build tool": cfg.evaluation.build_tool,
            "test timeout": f"{cfg.evaluation.test_timeout_seconds}s",
            "metrics": ", ".join(cfg.evaluation.metrics),
            "candidates": args.candidates or "(unset)",
            "report": args.report or cfg.evaluation.report_path,
        },
    )
    return _not_implemented("evaluate", "repairllama.evaluation")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_USAGE
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except StageNotImplemented as exc:
        print(f"not implemented: {exc}", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
