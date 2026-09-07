#!/usr/bin/env python3
"""Fine-tune the Java repair LoRA adapter.

    python scripts/train_java_repair.py                       # config defaults
    python scripts/train_java_repair.py --dry-run             # set up, don't train
    python scripts/train_java_repair.py --epochs 2 --lr 5e-4
    python scripts/train_java_repair.py --resume auto         # newest checkpoint
    python scripts/train_java_repair.py --max-examples 32     # smoke run

Everything is configurable through ``configs/java_repair.yaml``; the flags here
are shortcuts for the settings changed most often, and ``--set key=value``
reaches any of the rest.

The base model is never modified and only the adapter is saved, to
``adapters/java-repair/``.  Requires a local checkpoint — nothing is
downloaded; see ``repairllama model-check`` if the model cannot be found.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Run from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repairllama.config import (  # noqa: E402
    ConfigError,
    load_config,
    parse_override_value,
)
from repairllama.utils.logging import configure_from_config, get_logger  # noqa: E402
from repairllama.utils.paths import default_config_path  # noqa: E402

log = get_logger("scripts.train")

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_MODEL_UNAVAILABLE = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_java_repair.py",
        description="Fine-tune the RepairLLaMA Java repair LoRA adapter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config", default=None, help="YAML config file")
    parser.add_argument(
        "-s",
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="override any config value, e.g. --set training.warmup_ratio=0.1",
    )
    parser.add_argument("--splits-dir", default=None, help="where the JSONL splits live")
    parser.add_argument("--adapters-dir", default=None, help="parent of java-repair/")
    parser.add_argument("--output-dir", default=None, help="where checkpoints are written")
    parser.add_argument("--model", default=None, help="base checkpoint path or id")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--mixed-precision", choices=["auto", "no", "fp16", "bf16"], default=None
    )
    parser.add_argument("--device", default=None, help="auto | cpu | cuda | cuda:<n> | mps")

    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="resume training: 'auto' for the newest checkpoint, or a path",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="cap the training rows (0 = all), for smoke runs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="load everything and print the banner, but do not train",
    )
    parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING"])
    return parser


def _overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    for pair in args.overrides or []:
        if "=" not in pair:
            raise ConfigError(f"--set expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        overrides[key.strip()] = parse_override_value(raw)

    for flag, key in (
        ("epochs", "training.epochs"),
        ("lr", "training.learning_rate"),
        ("batch_size", "training.batch_size"),
        ("gradient_accumulation", "training.gradient_accumulation_steps"),
        ("max_length", "training.max_length"),
        ("seed", "training.seed"),
        ("mixed_precision", "training.mixed_precision"),
        ("model", "model.base_model"),
        ("device", "model.device"),
        ("log_level", "logging.level"),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            overrides[key] = value
    return overrides


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config_path = args.config
    if config_path is None:
        shipped = default_config_path()
        config_path = str(shipped) if shipped.is_file() else None

    try:
        cfg = load_config(config_path, _overrides(args))
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    configure_from_config(cfg)

    from repairllama.model.loader import ModelError, ModelNotAvailableError
    from repairllama.training import TrainingError, run_training

    try:
        result = run_training(
            cfg,
            splits_dir=args.splits_dir,
            adapters_dir=args.adapters_dir,
            output_dir=args.output_dir,
            resume=args.resume,
            dry_run=args.dry_run,
            max_examples=args.max_examples,
        )
    except ModelNotAvailableError as exc:
        log.error("%s", exc)
        return EXIT_MODEL_UNAVAILABLE
    except (TrainingError, ModelError, ConfigError) as exc:
        log.error("%s", exc)
        return EXIT_USAGE
    except KeyboardInterrupt:
        log.warning("interrupted; the newest checkpoint can be resumed with --resume auto")
        return 130

    if args.dry_run:
        log.info("dry run complete — nothing was trained")
        return EXIT_OK

    log.info("adapter written to %s", result.artifact_dir)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
