"""Configuration schema for the RepairLLaMA-Java pipeline.

The schema is expressed as plain dataclasses so the package stays dependency
light (only PyYAML is needed to read a config file).  Every section validates
itself in ``__post_init__`` so a bad YAML file fails at load time with a
message that names the offending key, rather than deep inside training.

Typical use::

    from repairllama.config import RepairConfig

    cfg = RepairConfig.from_yaml("configs/java_repair.yaml")
    print(cfg.model.base_model)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Type, TypeVar, get_type_hints

import yaml

__all__ = [
    "ConfigError",
    "PathsConfig",
    "DataConfig",
    "RepresentationConfig",
    "LocalizationConfig",
    "ModelConfig",
    "LoRAConfig",
    "TrainingConfig",
    "InferenceConfig",
    "PatchingConfig",
    "EvaluationConfig",
    "LoggingConfig",
    "RepairConfig",
    "load_config",
]


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or has invalid values."""


T = TypeVar("T")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _one_of(value: Any, allowed: List[Any], key: str) -> None:
    _require(
        value in allowed,
        f"{key}: expected one of {allowed}, got {value!r}",
    )


def _positive(value: Any, key: str) -> None:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0,
        f"{key}: expected a positive number, got {value!r}",
    )


def _non_negative(value: Any, key: str) -> None:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0,
        f"{key}: expected a non-negative number, got {value!r}",
    )


def _from_mapping(cls: Type[T], data: Mapping[str, Any], section: str) -> T:
    """Build a dataclass from a mapping, rejecting unknown keys."""
    if data is None:
        data = {}
    _require(
        isinstance(data, Mapping),
        f"{section}: expected a mapping, got {type(data).__name__}",
    )
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - known)
    _require(
        not unknown,
        f"{section}: unknown key(s) {unknown}; valid keys are {sorted(known)}",
    )
    # ``from __future__ import annotations`` turns field types into strings, so
    # resolve them before checking for nested dataclass sections.
    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):  # type: ignore[arg-type]
        if f.name not in data:
            continue
        value = data[f.name]
        hint = hints.get(f.name)
        if is_dataclass(hint) and isinstance(value, Mapping):
            kwargs[f.name] = _from_mapping(hint, value, f"{section}.{f.name}")
        else:
            kwargs[f.name] = value
    return cls(**kwargs)  # type: ignore[call-arg]


def _as_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _as_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, list):
        return [_as_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _as_dict(v) for k, v in obj.items()}
    return obj


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
@dataclass
class PathsConfig:
    """Filesystem layout.  Relative paths resolve against the project root."""

    project_root: str = "."
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    splits_dir: str = "data/splits"
    models_dir: str = "models"
    adapters_dir: str = "adapters"  # one directory per language beneath it
    output_dir: str = "outputs"
    log_dir: str = "outputs/logs"

    def resolve(self, key: str) -> Path:
        """Return an absolute :class:`~pathlib.Path` for one of the path keys."""
        known = {f.name for f in fields(self)} - {"project_root"}
        _require(key in known, f"paths: unknown path key {key!r}; valid: {sorted(known)}")
        root = Path(self.project_root).expanduser()
        value = Path(getattr(self, key)).expanduser()
        return value if value.is_absolute() else (root / value).resolve()


@dataclass
class DataConfig:
    """Where repair examples come from and how they are split."""

    dataset_name: str = "megadiff-java"
    language: str = "java"
    max_examples: int = 0  # 0 == no cap
    train_split: float = 0.9
    val_split: float = 0.05
    test_split: float = 0.05
    shuffle_seed: int = 42
    min_diff_lines: int = 1
    max_diff_lines: int = 50
    drop_test_only_changes: bool = True
    require_single_function: bool = True
    deduplicate: bool = True
    dedup_strategy: str = "normalized"  # exact | normalized | input_only
    split_by_project: bool = True
    minimal_records: bool = False  # emit only {"input", "output"} in the JSONL
    use_model_tokenizer: bool = False  # count tokens with the real tokenizer
    loader_format: str = "jsonl"  # jsonl | json | directory

    def __post_init__(self) -> None:
        _one_of(self.language, ["java"], "data.language")
        _one_of(
            self.dedup_strategy,
            ["exact", "normalized", "input_only"],
            "data.dedup_strategy",
        )
        _one_of(
            self.loader_format, ["jsonl", "json", "directory"], "data.loader_format"
        )
        _non_negative(self.max_examples, "data.max_examples")
        for key in ("train_split", "val_split", "test_split"):
            value = getattr(self, key)
            _require(
                isinstance(value, (int, float)) and 0.0 <= value <= 1.0,
                f"data.{key}: expected a fraction in [0, 1], got {value!r}",
            )
        total = self.train_split + self.val_split + self.test_split
        _require(
            abs(total - 1.0) < 1e-6,
            f"data: train_split + val_split + test_split must equal 1.0, got {total}",
        )
        _positive(self.max_diff_lines, "data.max_diff_lines")
        _require(
            self.min_diff_lines <= self.max_diff_lines,
            "data: min_diff_lines must be <= max_diff_lines",
        )


@dataclass
class RepresentationConfig:
    """Input/output representation, following the RepairLLaMA IR/OR study.

    ``input_format`` names an input representation (IR1..IR4) and
    ``output_format`` an output representation (OR1..OR4).  The implemented
    pair is IR4 x OR2: the buggy lines are kept but commented out, marked by
    start/end comments, and followed by a ``fill_token`` the model replaces;
    the target is only the replacement code for that region.  See
    ``repairllama.representation``.
    """

    input_format: str = "ir4"  # buggy lines commented out + fill token
    output_format: str = "or2"  # replacement hunk only
    fill_token: str = "<FILL_ME>"
    region_start_marker: str = "// buggy lines start here"
    region_end_marker: str = "// buggy lines end here"
    comment_prefix: str = "//"
    context_lines: int = 10
    show_line_numbers: bool = False
    include_file_path: bool = True
    include_test_failure: bool = False
    dedent_target: bool = False
    max_input_tokens: int = 1024
    max_output_tokens: int = 512

    def __post_init__(self) -> None:
        _one_of(
            self.input_format,
            ["ir1", "ir2", "ir3", "ir4"],
            "representation.input_format",
        )
        _one_of(
            self.output_format,
            ["or1", "or2", "or3", "or4"],
            "representation.output_format",
        )
        _require(
            bool(self.fill_token), "representation.fill_token: must not be empty"
        )
        _require(
            bool(self.comment_prefix),
            "representation.comment_prefix: must not be empty",
        )
        _non_negative(self.context_lines, "representation.context_lines")
        _positive(self.max_input_tokens, "representation.max_input_tokens")
        _positive(self.max_output_tokens, "representation.max_output_tokens")


@dataclass
class LocalizationConfig:
    """How the buggy region is found before a prompt is built."""

    strategy: str = "perfect"  # perfect | stacktrace | spectrum
    granularity: str = "hunk"  # line | hunk | function | file
    max_candidates: int = 5

    def __post_init__(self) -> None:
        _one_of(
            self.strategy,
            ["perfect", "stacktrace", "spectrum"],
            "localization.strategy",
        )
        _one_of(
            self.granularity,
            ["line", "hunk", "function", "file"],
            "localization.granularity",
        )
        _positive(self.max_candidates, "localization.max_candidates")


@dataclass
class LoRAConfig:
    """Parameter-efficient fine-tuning settings.

    The defaults are the RepairLLaMA paper's reported configuration:
    r=8, alpha=16, dropout=0.05, adapting the query and value projections
    only.  See ``repairllama.model.lora``.
    """

    enabled: bool = True
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self) -> None:
        _positive(self.r, "model.lora.r")
        _positive(self.alpha, "model.lora.alpha")
        _require(
            0.0 <= self.dropout < 1.0,
            f"model.lora.dropout: expected [0, 1), got {self.dropout!r}",
        )
        _require(
            isinstance(self.target_modules, list) and all(
                isinstance(m, str) for m in self.target_modules
            ),
            "model.lora.target_modules: expected a list of strings",
        )
        _one_of(self.bias, ["none", "all", "lora_only"], "model.lora.bias")


@dataclass
class ModelConfig:
    """Base checkpoint and how it is loaded."""

    base_model: str = "codellama/CodeLlama-7b-hf"
    tokenizer: str = ""  # empty == same as base_model
    revision: str = "main"
    dtype: str = "auto"  # auto | float16 | bfloat16 | float32
    device: str = "auto"  # auto | cpu | cuda | mps
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    allow_download: bool = False  # never fetch a checkpoint unless set
    padding_side: str = "right"   # right for training, left for generation
    adapter_path: str = ""  # empty == no adapter loaded
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    def __post_init__(self) -> None:
        _require(bool(self.base_model), "model.base_model: must not be empty")
        _one_of(self.dtype, ["auto", "float16", "bfloat16", "float32"], "model.dtype")
        _require(
            self.device in ["auto", "cpu", "cuda", "mps"]
            or self.device.startswith("cuda:"),
            f"model.device: expected auto|cpu|cuda|mps|cuda:<index>, got {self.device!r}",
        )
        _one_of(self.padding_side, ["left", "right"], "model.padding_side")

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer or self.base_model


@dataclass
class TrainingConfig:
    """Supervised fine-tuning hyper-parameters (not yet implemented)."""

    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    seed: int = 42
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 3
    resume_from_checkpoint: str = ""

    def __post_init__(self) -> None:
        _positive(self.epochs, "training.epochs")
        _positive(self.batch_size, "training.batch_size")
        _positive(
            self.gradient_accumulation_steps, "training.gradient_accumulation_steps"
        )
        _positive(self.learning_rate, "training.learning_rate")
        _non_negative(self.weight_decay, "training.weight_decay")
        _require(
            0.0 <= self.warmup_ratio <= 1.0,
            f"training.warmup_ratio: expected [0, 1], got {self.warmup_ratio!r}",
        )
        _one_of(
            self.lr_scheduler,
            ["linear", "cosine", "constant", "constant_with_warmup"],
            "training.lr_scheduler",
        )

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class InferenceConfig:
    """Patch sampling settings."""

    num_candidates: int = 10
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    do_sample: bool = True
    num_beams: int = 1
    batch_size: int = 1
    stop_sequences: List[str] = field(default_factory=list)
    deduplicate: bool = True

    def __post_init__(self) -> None:
        _positive(self.num_candidates, "inference.num_candidates")
        _positive(self.max_new_tokens, "inference.max_new_tokens")
        _non_negative(self.temperature, "inference.temperature")
        _require(
            0.0 < self.top_p <= 1.0,
            f"inference.top_p: expected (0, 1], got {self.top_p!r}",
        )
        _positive(self.num_beams, "inference.num_beams")
        _require(
            self.do_sample or self.num_beams > 1,
            "inference: greedy decoding (do_sample=false, num_beams=1) can only "
            "produce one candidate; set do_sample=true or num_beams>1",
        )


@dataclass
class PatchingConfig:
    """How a generated hunk is spliced back into the source file."""

    strategy: str = "hunk_replace"  # hunk_replace | full_function | unified_diff
    validate_syntax: bool = True
    normalize_whitespace: bool = True
    keep_original_indentation: bool = True
    backup_originals: bool = True
    max_patch_size_lines: int = 200

    def __post_init__(self) -> None:
        _one_of(
            self.strategy,
            ["hunk_replace", "full_function", "unified_diff"],
            "patching.strategy",
        )
        _positive(self.max_patch_size_lines, "patching.max_patch_size_lines")


@dataclass
class EvaluationConfig:
    """Plausible/correct patch evaluation."""

    build_tool: str = "maven"  # maven | gradle | none
    test_command: str = ""  # empty == derive from build_tool
    test_timeout_seconds: int = 600
    compile_only: bool = False
    metrics: List[str] = field(
        default_factory=lambda: ["compilable", "plausible", "exact_match"]
    )
    report_path: str = "outputs/reports/evaluation.json"

    def __post_init__(self) -> None:
        _one_of(self.build_tool, ["maven", "gradle", "none"], "evaluation.build_tool")
        _positive(self.test_timeout_seconds, "evaluation.test_timeout_seconds")
        allowed = {"compilable", "plausible", "exact_match", "ast_match", "pass_at_k"}
        unknown = sorted(set(self.metrics) - allowed)
        _require(
            not unknown,
            f"evaluation.metrics: unknown metric(s) {unknown}; valid: {sorted(allowed)}",
        )


@dataclass
class LoggingConfig:
    """Console/file logging behaviour; consumed by ``repairllama.utils.logging``."""

    level: str = "INFO"
    format: str = "text"  # text | json
    log_to_file: bool = True
    file_name: str = "repairllama.log"
    color: bool = True

    def __post_init__(self) -> None:
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        _require(
            isinstance(self.level, str) and self.level.upper() in levels,
            f"logging.level: expected one of {levels}, got {self.level!r}",
        )
        self.level = self.level.upper()
        _one_of(self.format, ["text", "json"], "logging.format")


# --------------------------------------------------------------------------- #
# root
# --------------------------------------------------------------------------- #
@dataclass
class RepairConfig:
    """Root configuration object for every stage of the pipeline."""

    experiment_name: str = "java-repair"
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    patching: PatchingConfig = field(default_factory=PatchingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        _require(
            bool(self.experiment_name), "experiment_name: must not be empty"
        )

    # -- constructors ------------------------------------------------------- #
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RepairConfig":
        return _from_mapping(cls, data or {}, "config")

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> "RepairConfig":
        path = Path(path).expanduser()
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        _require(
            isinstance(raw, Mapping),
            f"{path}: top level of a config file must be a mapping",
        )
        cfg = cls.from_dict(raw)
        # A config file's relative paths are relative to the file's project,
        # not to the caller's working directory.
        if cfg.paths.project_root == ".":
            cfg.paths.project_root = str(path.resolve().parent.parent)
        return cfg

    @classmethod
    def default(cls) -> "RepairConfig":
        return cls()

    # -- serialisation ------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)

    def to_yaml(self, path: str | os.PathLike[str] | None = None) -> str:
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)
        if path is not None:
            target = Path(path).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        return text

    def override(self, dotted: Mapping[str, Any]) -> "RepairConfig":
        """Return a new config with ``{"training.epochs": 5}``-style overrides."""
        data = self.to_dict()
        for key, value in dotted.items():
            if value is None:
                continue
            cursor: Any = data
            parts = key.split(".")
            for part in parts[:-1]:
                _require(
                    isinstance(cursor, dict) and part in cursor,
                    f"override {key!r}: no such config section {part!r}",
                )
                cursor = cursor[part]
            _require(
                isinstance(cursor, dict) and parts[-1] in cursor,
                f"override {key!r}: no such config key",
            )
            cursor[parts[-1]] = value
        return RepairConfig.from_dict(data)


def load_config(
    path: str | os.PathLike[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> RepairConfig:
    """Load a config from ``path`` (or defaults) and apply dotted overrides."""
    cfg = RepairConfig.from_yaml(path) if path else RepairConfig.default()
    return cfg.override(overrides) if overrides else cfg
