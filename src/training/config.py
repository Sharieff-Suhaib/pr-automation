"""Central configuration for the Agent-SWE program-repair model.

Every value can be overridden with an environment variable so the same code runs
locally, in Google Colab, or on Kaggle without editing source files.

Example:
    AGENT_SWE_EPOCHS=5 AGENT_SWE_LORA_R=32 python -m src.training.train
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Repository root: this file is src/training/config.py -> up three levels.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


@dataclass
class DataConfig:
    """Where the bug-fix data lives and how it is split."""

    # JSONL file with one bug-fix example per line (see data/raw/sample_bugs.jsonl).
    train_file: Path = field(
        default_factory=lambda: _env_path(
            "AGENT_SWE_TRAIN_FILE", PROJECT_ROOT / "data" / "raw" / "sample_bugs.jsonl"
        )
    )
    # Optional separate eval file. When unset we carve a holdout out of train_file.
    eval_file: Path | None = field(
        default_factory=lambda: (
            _env_path("AGENT_SWE_EVAL_FILE", PROJECT_ROOT)
            if os.environ.get("AGENT_SWE_EVAL_FILE")
            else None
        )
    )
    # Fraction of examples held out when eval_file is not given.
    eval_split: float = field(default_factory=lambda: _env_float("AGENT_SWE_EVAL_SPLIT", 0.2))
    seed: int = field(default_factory=lambda: _env_int("AGENT_SWE_SEED", 42))
    processed_dir: Path = field(
        default_factory=lambda: _env_path(
            "AGENT_SWE_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"
        )
    )


@dataclass
class ModelConfig:
    """Base model + quantization settings."""

    base_model: str = field(
        default_factory=lambda: _env_str("AGENT_SWE_BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    )
    max_seq_length: int = field(default_factory=lambda: _env_int("AGENT_SWE_MAX_SEQ_LEN", 1024))
    # 4-bit QLoRA. Requires CUDA + bitsandbytes; auto-disabled elsewhere by model_loader.
    load_in_4bit: bool = field(default_factory=lambda: _env_bool("AGENT_SWE_LOAD_IN_4BIT", True))
    bnb_4bit_quant_type: str = field(
        default_factory=lambda: _env_str("AGENT_SWE_BNB_QUANT_TYPE", "nf4")
    )
    bnb_4bit_use_double_quant: bool = field(
        default_factory=lambda: _env_bool("AGENT_SWE_BNB_DOUBLE_QUANT", True)
    )
    # "auto" picks bfloat16 on CUDA/MPS when supported, else float32.
    torch_dtype: str = field(default_factory=lambda: _env_str("AGENT_SWE_DTYPE", "auto"))
    trust_remote_code: bool = field(
        default_factory=lambda: _env_bool("AGENT_SWE_TRUST_REMOTE_CODE", False)
    )


@dataclass
class LoRAConfig:
    """Parameter-efficient fine-tuning settings (only these weights are trained)."""

    r: int = field(default_factory=lambda: _env_int("AGENT_SWE_LORA_R", 16))
    alpha: int = field(default_factory=lambda: _env_int("AGENT_SWE_LORA_ALPHA", 32))
    dropout: float = field(default_factory=lambda: _env_float("AGENT_SWE_LORA_DROPOUT", 0.05))
    # Qwen2 attention + MLP projections. Attention-only is cheaper: override with
    # AGENT_SWE_LORA_TARGETS="q_proj,k_proj,v_proj,o_proj".
    target_modules: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            _env_str(
                "AGENT_SWE_LORA_TARGETS",
                "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
            ).split(",")
        )
    )


@dataclass
class TrainConfig:
    """Optimizer / scheduler / checkpointing settings."""

    output_dir: Path = field(
        default_factory=lambda: _env_path(
            "AGENT_SWE_OUTPUT_DIR", PROJECT_ROOT / "outputs" / "repair-lora"
        )
    )
    num_train_epochs: float = field(default_factory=lambda: _env_float("AGENT_SWE_EPOCHS", 3.0))
    learning_rate: float = field(default_factory=lambda: _env_float("AGENT_SWE_LR", 2e-4))
    per_device_train_batch_size: int = field(
        default_factory=lambda: _env_int("AGENT_SWE_BATCH_SIZE", 1)
    )
    gradient_accumulation_steps: int = field(
        default_factory=lambda: _env_int("AGENT_SWE_GRAD_ACCUM", 8)
    )
    warmup_ratio: float = field(default_factory=lambda: _env_float("AGENT_SWE_WARMUP_RATIO", 0.03))
    weight_decay: float = field(default_factory=lambda: _env_float("AGENT_SWE_WEIGHT_DECAY", 0.0))
    lr_scheduler_type: str = field(
        default_factory=lambda: _env_str("AGENT_SWE_LR_SCHEDULER", "cosine")
    )
    logging_steps: int = field(default_factory=lambda: _env_int("AGENT_SWE_LOGGING_STEPS", 1))
    save_strategy: str = field(default_factory=lambda: _env_str("AGENT_SWE_SAVE_STRATEGY", "epoch"))
    # Trades compute for memory; keep on for small GPUs.
    gradient_checkpointing: bool = field(
        default_factory=lambda: _env_bool("AGENT_SWE_GRAD_CHECKPOINTING", True)
    )
    # Hard cap on optimizer steps. -1 = no cap. Set to a small number for smoke tests.
    max_steps: int = field(default_factory=lambda: _env_int("AGENT_SWE_MAX_STEPS", -1))
    seed: int = field(default_factory=lambda: _env_int("AGENT_SWE_SEED", 42))


@dataclass
class GenerationConfig:
    """Candidate patch sampling settings."""

    adapter_dir: Path = field(
        default_factory=lambda: _env_path(
            "AGENT_SWE_ADAPTER_DIR", PROJECT_ROOT / "outputs" / "repair-lora"
        )
    )
    max_new_tokens: int = field(default_factory=lambda: _env_int("AGENT_SWE_MAX_NEW_TOKENS", 512))
    # Number of candidate patches per bug (RepairLLaMA-style beam of candidates).
    num_candidates: int = field(default_factory=lambda: _env_int("AGENT_SWE_NUM_CANDIDATES", 1))
    temperature: float = field(default_factory=lambda: _env_float("AGENT_SWE_TEMPERATURE", 0.2))
    top_p: float = field(default_factory=lambda: _env_float("AGENT_SWE_TOP_P", 0.95))
    # Greedy decoding when only one candidate is requested; sampling otherwise.
    do_sample: bool = field(
        default_factory=lambda: _env_bool(
            "AGENT_SWE_DO_SAMPLE", _env_int("AGENT_SWE_NUM_CANDIDATES", 1) > 1
        )
    )


@dataclass
class Config:
    """Full experiment configuration; serialized next to the saved adapter."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def to_dict(self) -> dict:
        """JSON-serializable view (Paths -> str, tuples -> list)."""

        def _clean(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return list(value)
            if isinstance(value, dict):
                return {k: _clean(v) for k, v in value.items()}
            return value

        return {k: _clean(v) for k, v in asdict(self).items()}


def load_config() -> Config:
    """Build a Config from the current environment."""
    return Config()
