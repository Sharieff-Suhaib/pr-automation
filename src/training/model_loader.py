"""Load Qwen2.5-Coder + tokenizer with QLoRA when possible, sane fallbacks otherwise.

Hardware reality this module handles:

  * CUDA + bitsandbytes  -> true 4-bit QLoRA (the intended path; Colab/Kaggle).
  * Apple Silicon (MPS)  -> no bitsandbytes; falls back to bf16/fp16 LoRA.
  * CPU only             -> fp32 LoRA; fine for smoke tests, too slow for real runs.

The fallback is automatic and loudly announced, so a run never silently trains in
a mode you did not expect.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.training.config import LoRAConfig, ModelConfig


def detect_device() -> str:
    """Return the best available compute device: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def bitsandbytes_available() -> bool:
    """True only when bitsandbytes imports AND the device can actually use it."""
    if detect_device() != "cuda":
        return False
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return False
    return True


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    """Map the configured dtype string to a torch dtype for this device."""
    if requested != "auto":
        return getattr(torch, requested)
    if device == "cuda":
        # bf16 needs Ampere or newer; older cards (e.g. Colab T4) use fp16.
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32  # CPU: fp16 matmuls are unsupported/slow


def dtype_kwarg(dtype: torch.dtype) -> dict:
    """Return the dtype kwarg under the name this transformers version expects.

    transformers 4.56 renamed `torch_dtype` -> `dtype`. Passing the wrong one is
    not an error — it is silently absorbed into the config, and the model loads
    in fp32 instead. So pick the name explicitly rather than guessing.
    """
    try:
        from importlib.metadata import version

        major, minor = (int(part) for part in version("transformers").split(".")[:2])
        renamed = (major, minor) >= (4, 56)
    except Exception:
        renamed = False  # unknown version: the old name is the safer default
    return {"dtype": dtype} if renamed else {"torch_dtype": dtype}


def load_tokenizer(config: ModelConfig):
    """Load the tokenizer and make sure padding is well-defined.

    Qwen ships a pad token, but we fall back to EOS if a variant does not, and we
    left-pad because generation requires it for correct batched decoding.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model, trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_quantization_config(config: ModelConfig):
    """BitsAndBytesConfig for 4-bit NF4 QLoRA, or None if unsupported/disabled."""
    if not config.load_in_4bit:
        return None

    device = detect_device()
    if not bitsandbytes_available():
        reason = (
            "no CUDA GPU detected" if device != "cuda" else "bitsandbytes is not installed/usable"
        )
        print(
            f"[model_loader] 4-bit QLoRA requested but unavailable ({reason}).\n"
            f"[model_loader] Falling back to standard LoRA on '{device}'. "
            f"This uses more memory; reduce AGENT_SWE_MAX_SEQ_LEN or "
            f"AGENT_SWE_BATCH_SIZE if you hit OOM."
        )
        return None

    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,  # nf4 = normal-float 4-bit
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,  # quantize the quant constants
        bnb_4bit_compute_dtype=resolve_dtype(config.torch_dtype, "cuda"),
    )


def load_base_model(config: ModelConfig, for_training: bool = True):
    """Load the base causal LM, quantized when the environment supports it."""
    from transformers import AutoModelForCausalLM

    device = detect_device()
    dtype = resolve_dtype(config.torch_dtype, device)
    quantization_config = build_quantization_config(config)

    print(
        f"[model_loader] loading {config.base_model} "
        f"(device={device}, dtype={dtype}, 4bit={quantization_config is not None})"
    )

    kwargs = {
        **dtype_kwarg(dtype),
        "trust_remote_code": config.trust_remote_code,
    }
    if quantization_config is not None:
        # accelerate places the quantized shards; device_map is required here.
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(config.base_model, **kwargs)

    if quantization_config is None:
        model = model.to(device)

    if for_training:
        # Cache and training-time gradient checkpointing are incompatible.
        model.config.use_cache = False
    return model


def apply_lora(model, lora_config: LoRAConfig, gradient_checkpointing: bool = True):
    """Freeze the base model and attach trainable LoRA adapters."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # For quantized models this casts norms/embeddings to fp32 and enables input
    # grads so gradients flow through the frozen 4-bit layers.
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    elif gradient_checkpointing:
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=list(lora_config.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config


def load_model_with_adapter(config: ModelConfig, adapter_dir: Path | str):
    """Load base model + a saved LoRA adapter for inference."""
    from peft import PeftModel

    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"No LoRA adapter found in {adapter_dir}.\n"
            f"Train one first (python -m src.training.train), or point "
            f"AGENT_SWE_ADAPTER_DIR at an existing adapter directory."
        )

    model = load_base_model(config, for_training=False)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    model.config.use_cache = True
    print(f"[model_loader] loaded LoRA adapter from {adapter_dir}")
    return model
