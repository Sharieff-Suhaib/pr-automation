"""Base checkpoint, tokenizer and LoRA adapter loading.

    device.py     resolve where the model goes and in what precision
    tokenizer.py  load the tokenizer and its special tokens
    loader.py     find a checkpoint locally and load the base causal LM
    adapter.py    attach / load / save / unload LoRA adapters (generic)
    lora.py       the Java repair adapter: the paper's r=8 / alpha=16 /
                  q_proj+v_proj configuration, its assertions, and the
                  per-language adapters/<lang>-repair layout

Typical use::

    from repairllama.model import LoadOptions, load_base_model, attach_lora

    loaded = load_base_model(LoadOptions.from_config(cfg))
    model = attach_java_repair_adapter(loaded.model)   # r=8, q_proj + v_proj
    save_adapter(model, cfg.paths.resolve("adapters_dir"))  # adapters/java-repair

Two rules this package enforces rather than assumes:

* **Nothing is downloaded.** A checkpoint is resolved from a local path, the
  project's ``models/`` directory or the HuggingFace cache; if none has it,
  the error explains how to configure the path instead of fetching gigabytes.
* **The base model is never modified.** Parameters are frozen at load time,
  ``attach_lora`` verifies that only adapter tensors are trainable, and
  :class:`BaseWeightGuard` can prove after the fact that no base weight moved.

Heavy imports (torch, transformers, peft) happen inside functions, so
importing this package costs nothing in an environment without them.
"""

from repairllama.model.adapter import (
    AdapterError,
    BaseWeightGuard,
    BaseWeightsModified,
    LoRASettings,
    adapter_state,
    attach_lora,
    is_adapter_parameter,
    load_adapter,
    trainable_parameter_names,
    unload_adapter,
)
from repairllama.model.device import (
    DeviceError,
    DeviceSpec,
    available_devices,
    describe_device,
    format_bytes,
    format_count,
    log_device_summary,
    memory_snapshot,
    resolve_device,
    resolve_device_spec,
    resolve_dtype,
    torch_available,
)
from repairllama.model.loader import (
    LoadedModel,
    LoadOptions,
    ModelError,
    ModelNotAvailableError,
    ModelSource,
    ParameterCounts,
    count_parameters,
    describe_missing_model,
    freeze_parameters,
    load_base_model,
    resolve_model_source,
)
from repairllama.model.lora import (
    ADAPTER_PROFILES,
    JAVA_REPAIR,
    PLANNED_LANGUAGES,
    AdapterProfile,
    LoRAAssertionError,
    LoRACheck,
    TrainableSummary,
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
from repairllama.model.tokenizer import (
    TokenizerError,
    describe_tokenizer,
    ensure_pad_token,
    ensure_representation_tokens,
    load_tokenizer,
    log_tokenizer_summary,
)

__all__ = [
    # device
    "DeviceError",
    "DeviceSpec",
    "resolve_device",
    "resolve_dtype",
    "resolve_device_spec",
    "available_devices",
    "describe_device",
    "memory_snapshot",
    "log_device_summary",
    "format_bytes",
    "format_count",
    "torch_available",
    # tokenizer
    "TokenizerError",
    "load_tokenizer",
    "ensure_pad_token",
    "ensure_representation_tokens",
    "describe_tokenizer",
    "log_tokenizer_summary",
    # loader
    "ModelError",
    "ModelNotAvailableError",
    "LoadOptions",
    "ModelSource",
    "LoadedModel",
    "ParameterCounts",
    "load_base_model",
    "resolve_model_source",
    "count_parameters",
    "freeze_parameters",
    "describe_missing_model",
    # adapter
    "AdapterError",
    "BaseWeightsModified",
    "LoRASettings",
    "BaseWeightGuard",
    "attach_lora",
    "load_adapter",
    "unload_adapter",
    "adapter_state",
    "trainable_parameter_names",
    "is_adapter_parameter",
    # lora (the Java repair adapter)
    "JAVA_REPAIR",
    "ADAPTER_PROFILES",
    "PLANNED_LANGUAGES",
    "AdapterProfile",
    "LoRAAssertionError",
    "LoRACheck",
    "TrainableSummary",
    "create_lora_config",
    "attach_java_repair_adapter",
    "print_trainable_parameters",
    "save_adapter",
    "load_java_repair_adapter",
    "assert_java_repair_lora",
    "check_java_repair_lora",
    "adapter_directory",
    "available_languages",
    "profile_for",
]
