"""Loading the base causal LM — deliberately without downloading anything.

A CodeLlama-7B checkpoint is ~13 GB in 16-bit.  Pulling that automatically
because a config file names a repo id is the kind of surprise that fills a
laptop disk mid-command, so :func:`load_base_model` resolves a checkpoint from
local sources only and raises :class:`ModelNotAvailableError` — with the exact
commands to fix it — when there is nothing local to load.  Downloading happens
only when the caller explicitly sets ``allow_download``.

Resolution order for ``model.base_model``:

1. a local directory containing ``config.json`` (an absolute or relative path);
2. ``<models_dir>/<name>`` — so a checkpoint unpacked into the project's
   ``models/`` directory is found by its bare name;
3. the HuggingFace cache, if the repo has already been fetched;
4. otherwise: an error explaining all three.

The loader returns a :class:`LoadedModel` bundling the model, its tokenizer
and the resolved :class:`~repairllama.model.device.DeviceSpec`, and logs the
device, dtype, parameter counts and memory footprint on the way through.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from repairllama.model.device import (
    DeviceSpec,
    format_bytes,
    format_count,
    log_device_summary,
    memory_snapshot,
    resolve_device_spec,
)
from repairllama.utils.logging import get_logger

__all__ = [
    "ModelError",
    "ModelNotAvailableError",
    "LoadOptions",
    "ModelSource",
    "ParameterCounts",
    "LoadedModel",
    "resolve_model_source",
    "load_base_model",
    "count_parameters",
    "describe_missing_model",
]

log = get_logger("model.loader")

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf")


class ModelError(RuntimeError):
    """Raised when a model cannot be loaded."""


class ModelNotAvailableError(ModelError):
    """Raised when no local checkpoint could be found — never a download."""


# --------------------------------------------------------------------------- #
# options and results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LoadOptions:
    """Everything that decides how the base model is loaded."""

    base_model: str = "codellama/CodeLlama-7b-hf"
    tokenizer: str = ""  # empty == same as base_model
    revision: str = "main"
    dtype: str = "auto"
    device: str = "auto"
    models_dir: Optional[str] = None
    allow_download: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    padding_side: str = "right"
    extra_tokens: Tuple[str, ...] = ()
    low_cpu_mem_usage: bool = True
    freeze_base: bool = True

    @property
    def tokenizer_source(self) -> str:
        return self.tokenizer or self.base_model

    @property
    def local_files_only(self) -> bool:
        return not self.allow_download

    @classmethod
    def from_config(cls, cfg: Any) -> "LoadOptions":
        """Build from a :class:`repairllama.config.RepairConfig`."""
        representation = getattr(cfg, "representation", None)
        extra: Tuple[str, ...] = ()
        if representation is not None:
            extra = (representation.fill_token,)
        return cls(
            base_model=cfg.model.base_model,
            tokenizer=cfg.model.tokenizer,
            revision=cfg.model.revision,
            dtype=cfg.model.dtype,
            device=cfg.model.device,
            models_dir=str(cfg.paths.resolve("models_dir")),
            allow_download=cfg.model.allow_download,
            load_in_4bit=cfg.model.load_in_4bit,
            trust_remote_code=cfg.model.trust_remote_code,
            padding_side=cfg.model.padding_side,
            extra_tokens=extra,
        )


@dataclass(frozen=True)
class ModelSource:
    """Where a checkpoint was found."""

    reference: str  # what to hand to from_pretrained
    origin: str  # "local_path" | "models_dir" | "hf_cache" | "remote"
    requested: str
    searched: Tuple[str, ...] = ()

    @property
    def is_local(self) -> bool:
        return self.origin in {"local_path", "models_dir"}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference": self.reference,
            "origin": self.origin,
            "requested": self.requested,
            "is_local": self.is_local,
        }


@dataclass(frozen=True)
class ParameterCounts:
    """Parameter accounting for a model, in whatever state it is in."""

    total: int
    trainable: int
    frozen: int
    by_dtype: Dict[str, int] = field(default_factory=dict)

    @property
    def trainable_fraction(self) -> float:
        return (self.trainable / self.total) if self.total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "trainable": self.trainable,
            "frozen": self.frozen,
            "trainable_fraction": round(self.trainable_fraction, 6),
            "by_dtype": dict(self.by_dtype),
        }

    def render(self) -> str:
        return (
            f"total {format_count(self.total)} | "
            f"trainable {format_count(self.trainable)} "
            f"({self.trainable_fraction:.4%}) | "
            f"frozen {format_count(self.frozen)}"
        )


@dataclass
class LoadedModel:
    """A loaded model with its tokenizer and how it was loaded."""

    model: Any
    tokenizer: Any
    spec: DeviceSpec
    source: ModelSource
    options: LoadOptions
    tokenizer_changes: Any = None

    @property
    def is_peft(self) -> bool:
        return hasattr(self.model, "peft_config")

    def parameter_counts(self) -> ParameterCounts:
        return count_parameters(self.model)

    def memory_footprint(self) -> Optional[int]:
        """Bytes of parameters and buffers, as torch reports them."""
        getter = getattr(self.model, "get_memory_footprint", None)
        if getter is not None:
            try:
                return int(getter())
            except Exception:  # noqa: BLE001 - informational only
                pass
        try:
            return sum(
                parameter.numel() * parameter.element_size()
                for parameter in self.model.parameters()
            )
        except Exception:  # noqa: BLE001 - informational only
            return None

    def actual_device(self) -> str:
        """The device the parameters actually live on."""
        for parameter in self.model.parameters():
            return str(parameter.device)
        return "unknown"

    def actual_dtype(self) -> str:
        """The dtype of the first floating-point parameter."""
        for parameter in self.model.parameters():
            if parameter.is_floating_point():
                return str(parameter.dtype).replace("torch.", "")
        return "unknown"

    def describe(self) -> Dict[str, Any]:
        counts = self.parameter_counts()
        return {
            "source": self.source.to_dict(),
            "requested": self.spec.to_dict(),
            "actual_device": self.actual_device(),
            "actual_dtype": self.actual_dtype(),
            "parameters": counts.to_dict(),
            "memory_footprint": self.memory_footprint(),
            "is_peft": self.is_peft,
            "model_class": type(self.model).__name__,
        }

    def log_summary(self) -> None:
        counts = self.parameter_counts()
        log.info("model: %s from %s", type(self.model).__name__, self.source.reference)
        log.info("parameters: %s", counts.render())
        log.info(
            "footprint: %s on %s (%s)",
            format_bytes(self.memory_footprint()),
            self.actual_device(),
            self.actual_dtype(),
        )
        snapshot = memory_snapshot(self.spec.device)
        if "allocated" in snapshot:
            log.info(
                "device memory after load: %s allocated, %s reserved",
                format_bytes(snapshot.get("allocated")),
                format_bytes(snapshot.get("reserved")),
            )


# --------------------------------------------------------------------------- #
# finding a checkpoint, locally
# --------------------------------------------------------------------------- #
def _is_checkpoint_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False
    return any(
        entry.suffix in _WEIGHT_SUFFIXES for entry in path.iterdir() if entry.is_file()
    ) or (path / "model.safetensors.index.json").is_file()


def _hf_cache_dir(repo_id: str) -> Optional[Path]:
    """Locate ``repo_id`` in the local HuggingFace cache, without network."""
    roots: List[Path] = []
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value))
    home = os.environ.get("HF_HOME")
    roots.append(Path(home) / "hub" if home else Path.home() / ".cache" / "huggingface" / "hub")

    folder = "models--" + repo_id.replace("/", "--")
    for root in roots:
        candidate = root / folder / "snapshots"
        if not candidate.is_dir():
            continue
        snapshots = sorted(
            (entry for entry in candidate.iterdir() if entry.is_dir()),
            key=lambda entry: entry.stat().st_mtime,
            reverse=True,
        )
        for snapshot in snapshots:
            if (snapshot / "config.json").exists():
                return snapshot
    return None


def resolve_model_source(
    reference: str,
    models_dir: Optional[str] = None,
    allow_download: bool = False,
) -> ModelSource:
    """Find ``reference`` locally, or explain why it could not be found.

    See the module docstring for the search order.  With ``allow_download`` the
    reference is passed through untouched for ``transformers`` to fetch.
    """
    if not reference:
        raise ModelError("model.base_model is empty; name a checkpoint or a local path")

    searched: List[str] = []

    candidate = Path(reference).expanduser()
    searched.append(str(candidate))
    if _is_checkpoint_dir(candidate):
        return ModelSource(str(candidate), "local_path", reference, tuple(searched))

    if models_dir:
        local = Path(models_dir).expanduser() / Path(reference).name
        searched.append(str(local))
        if _is_checkpoint_dir(local):
            return ModelSource(str(local), "models_dir", reference, tuple(searched))

    if "/" in reference and not candidate.exists():
        cached = _hf_cache_dir(reference)
        searched.append(f"HuggingFace cache entry for {reference}")
        if cached is not None:
            return ModelSource(str(cached), "hf_cache", reference, tuple(searched))

    if allow_download:
        log.warning(
            "%s was not found locally; allow_download is set, so transformers "
            "will fetch it (this can be many GB)",
            reference,
        )
        return ModelSource(reference, "remote", reference, tuple(searched))

    raise ModelNotAvailableError(
        describe_missing_model(reference, "model", searched=searched, models_dir=models_dir)
    )


def describe_missing_model(
    reference: str,
    kind: str = "model",
    error: Optional[Exception] = None,
    searched: Optional[Sequence[str]] = None,
    models_dir: Optional[str] = None,
) -> str:
    """The message shown when a checkpoint is missing: what to do about it."""
    target = models_dir or "models/"
    name = Path(reference).name
    lines = [
        f"no local {kind} found for {reference!r}, and downloading is disabled.",
        "",
        "Checked:",
    ]
    for entry in searched or [reference]:
        lines.append(f"  - {entry}")
    lines += [
        "",
        "Fix it in one of these ways:",
        "",
        f"  1. Point the config at a checkpoint you already have:",
        f"       repairllama model-check --set model.base_model=/path/to/checkpoint",
        f"     (or edit model.base_model in configs/java_repair.yaml)",
        "",
        f"  2. Download it once, into the project's models directory:",
        f"       huggingface-cli download {reference} --local-dir {target}/{name}",
        f"     It is then found by its bare name, with no further configuration.",
        "",
        "  3. Allow this command to download it (many GB, not recommended on a",
        "     metered or small-disk machine):",
        "       repairllama model-check --set model.allow_download=true",
    ]
    if error is not None:
        lines += ["", f"Underlying error: {type(error).__name__}: {error}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# counting
# --------------------------------------------------------------------------- #
def count_parameters(model: Any) -> ParameterCounts:
    """Count total, trainable and frozen parameters, grouped by dtype."""
    total = trainable = 0
    by_dtype: Dict[str, int] = {}
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
        key = str(parameter.dtype).replace("torch.", "")
        by_dtype[key] = by_dtype.get(key, 0) + count
    return ParameterCounts(
        total=total, trainable=trainable, frozen=total - trainable, by_dtype=by_dtype
    )


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_base_model(
    options: Optional[LoadOptions] = None,
    *,
    load_tokenizer_too: bool = True,
    **overrides: Any,
) -> LoadedModel:
    """Load the base causal LM (and its tokenizer) onto the resolved device.

    The base model comes back **frozen** by default
    (``requires_grad=False`` everywhere): LoRA training must not update these
    weights, and freezing here means that holds from the moment the checkpoint
    is in memory, not from whenever an adapter happens to be attached.
    """
    from dataclasses import replace as _replace

    options = options or LoadOptions()
    if overrides:
        options = _replace(options, **overrides)

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ModelError(
            "torch and transformers are required to load a model. Install the "
            'model stack with `pip install -e ".[train]"`.'
        ) from exc

    spec = resolve_device_spec(options.device, options.dtype)
    log_device_summary(spec)

    source = resolve_model_source(
        options.base_model, options.models_dir, options.allow_download
    )
    log.info("resolved %s -> %s (%s)", options.base_model, source.reference, source.origin)

    tokenizer = None
    changes = None
    if load_tokenizer_too:
        from repairllama.model.tokenizer import load_tokenizer as _load_tokenizer

        tokenizer_source = options.tokenizer_source
        if tokenizer_source == options.base_model:
            tokenizer_source = source.reference
        tokenizer, changes = _load_tokenizer(
            tokenizer_source,
            revision=options.revision,
            local_files_only=options.local_files_only,
            trust_remote_code=options.trust_remote_code,
            padding_side=options.padding_side,
            extra_tokens=options.extra_tokens,
        )

    kwargs: Dict[str, Any] = {
        "revision": options.revision,
        "local_files_only": options.local_files_only,
        "trust_remote_code": options.trust_remote_code,
        "low_cpu_mem_usage": options.low_cpu_mem_usage,
    }
    _set_dtype_kwarg(kwargs, spec.torch_dtype)
    if options.load_in_4bit:
        kwargs["quantization_config"] = _quantization_config(spec)

    log.info("loading weights (this can take a while for a 7B checkpoint)")
    try:
        model = AutoModelForCausalLM.from_pretrained(source.reference, **kwargs)
    except Exception as exc:  # noqa: BLE001 - re-raised with guidance
        raise _translate_load_error(exc, source, options) from exc

    if tokenizer is not None and changes is not None and changes.needs_embedding_resize:
        log.info("resizing token embeddings to %d", len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))

    if not options.load_in_4bit:
        model.to(torch.device(spec.device))

    if options.freeze_base:
        frozen = freeze_parameters(model)
        log.info("froze %s base parameters (LoRA must not update them)", format_count(frozen))

    model.eval()
    loaded = LoadedModel(
        model=model,
        tokenizer=tokenizer,
        spec=spec,
        source=source,
        options=options,
        tokenizer_changes=changes,
    )
    loaded.log_summary()
    return loaded


def _dtype_kwarg_name() -> str:
    """``from_pretrained`` renamed ``torch_dtype`` to ``dtype`` in 4.56.

    Both take ``**kwargs``, so the signature cannot be inspected — the version
    decides.  Passing the wrong one is silently ignored on old versions and
    warns on new ones, which is exactly the sort of "loaded in the wrong
    precision" bug this module exists to prevent.
    """
    import transformers

    try:
        major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    except ValueError:  # pragma: no cover - unusual version strings
        return "dtype"
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def _set_dtype_kwarg(kwargs: Dict[str, Any], torch_dtype: Any) -> None:
    """Pass the dtype under the name this transformers version expects."""
    kwargs[_dtype_kwarg_name()] = torch_dtype


def _quantization_config(spec: DeviceSpec) -> Any:
    """4-bit config, with a clear error when the machine cannot do it."""
    if not spec.device.startswith("cuda"):
        raise ModelError(
            f"load_in_4bit requires a CUDA device, but the resolved device is "
            f"{spec.device}. Set model.load_in_4bit=false, or use device: cuda."
        )
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise ModelError(
            "load_in_4bit requires bitsandbytes. Install it with "
            '`pip install -e ".[quant]"`, or set model.load_in_4bit=false.'
        ) from exc
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=spec.torch_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _translate_load_error(
    exc: Exception, source: ModelSource, options: LoadOptions
) -> ModelError:
    from repairllama.model.tokenizer import _looks_like_a_missing_checkpoint

    if _looks_like_a_missing_checkpoint(exc):
        return ModelNotAvailableError(
            describe_missing_model(
                options.base_model,
                "model",
                exc,
                searched=source.searched,
                models_dir=options.models_dir,
            )
        )
    text = str(exc).lower()
    if "out of memory" in text or "can't allocate" in text:
        return ModelError(
            f"ran out of memory loading {source.reference}. Try a smaller dtype "
            "(model.dtype=float16), 4-bit loading (model.load_in_4bit=true, CUDA "
            f"only), or a smaller base model.\nUnderlying error: {exc}"
        )
    return ModelError(f"could not load the model from {source.reference}: {exc}")


def freeze_parameters(model: Any) -> int:
    """Set ``requires_grad=False`` on every parameter; return how many."""
    frozen = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.requires_grad_(False)
        frozen += parameter.numel()
    return frozen
