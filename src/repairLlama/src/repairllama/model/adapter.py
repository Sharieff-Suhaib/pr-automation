"""LoRA adapters: attaching, loading, saving, unloading — and proving the
base weights never move.

The whole point of LoRA is that the 7B base stays fixed while a few million
adapter parameters learn the task.  That is easy to assume and easy to get
wrong (an unfrozen embedding layer, a merge called by accident, a checkpoint
saved with the base folded in), so this module makes it checkable:

    guard = BaseWeightGuard.capture(model)   # hashes a sample of base weights
    model = attach_lora(model, settings)     # base frozen, adapter trainable
    ...
    guard.verify(model)                      # raises if a base weight changed

:func:`attach_lora` freezes every base parameter *before* handing the model to
PEFT and asserts afterwards that the only trainable tensors are adapter ones,
so a misconfigured ``target_modules`` cannot quietly leave the base unfrozen.

The four operations the pipeline needs:

    attach_lora(model, settings)     add a fresh, randomly-initialised adapter
    load_adapter(model, path)        attach a trained adapter from disk
    save_adapter(model, path)        write the adapter only, never the base
    unload_adapter(model)            get the untouched base model back
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from repairllama.model.device import format_count
from repairllama.model.loader import ParameterCounts, count_parameters, freeze_parameters
from repairllama.utils.logging import get_logger

__all__ = [
    "AdapterError",
    "BaseWeightsModified",
    "LoRASettings",
    "BaseWeightGuard",
    "attach_lora",
    "load_adapter",
    "unload_adapter",
    "save_adapter",
    "adapter_state",
    "trainable_parameter_names",
]

log = get_logger("model.adapter")

_ADAPTER_MARKERS = ("lora_", "adapter", "modules_to_save")


class AdapterError(RuntimeError):
    """Raised when an adapter cannot be attached, loaded or removed."""


class BaseWeightsModified(AssertionError):
    """Raised when base weights changed under a guard that forbade it."""


@dataclass(frozen=True)
class LoRASettings:
    """LoRA hyper-parameters, mirroring ``model.lora`` in the config."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self) -> None:
        if self.r < 1:
            raise AdapterError(f"lora.r must be >= 1, got {self.r}")
        if self.alpha < 1:
            raise AdapterError(f"lora.alpha must be >= 1, got {self.alpha}")
        if not 0.0 <= self.dropout < 1.0:
            raise AdapterError(f"lora.dropout must be in [0, 1), got {self.dropout}")
        if not self.target_modules:
            raise AdapterError("lora.target_modules must not be empty")

    @classmethod
    def from_config(cls, cfg: Any) -> "LoRASettings":
        """Build from a :class:`repairllama.config.LoRAConfig`."""
        return cls(
            r=cfg.r,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
            target_modules=tuple(cfg.target_modules),
            bias=cfg.bias,
            task_type=cfg.task_type,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "bias": self.bias,
            "task_type": self.task_type,
        }


def _require_peft(action: str):
    try:
        import peft
    except ImportError as exc:
        raise AdapterError(
            f"peft is required to {action}. Install the model stack with "
            '`pip install -e ".[train]"` (or `pip install peft`).'
        ) from exc
    return peft


def is_adapter_parameter(name: str) -> bool:
    """True when a parameter name belongs to an adapter rather than the base."""
    return any(marker in name for marker in _ADAPTER_MARKERS)


def trainable_parameter_names(model: Any) -> List[str]:
    """Names of every parameter currently marked trainable."""
    return [name for name, param in model.named_parameters() if param.requires_grad]


# --------------------------------------------------------------------------- #
# the guarantee: base weights do not move
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BaseWeightGuard:
    """Fingerprints of base weights, so a change can be proven or ruled out.

    Hashing every tensor of a 7B model is wasteful, so a deterministic sample
    is taken: the first and last few weight tensors plus an evenly spaced
    selection between them, which is where a bad merge or an unfrozen layer
    shows up.  ``sample=0`` hashes everything, for tests and small models.
    """

    fingerprints: Dict[str, str] = field(default_factory=dict)
    total_parameters: int = 0

    @staticmethod
    def _fingerprint(tensor: Any) -> str:
        import torch

        with torch.no_grad():
            flat = tensor.detach().to("cpu", copy=True).reshape(-1).float()
            if flat.numel() > 4096:  # sample large tensors deterministically
                stride = max(1, flat.numel() // 4096)
                flat = flat[::stride][:4096]
            payload = flat.numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def capture(cls, model: Any, sample: int = 24) -> "BaseWeightGuard":
        """Fingerprint the base weights of ``model`` as they are now."""
        names = [
            name
            for name, _ in model.named_parameters()
            if not is_adapter_parameter(name)
        ]
        if not names:
            raise AdapterError("model has no base parameters to guard")
        chosen = names if sample <= 0 or len(names) <= sample else _spread(names, sample)
        lookup = dict(model.named_parameters())
        fingerprints = {name: cls._fingerprint(lookup[name]) for name in chosen}
        total = sum(param.numel() for _, param in model.named_parameters())
        log.debug("guarding %d base tensors against modification", len(fingerprints))
        return cls(fingerprints=fingerprints, total_parameters=total)

    def changed(self, model: Any) -> List[str]:
        """Names of guarded tensors whose values differ from the capture."""
        lookup = dict(model.named_parameters())
        # Wrapping a model in PEFT renames every base parameter — a prefix is
        # added and an adapted Linear gains a `.base_layer` level — so guarded
        # names are matched on their normalised form as well as verbatim.
        normalised: Dict[str, Any] = {}
        for candidate, parameter in lookup.items():
            normalised.setdefault(_normalise_param_name(candidate), parameter)

        differences: List[str] = []
        for name, expected in self.fingerprints.items():
            parameter = lookup.get(name)
            if parameter is None:
                parameter = normalised.get(_normalise_param_name(name))
            if parameter is None:
                differences.append(f"{name} (missing)")
                continue
            if self._fingerprint(parameter) != expected:
                differences.append(name)
        return differences

    def verify(self, model: Any, context: str = "") -> None:
        """Raise :class:`BaseWeightsModified` if any guarded weight changed."""
        differences = self.changed(model)
        if differences:
            where = f" after {context}" if context else ""
            raise BaseWeightsModified(
                f"{len(differences)} base weight tensor(s) changed{where}: "
                + ", ".join(differences[:5])
                + (" ..." if len(differences) > 5 else "")
            )
        log.debug("base weights unchanged (%d tensors verified)", len(self.fingerprints))


def _spread(names: Sequence[str], count: int) -> List[str]:
    """A deterministic, evenly spread sample that keeps both ends."""
    if count >= len(names):
        return list(names)
    step = (len(names) - 1) / (count - 1) if count > 1 else 1
    picked = {int(round(index * step)) for index in range(count)}
    picked.update({0, len(names) - 1})
    return [names[index] for index in sorted(picked)]


def _normalise_param_name(name: str) -> str:
    """Strip PEFT's wrapper prefix and adapted-layer level from a name.

    ``base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight``
    and ``model.layers.0.self_attn.q_proj.weight`` are the same tensor before
    and after wrapping, and must compare equal.
    """
    for prefix in ("base_model.model.", "base_model."):
        while name.startswith(prefix):
            name = name[len(prefix) :]
    return name.replace(".base_layer.", ".").replace(".modules_to_save.default.", ".")


# --------------------------------------------------------------------------- #
# the four operations
# --------------------------------------------------------------------------- #
def attach_lora(
    model: Any,
    settings: Optional[LoRASettings] = None,
    *,
    adapter_name: str = "default",
    verify: bool = True,
) -> Any:
    """Attach a fresh LoRA adapter, leaving the base frozen.

    The base is frozen before PEFT sees the model, and afterwards every
    trainable parameter is checked to be an adapter parameter — a
    ``target_modules`` entry that matched nothing, or a config that would train
    the base, fails here rather than silently during training.
    """
    peft = _require_peft("attach a LoRA adapter")
    settings = settings or LoRASettings()

    frozen = freeze_parameters(model)
    log.info("froze %s base parameters before attaching LoRA", format_count(frozen))

    config = peft.LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(settings.target_modules),
        bias=settings.bias,
        task_type=settings.task_type,
    )
    try:
        wrapped = peft.get_peft_model(model, config, adapter_name=adapter_name)
    except Exception as exc:  # noqa: BLE001 - re-raised with guidance
        raise AdapterError(
            f"could not attach a LoRA adapter with target_modules="
            f"{list(settings.target_modules)}: {exc}\n"
            "Check the module names against the model architecture "
            "(print(model) lists them)."
        ) from exc

    counts = count_parameters(wrapped)
    if verify:
        _verify_only_adapter_is_trainable(wrapped, settings)
    log.info(
        "attached LoRA (r=%d, alpha=%d) | trainable %s of %s (%.4f%%)",
        settings.r,
        settings.alpha,
        format_count(counts.trainable),
        format_count(counts.total),
        counts.trainable_fraction * 100,
    )
    return wrapped


def _verify_only_adapter_is_trainable(model: Any, settings: LoRASettings) -> None:
    trainable = trainable_parameter_names(model)
    if not trainable:
        raise AdapterError(
            "no trainable parameters after attaching LoRA — target_modules "
            f"{list(settings.target_modules)} matched nothing in this model"
        )
    leaked = [name for name in trainable if not is_adapter_parameter(name)]
    if leaked:
        raise AdapterError(
            "base parameters are trainable after attaching LoRA, which would "
            "modify the base model: " + ", ".join(leaked[:5])
        )


def load_adapter(
    model: Any,
    path: str | Path,
    *,
    adapter_name: str = "default",
    is_trainable: bool = False,
) -> Any:
    """Attach a trained adapter from ``path`` to a base model.

    Works both on a plain base model (wraps it in a ``PeftModel``) and on a
    model that already carries adapters (registers another one under
    ``adapter_name``).  Loaded adapters are inference-only unless
    ``is_trainable`` is set.
    """
    # Validate the path before requiring peft: "that directory does not exist"
    # is more useful than "install peft" when the path is simply wrong.
    target = Path(path).expanduser()
    if not target.is_dir():
        raise AdapterError(
            f"adapter directory not found: {target}\n"
            "Point model.adapter_path at a directory containing "
            "adapter_config.json and adapter_model.safetensors."
        )
    if not (target / "adapter_config.json").is_file():
        raise AdapterError(
            f"{target} does not look like a PEFT adapter (no adapter_config.json). "
            "It may be a full checkpoint — load that as model.base_model instead."
        )
    peft = _require_peft("load a LoRA adapter")

    freeze_parameters(model)
    try:
        if hasattr(model, "load_adapter") and hasattr(model, "peft_config"):
            model.load_adapter(str(target), adapter_name=adapter_name, is_trainable=is_trainable)
            wrapped = model
        else:
            wrapped = peft.PeftModel.from_pretrained(
                model, str(target), adapter_name=adapter_name, is_trainable=is_trainable
            )
    except Exception as exc:  # noqa: BLE001 - re-raised with guidance
        raise AdapterError(f"could not load the adapter from {target}: {exc}") from exc

    counts = count_parameters(wrapped)
    log.info(
        "loaded adapter %r from %s | trainable %s of %s",
        adapter_name,
        target,
        format_count(counts.trainable),
        format_count(counts.total),
    )
    return wrapped


def unload_adapter(model: Any) -> Any:
    """Remove every adapter and return the untouched base model.

    Uses PEFT's ``unload`` (which detaches without merging), so the base
    weights that come back are the ones that went in.  A model with no adapter
    is returned unchanged.
    """
    if not hasattr(model, "peft_config"):
        log.debug("unload_adapter: model has no adapter; returning it unchanged")
        return model

    unload = getattr(model, "unload", None)
    if unload is None:  # pragma: no cover - very old peft
        raise AdapterError(
            "this peft version cannot unload adapters; upgrade peft to >= 0.6"
        )
    base = unload()
    log.info("unloaded adapter(s); base model restored (%s parameters)",
             format_count(count_parameters(base).total))
    return base


def save_adapter(model: Any, path: str | Path, *, adapter_name: str = "default") -> Path:
    """Save the adapter weights only — never the base checkpoint.

    ``model.save_pretrained`` on a PEFT model writes adapter tensors and their
    config; the base weights are not copied and not merged.
    """
    if not hasattr(model, "save_pretrained") or not hasattr(model, "peft_config"):
        raise AdapterError("save_adapter expects a PEFT-wrapped model")
    names = sorted(model.peft_config)
    selected = adapter_name if adapter_name in names else (names[0] if names else adapter_name)
    target = Path(path).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(str(target), selected_adapters=[selected])
    except TypeError:  # pragma: no cover - older peft without the argument
        model.save_pretrained(str(target))
    log.info("saved adapter %r to %s", selected, target)
    return target


def adapter_state(model: Any) -> Dict[str, Any]:
    """What adapters a model carries, and what is trainable."""
    counts: ParameterCounts = count_parameters(model)
    state: Dict[str, Any] = {
        "is_peft": hasattr(model, "peft_config"),
        "adapters": [],
        "active_adapter": None,
        "parameters": counts.to_dict(),
    }
    if state["is_peft"]:
        state["adapters"] = sorted(model.peft_config)
        state["active_adapter"] = getattr(model, "active_adapter", None)
        config = next(iter(model.peft_config.values()), None)
        if config is not None:
            state["lora"] = {
                "r": getattr(config, "r", None),
                "alpha": getattr(config, "lora_alpha", None),
                "dropout": getattr(config, "lora_dropout", None),
                "target_modules": sorted(getattr(config, "target_modules", []) or []),
            }
    return state
