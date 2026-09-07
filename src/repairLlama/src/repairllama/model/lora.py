"""The Java repair LoRA adapter — the paper's configuration, made checkable.

RepairLLaMA reports a small, specific LoRA setup, and reproducing the paper
means using exactly it::

    r = 8
    alpha = 16                  # alpha / r = 2, the usual scaling
    dropout = 0.05
    target_modules = ["q_proj", "v_proj"]

Adapting only the query and value projections is the original LoRA recipe:
it touches roughly a tenth of the parameters that adapting all four attention
projections would, and the paper found it sufficient for program repair.  The
values live in :data:`JAVA_REPAIR` and in ``model.lora`` in the config, which
default to the same thing.

What this module guarantees
---------------------------
CodeLlama-7B stays frozen.  Only the adapter learns.  Those are easy claims to
make and easy to get wrong, so :func:`assert_java_repair_lora` checks all four
properties and refuses to continue if any fails:

1. every base parameter has ``requires_grad=False``;
2. the LoRA parameters are trainable (an adapter that matched nothing trains
   nothing, silently, for a whole run);
3. the adapted modules are exactly ``q_proj`` and ``v_proj``;
4. trainable parameters are a tiny fraction of the total — on CodeLlama-7B
   this configuration trains about 4.2M of 6.7B parameters (~0.06%).

The adapter is **never merged into the base model**.  Merging writes the
adapter into the base weights, which destroys the one property that makes this
approach cheap: one frozen base serving many adapters.  :func:`save_adapter`
writes adapter tensors only, and verifies the base weights are untouched
before it does.

One base, many languages
------------------------
Adapters live one directory per language under ``paths.adapters_dir``::

    adapters/
        java-repair/        <- implemented here
        python-repair/      <- planned
        cpp-repair/         <- planned

:data:`ADAPTER_PROFILES` is that registry.  Only Java is implemented; asking
for another language raises with a message naming what exists, rather than
silently producing a Java adapter under a Python name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from repairllama.model.adapter import (
    AdapterError,
    BaseWeightGuard,
    LoRASettings,
    attach_lora,
    is_adapter_parameter,
    load_adapter,
    trainable_parameter_names,
)
from repairllama.model.adapter import save_adapter as _save_adapter_files
from repairllama.model.device import format_count
from repairllama.model.loader import ParameterCounts, count_parameters
from repairllama.utils.logging import get_logger

__all__ = [
    "LoRAAssertionError",
    "AdapterProfile",
    "JAVA_REPAIR",
    "ADAPTER_PROFILES",
    "PLANNED_LANGUAGES",
    "ADAPTER_METADATA_FILE",
    "available_languages",
    "profile_for",
    "adapter_directory",
    "create_lora_config",
    "attach_java_repair_adapter",
    "print_trainable_parameters",
    "save_adapter",
    "load_java_repair_adapter",
    "assert_java_repair_lora",
    "check_java_repair_lora",
    "TrainableSummary",
    "LoRACheck",
]

log = get_logger("model.lora")

ADAPTER_METADATA_FILE = "repairllama_adapter.json"

# Trainable parameters must be at most this fraction of the total.  LoRA on
# q_proj/v_proj at r=8 gives ~0.06% on a 7B model; 5% is a loose bound whose
# only job is to catch a configuration that is quietly training the base.
MAX_TRAINABLE_FRACTION = 0.05


class LoRAAssertionError(AdapterError):
    """Raised when the LoRA setup does not match what training requires."""


# --------------------------------------------------------------------------- #
# the per-language registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AdapterProfile:
    """One language's adapter: where it lives and how it is configured."""

    language: str
    directory: str  # relative to paths.adapters_dir
    settings: LoRASettings
    description: str = ""

    @property
    def adapter_name(self) -> str:
        """The PEFT adapter name, so several can be loaded side by side."""
        return self.directory

    def path(self, adapters_dir: str | Path) -> Path:
        return Path(adapters_dir).expanduser() / self.directory

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "directory": self.directory,
            "description": self.description,
            "lora": self.settings.to_dict(),
        }


#: The RepairLLaMA paper's reported LoRA configuration for Java repair.
JAVA_REPAIR = AdapterProfile(
    language="java",
    directory="java-repair",
    settings=LoRASettings(
        r=8,
        alpha=16,
        dropout=0.05,
        target_modules=("q_proj", "v_proj"),
        bias="none",
        task_type="CAUSAL_LM",
    ),
    description="RepairLLaMA Java program repair (paper configuration)",
)

#: Implemented languages, keyed by language name.
ADAPTER_PROFILES: Dict[str, AdapterProfile] = {"java": JAVA_REPAIR}

#: Languages the layout reserves but which are not implemented yet.
PLANNED_LANGUAGES: Dict[str, str] = {"python": "python-repair", "cpp": "cpp-repair"}


def available_languages() -> List[str]:
    """Languages with an implemented adapter profile."""
    return sorted(ADAPTER_PROFILES)


def profile_for(language: str = "java") -> AdapterProfile:
    """The profile for ``language``, or a clear error naming what exists."""
    key = language.lower()
    if key in ADAPTER_PROFILES:
        return ADAPTER_PROFILES[key]
    if key in PLANNED_LANGUAGES:
        raise AdapterError(
            f"no adapter is implemented for {language!r} yet — "
            f"{PLANNED_LANGUAGES[key]}/ is reserved in the layout but empty. "
            f"Implemented: {available_languages()}."
        )
    raise AdapterError(
        f"unknown language {language!r}; implemented: {available_languages()}, "
        f"planned: {sorted(PLANNED_LANGUAGES)}"
    )


def adapter_directory(
    adapters_dir: str | Path, language: str = "java", create: bool = False
) -> Path:
    """``<adapters_dir>/<language>-repair`` — where that adapter is stored."""
    path = profile_for(language).path(adapters_dir)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# 1. the config
# --------------------------------------------------------------------------- #
def create_lora_config(
    settings: Optional[LoRASettings] = None,
    *,
    language: str = "java",
    inference_mode: bool = False,
    **overrides: Any,
):
    """Build the PEFT ``LoraConfig`` for a language's adapter.

    Defaults to the paper's Java configuration (r=8, alpha=16, dropout=0.05,
    q_proj/v_proj).  ``overrides`` change individual fields — useful for an
    ablation — and are validated by :class:`LoRASettings` before PEFT sees
    them.

    Returns:
        ``peft.LoraConfig``.
    """
    from repairllama.model.adapter import _require_peft

    peft = _require_peft("create a LoRA config")
    settings = settings or profile_for(language).settings
    if overrides:
        if "target_modules" in overrides:
            overrides["target_modules"] = tuple(overrides["target_modules"])
        settings = replace(settings, **overrides)

    return peft.LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(settings.target_modules),
        bias=settings.bias,
        task_type=settings.task_type,
        inference_mode=inference_mode,
    )


# --------------------------------------------------------------------------- #
# 2. attaching
# --------------------------------------------------------------------------- #
def attach_java_repair_adapter(
    model: Any,
    settings: Optional[LoRASettings] = None,
    *,
    adapter_name: Optional[str] = None,
    verify: bool = True,
    guard: Optional[BaseWeightGuard] = None,
) -> Any:
    """Attach the Java repair LoRA adapter to a frozen base model.

    The base is frozen before PEFT wraps the model, and afterwards the four
    properties in :func:`assert_java_repair_lora` are checked — so a
    misconfigured adapter fails here rather than after an hour of training
    that changed nothing.

    Args:
        model: the base causal LM (or one already carrying adapters).
        settings: defaults to the paper's configuration.
        adapter_name: PEFT adapter name; defaults to ``java-repair``.
        guard: an optional :class:`BaseWeightGuard` captured before the call;
            when given, the base weights are verified unchanged afterwards.

    Returns:
        The PEFT-wrapped model.  The base weights are **not** merged.
    """
    profile = JAVA_REPAIR
    settings = settings or profile.settings
    name = adapter_name or profile.adapter_name

    log.info(
        "attaching the %s adapter: r=%d alpha=%d dropout=%s targets=%s",
        name,
        settings.r,
        settings.alpha,
        settings.dropout,
        list(settings.target_modules),
    )
    wrapped = attach_lora(model, settings, adapter_name=name, verify=verify)

    if verify:
        assert_java_repair_lora(wrapped, settings)
    if guard is not None:
        guard.verify(wrapped, context="attaching the Java repair adapter")
    print_trainable_parameters(wrapped)
    return wrapped


# --------------------------------------------------------------------------- #
# 3. reporting
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainableSummary:
    """Parameter accounting for an adapted model."""

    total: int
    trainable: int
    frozen: int
    percentage: float
    adapter_parameters: int
    adapted_modules: Tuple[str, ...] = ()

    @property
    def ratio(self) -> float:
        """How many times smaller the trainable set is than the whole model."""
        return (self.total / self.trainable) if self.trainable else float("inf")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "trainable": self.trainable,
            "frozen": self.frozen,
            "percentage": self.percentage,
            "adapter_parameters": self.adapter_parameters,
            "adapted_modules": list(self.adapted_modules),
        }

    def render(self) -> str:
        return (
            f"trainable params: {self.trainable:,} || "
            f"all params: {self.total:,} || "
            f"trainable%: {self.percentage:.4f} "
            f"({self.ratio:,.0f}x smaller)"
        )


def print_trainable_parameters(
    model: Any, printer: Optional[Callable[[str], None]] = None
) -> TrainableSummary:
    """Print (and return) what is trainable after attaching an adapter.

    Mirrors PEFT's own ``print_trainable_parameters`` line so the numbers are
    comparable with published ones, and adds which modules were adapted.
    By default it logs; pass ``printer=print`` to write to stdout instead.
    """
    counts: ParameterCounts = count_parameters(model)
    adapter_total = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if is_adapter_parameter(name)
    )
    summary = TrainableSummary(
        total=counts.total,
        trainable=counts.trainable,
        frozen=counts.frozen,
        percentage=counts.trainable_fraction * 100,
        adapter_parameters=adapter_total,
        adapted_modules=adapted_modules(model),
    )

    emit = printer or (lambda line: log.info("%s", line))
    emit(summary.render())
    emit(
        f"  base (frozen): {format_count(summary.frozen)} | "
        f"adapter: {format_count(summary.adapter_parameters)} | "
        f"adapted modules: {', '.join(summary.adapted_modules) or 'none'}"
    )
    return summary


def adapted_modules(model: Any) -> Tuple[str, ...]:
    """The module names an attached adapter targets, from its own config."""
    configs = getattr(model, "peft_config", None)
    if not configs:
        return ()
    modules: set = set()
    for config in configs.values():
        targets = getattr(config, "target_modules", None) or ()
        modules.update(targets if not isinstance(targets, str) else {targets})
    return tuple(sorted(modules))


# --------------------------------------------------------------------------- #
# 4. the four assertions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LoRACheck:
    """One verified property of the LoRA setup."""

    name: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


def check_java_repair_lora(
    model: Any,
    settings: Optional[LoRASettings] = None,
    *,
    max_trainable_fraction: float = MAX_TRAINABLE_FRACTION,
) -> List[LoRACheck]:
    """Run the four checks and return them, without raising."""
    settings = settings or JAVA_REPAIR.settings
    counts = count_parameters(model)
    trainable = trainable_parameter_names(model)
    base_trainable = [name for name in trainable if not is_adapter_parameter(name)]
    adapter_trainable = [name for name in trainable if is_adapter_parameter(name)]
    expected = tuple(sorted(settings.target_modules))
    actual = adapted_modules(model)

    fraction = counts.trainable_fraction
    return [
        LoRACheck(
            "base parameters are frozen",
            not base_trainable,
            "all base parameters have requires_grad=False"
            if not base_trainable
            else f"{len(base_trainable)} trainable base tensor(s): "
            + ", ".join(base_trainable[:3]),
        ),
        LoRACheck(
            "LoRA parameters are trainable",
            bool(adapter_trainable),
            f"{len(adapter_trainable)} adapter tensor(s), "
            f"{format_count(counts.trainable)} parameters"
            if adapter_trainable
            else "no adapter parameter is trainable — the adapter would learn nothing",
        ),
        LoRACheck(
            f"target modules are {list(expected)}",
            tuple(sorted(actual)) == expected,
            f"adapted {list(actual)}" if actual else "no adapter attached",
        ),
        LoRACheck(
            "trainable parameters are a small fraction of the total",
            0 < fraction <= max_trainable_fraction,
            f"{counts.trainable:,} of {counts.total:,} "
            f"({fraction:.4%}, {counts.total / max(counts.trainable, 1):,.0f}x smaller)",
        ),
    ]


def assert_java_repair_lora(
    model: Any,
    settings: Optional[LoRASettings] = None,
    *,
    max_trainable_fraction: float = MAX_TRAINABLE_FRACTION,
) -> List[LoRACheck]:
    """Assert the four properties training depends on; raise if any fails."""
    checks = check_java_repair_lora(
        model, settings, max_trainable_fraction=max_trainable_fraction
    )
    failures = [check for check in checks if not check.ok]
    if failures:
        raise LoRAAssertionError(
            "the LoRA setup is not ready for training:\n"
            + "\n".join(f"  - {check.name}: {check.detail}" for check in failures)
        )
    for check in checks:
        log.debug("LoRA check passed: %s (%s)", check.name, check.detail)
    log.info("LoRA verified: %d/%d checks passed", len(checks), len(checks))
    return checks


# --------------------------------------------------------------------------- #
# 5. saving and loading — adapter only, never merged
# --------------------------------------------------------------------------- #
def save_adapter(
    model: Any,
    adapters_dir: str | Path,
    *,
    language: str = "java",
    adapter_name: Optional[str] = None,
    base_model: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    guard: Optional[BaseWeightGuard] = None,
) -> Path:
    """Save the adapter under ``<adapters_dir>/<language>-repair``.

    Only adapter tensors are written — the base checkpoint is never copied or
    merged into.  A ``repairllama_adapter.json`` sidecar records the language,
    base model and LoRA settings, so an adapter directory says what it is and
    what it was trained against.

    Returns the directory written.
    """
    if not hasattr(model, "peft_config"):
        raise AdapterError(
            "save_adapter expects a PEFT-wrapped model; attach an adapter first "
            "with attach_java_repair_adapter()"
        )
    profile = profile_for(language)
    name = adapter_name or profile.adapter_name
    target = adapter_directory(adapters_dir, language, create=True)

    if guard is not None:
        # Saving must never be preceded by a merge; prove it before writing.
        guard.verify(model, context="saving the adapter")

    names = sorted(model.peft_config)
    selected = name if name in names else names[0]
    _save_adapter_files(model, target, adapter_name=selected)

    written = _flatten_saved_adapter(target, selected)
    config = model.peft_config.get(selected)
    payload = {
        "language": profile.language,
        "adapter_name": selected,
        "directory": profile.directory,
        "base_model": base_model
        or getattr(config, "base_model_name_or_path", None),
        "lora": {
            "r": getattr(config, "r", None),
            "alpha": getattr(config, "lora_alpha", None),
            "dropout": getattr(config, "lora_dropout", None),
            "target_modules": sorted(getattr(config, "target_modules", []) or []),
        },
        "merged_into_base": False,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trainable_parameters": count_parameters(model).trainable,
    }
    if metadata:
        payload["metadata"] = metadata
    (written / ADAPTER_METADATA_FILE).write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )

    log.info(
        "saved the %s adapter to %s (%s trainable parameters, base not merged)",
        selected,
        written,
        format_count(payload["trainable_parameters"]),
    )
    return written


def _resolve_adapter_files(target: Path, adapter_name: str) -> Path:
    """Read-only: the directory holding ``adapter_config.json`` under ``target``.

    Tolerates an adapter saved by an older version that nested the files under
    ``<target>/<adapter_name>``; unlike the save path, it changes nothing.
    """
    nested = target / adapter_name
    if (nested / "adapter_config.json").is_file():
        return nested
    return target


def _flatten_saved_adapter(target: Path, adapter_name: str) -> Path:
    """Where a saved adapter's files actually are, under ``target``.

    ``save_pretrained`` with a named adapter nests the files one level deeper,
    which would give ``adapters/java-repair/java-repair/``.  The contract is
    that an adapter lives directly in ``adapters/<language>-repair/``, so a
    nested save is flattened back up.
    """
    nested = target / adapter_name
    if not (nested / "adapter_config.json").is_file():
        return target
    for entry in nested.iterdir():
        destination = target / entry.name
        if destination.exists():
            destination.unlink()
        entry.replace(destination)
    nested.rmdir()
    log.debug("flattened the nested adapter directory %s into %s", nested, target)
    return target


def load_java_repair_adapter(
    model: Any,
    adapters_dir: Optional[str | Path] = None,
    *,
    path: Optional[str | Path] = None,
    adapter_name: Optional[str] = None,
    is_trainable: bool = False,
    verify: bool = True,
) -> Any:
    """Load the trained Java adapter onto a frozen base model.

    Give either ``adapters_dir`` (the adapter is read from
    ``<adapters_dir>/java-repair``) or an explicit ``path``.  The adapter is
    attached, not merged, so the same base model can carry another language's
    adapter alongside it later.
    """
    if path is None and adapters_dir is None:
        raise AdapterError("give either adapters_dir or an explicit path")
    profile = JAVA_REPAIR
    name = adapter_name or profile.adapter_name
    target = Path(path).expanduser() if path else profile.path(adapters_dir)  # type: ignore[arg-type]
    target = _resolve_adapter_files(target, name)

    wrapped = load_adapter(model, target, adapter_name=name, is_trainable=is_trainable)

    sidecar = target / ADAPTER_METADATA_FILE
    if sidecar.is_file():
        try:
            info = json.loads(sidecar.read_text(encoding="utf-8"))
            log.info(
                "adapter metadata: language=%s base_model=%s saved_at=%s",
                info.get("language"),
                info.get("base_model"),
                info.get("saved_at"),
            )
        except json.JSONDecodeError:  # pragma: no cover - corrupt sidecar
            log.warning("could not read %s", sidecar)

    if verify:
        modules = adapted_modules(wrapped)
        expected = tuple(sorted(profile.settings.target_modules))
        if tuple(sorted(modules)) != expected:
            log.warning(
                "loaded adapter targets %s, not the paper's %s",
                list(modules),
                list(expected),
            )
    print_trainable_parameters(wrapped)
    return wrapped
