"""The end-to-end dataset build: corpus in, JSONL splits and a report out.

The stages run in the order the RepairLLaMA paper describes:

    load          normalise a corpus into BugFixPair records
    clean         drop non-single-function, oversized and test-only changes
    deduplicate   drop repeats before splitting, so nothing leaks across it
    represent     render each pair as IR4 input / OR2 output
    length filter drop what does not fit the tokenizer budget
    split         deterministic train / validation / test partition
    write         one JSONL file per split, plus report.json and report.txt

Each stage records what it received and dropped, so the final
:class:`~repairllama.data.models.DatasetReport` accounts for every input
sample.

Splitting is deterministic in two senses: it depends only on the seed and the
content (records are sorted by ``bug_id`` before shuffling, so input order
does not matter), and with ``split_by_project`` all examples sharing a group
key land in the same split, which keeps near-identical files from the same
project out of both train and test.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from repairllama.data.cleaner import Cleaner, CleanerOptions
from repairllama.data.deduplicator import Deduplicator
from repairllama.data.loader import LoaderOptions, load_pairs
from repairllama.data.models import (
    BugFixPair,
    DatasetReport,
    DataError,
    Rejection,
    RejectionReason,
    StageReport,
    TrainingRecord,
)
from repairllama.data.tokenizer_filter import (
    TokenCounter,
    TokenizerFilter,
    TokenizerFilterOptions,
    get_token_counter,
)
from repairllama.representation import (
    RepresentationError,
    RepresentationOptions,
    build_training_example,
)
from repairllama.utils.io import write_jsonl
from repairllama.utils.logging import get_logger, log_section

__all__ = [
    "SplitRatios",
    "BuildOptions",
    "BuildResult",
    "DatasetBuilder",
    "split_records",
    "build_dataset",
]

log = get_logger("data.dataset_builder")

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitRatios:
    """Train/validation/test fractions; must sum to 1."""

    train: float = 0.9
    validation: float = 0.05
    test: float = 0.05

    def __post_init__(self) -> None:
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-6:
            raise DataError(f"split ratios must sum to 1.0, got {total}")
        if min(self.train, self.validation, self.test) < 0:
            raise DataError("split ratios must be non-negative")

    def as_dict(self) -> Dict[str, float]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


@dataclass(frozen=True)
class BuildOptions:
    """Everything the builder needs that is not a stage's own options."""

    dataset_name: str = "java-repair"
    seed: int = 42
    ratios: SplitRatios = field(default_factory=SplitRatios)
    split_by_project: bool = True
    deduplicate: bool = True
    dedup_strategy: str = "normalized"
    max_examples: int = 0  # 0 == no cap, applied to loaded pairs
    minimal_records: bool = False  # emit only {"input", "output"}
    skip_representation_errors: bool = True

    @classmethod
    def from_config(cls, cfg: Any) -> "BuildOptions":
        """Build from a :class:`repairllama.config.RepairConfig`."""
        return cls(
            dataset_name=cfg.data.dataset_name,
            seed=cfg.data.shuffle_seed,
            ratios=SplitRatios(
                train=cfg.data.train_split,
                validation=cfg.data.val_split,
                test=cfg.data.test_split,
            ),
            split_by_project=cfg.data.split_by_project,
            deduplicate=cfg.data.deduplicate,
            dedup_strategy=cfg.data.dedup_strategy,
            max_examples=cfg.data.max_examples,
            minimal_records=cfg.data.minimal_records,
        )


@dataclass
class BuildResult:
    """The finished dataset: records per split, the rejections, the report."""

    splits: Dict[str, List[TrainingRecord]] = field(default_factory=dict)
    rejections: List[Rejection] = field(default_factory=list)
    report: DatasetReport = field(default_factory=DatasetReport)

    @property
    def records(self) -> List[TrainingRecord]:
        return [record for name in SPLIT_NAMES for record in self.splits.get(name, [])]

    @property
    def total(self) -> int:
        return sum(len(records) for records in self.splits.values())

    def write(
        self,
        splits_dir: Union[str, Path],
        report_path: Optional[Union[str, Path]] = None,
        minimal: bool = False,
    ) -> Dict[str, Path]:
        """Write ``<split>.jsonl`` files, plus the report; return the paths."""
        target = Path(splits_dir).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        written: Dict[str, Path] = {}
        for name in SPLIT_NAMES:
            path = target / f"{name}.jsonl"
            count = write_jsonl(
                path,
                (
                    record.to_json_dict(minimal=minimal)
                    for record in self.splits.get(name, [])
                ),
            )
            written[name] = path
            log.info("wrote %d records to %s", count, path)

        if self.rejections:
            path = target / "rejected.jsonl"
            write_jsonl(path, (rejection.to_dict() for rejection in self.rejections))
            written["rejected"] = path

        report_target = Path(report_path) if report_path else target / "report.json"
        self.report.save(report_target)
        written["report"] = report_target
        log.info("wrote the dataset report to %s", report_target)
        return written


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def split_records(
    records: Sequence[TrainingRecord],
    ratios: Optional[SplitRatios] = None,
    seed: int = 42,
    group_by_project: bool = True,
) -> Dict[str, List[TrainingRecord]]:
    """Partition records into train/validation/test.

    Deterministic for a given seed and set of records, regardless of the order
    they arrive in: records (or groups) are sorted by key before shuffling.

    With ``group_by_project`` every record sharing a group key goes to the same
    split, which is what keeps near-identical files from one project out of
    both train and test.  Groups are packed largest-first into whichever split
    the group would leave *least* over its target, so a single big project
    cannot swamp a small split.  The cost is that ratios become approximate:
    they can only be met to within one group, and a corpus with very few
    groups may leave a small split empty (the builder warns when that
    happens).  Set ``group_by_project=False`` for exact record-level ratios.
    """
    ratios = ratios or SplitRatios()
    splits: Dict[str, List[TrainingRecord]] = {name: [] for name in SPLIT_NAMES}
    if not records:
        return splits

    rng = random.Random(seed)
    total = len(records)
    targets = {
        "train": ratios.train * total,
        "validation": ratios.validation * total,
        "test": ratios.test * total,
    }

    if not group_by_project:
        ordered = sorted(records, key=lambda record: record.bug_id)
        rng.shuffle(ordered)
        train_end = round(ratios.train * total)
        val_end = train_end + round(ratios.validation * total)
        assigned = {
            "train": ordered[:train_end],
            "validation": ordered[train_end:val_end],
            "test": ordered[val_end:],
        }
        return {name: [r.with_split(name) for r in items] for name, items in assigned.items()}

    groups: Dict[str, List[TrainingRecord]] = {}
    for record in records:
        groups.setdefault(record.group_key, []).append(record)

    ordered_keys = sorted(groups)
    rng.shuffle(ordered_keys)
    # Largest groups first so a single big project cannot overshoot a small
    # split; ties break on the shuffled order, keeping the result seed-driven.
    ordered_keys.sort(key=lambda key: -len(groups[key]))

    counts = {name: 0 for name in SPLIT_NAMES}
    eligible = [name for name in SPLIT_NAMES if targets[name] > 0] or ["train"]
    for key in ordered_keys:
        members = sorted(groups[key], key=lambda record: record.bug_id)
        # Whichever split this group leaves least over its target; ties break
        # on the split name, so the result depends only on the seed.
        name = min(
            eligible,
            key=lambda split: (
                (counts[split] + len(members)) / max(targets[split], 1e-9),
                split,
            ),
        )
        splits[name].extend(record.with_split(name) for record in members)
        counts[name] += len(members)

    empty = [name for name in eligible if not splits[name]]
    if empty:
        log.warning(
            "split(s) %s are empty: %d group(s) cannot be divided %s. Use more "
            "projects, or set split_by_project=false for record-level splits.",
            ", ".join(empty),
            len(ordered_keys),
            "/".join(f"{targets[name] / max(total, 1):.2f}" for name in SPLIT_NAMES),
        )

    return splits


# --------------------------------------------------------------------------- #
# the builder
# --------------------------------------------------------------------------- #
class DatasetBuilder:
    """Runs the stages and assembles the report."""

    def __init__(
        self,
        options: Optional[BuildOptions] = None,
        *,
        cleaner_options: Optional[CleanerOptions] = None,
        representation_options: Optional[RepresentationOptions] = None,
        token_counter: Optional[TokenCounter] = None,
        token_options: Optional[TokenizerFilterOptions] = None,
    ) -> None:
        self.options = options or BuildOptions()
        self.cleaner = Cleaner(cleaner_options)
        self.representation = representation_options or RepresentationOptions()
        self.token_filter = TokenizerFilter(token_counter, token_options)

    # -- config bridge ------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: Any, token_counter: Optional[TokenCounter] = None) -> "DatasetBuilder":
        """Build every stage's options from a :class:`RepairConfig`."""
        counter = token_counter or get_token_counter(
            cfg.model.tokenizer_name,
            use_model_tokenizer=cfg.data.use_model_tokenizer,
        )
        return cls(
            options=BuildOptions.from_config(cfg),
            cleaner_options=CleanerOptions(
                min_diff_lines=cfg.data.min_diff_lines,
                max_diff_lines=cfg.data.max_diff_lines,
                require_single_function=cfg.data.require_single_function,
                drop_test_only_changes=cfg.data.drop_test_only_changes,
            ),
            representation_options=RepresentationOptions.from_config(cfg.representation),
            token_counter=counter,
            token_options=TokenizerFilterOptions(
                max_input_tokens=cfg.representation.max_input_tokens,
                max_output_tokens=cfg.representation.max_output_tokens,
            ),
        )

    # -- stages ------------------------------------------------------------- #
    def represent(
        self, pairs: Sequence[BugFixPair]
    ) -> Tuple[List[TrainingRecord], List[Rejection]]:
        """Render each pair as an IR4 prompt and an OR2 target."""
        records: List[TrainingRecord] = []
        rejections: List[Rejection] = []
        for pair in pairs:
            try:
                resolved = pair if pair.has_region else pair.with_region()
                example = build_training_example(
                    resolved.buggy_code,
                    resolved.fixed_code,
                    resolved.suspicious_start,
                    resolved.suspicious_end,
                    options=self.representation,
                    file_path=resolved.file_path,
                    bug_id=resolved.bug_id,
                )
            except (RepresentationError, DataError) as exc:
                if not self.options.skip_representation_errors:
                    raise
                rejections.append(
                    Rejection(
                        bug_id=pair.bug_id,
                        reason=RejectionReason.REPRESENTATION_ERROR,
                        detail=str(exc),
                        stage="represent",
                    )
                )
                continue
            records.append(
                TrainingRecord(
                    input=example.prompt,
                    output=example.target,
                    bug_id=resolved.bug_id,
                    suspicious_start=resolved.suspicious_start,
                    suspicious_end=resolved.suspicious_end,
                    file_path=resolved.file_path,
                    project=resolved.project,
                    metadata=dict(resolved.metadata),
                )
            )
        return records, rejections

    # -- the whole pipeline ------------------------------------------------- #
    def build(self, pairs: Iterable[BugFixPair]) -> BuildResult:
        """Run every stage over ``pairs`` and return the finished dataset."""
        options = self.options
        loaded = list(pairs)
        if options.max_examples:
            loaded = loaded[: options.max_examples]

        raw_count = len(loaded)
        log_section(log, f"building dataset '{options.dataset_name}' from {raw_count} pairs")
        stages: List[StageReport] = []
        rejections: List[Rejection] = []

        cleaned = self.cleaner.run(loaded)
        rejections.extend(cleaned.rejections)
        stages.append(
            StageReport("clean", cleaned.received, len(cleaned.kept), cleaned.counts_by_reason())
        )

        if options.deduplicate:
            deduped = Deduplicator(options.dedup_strategy).run(cleaned.kept)
            rejections.extend(deduped.rejections)
            stages.append(
                StageReport(
                    "deduplicate",
                    deduped.received,
                    len(deduped.kept),
                    deduped.counts_by_reason(),
                )
            )
            surviving = deduped.kept
            duplicate_count = deduped.duplicates
        else:
            surviving = cleaned.kept
            duplicate_count = 0

        records, represent_rejections = self.represent(surviving)
        rejections.extend(represent_rejections)
        counts: Dict[str, int] = {}
        for rejection in represent_rejections:
            counts[rejection.reason.value] = counts.get(rejection.reason.value, 0) + 1
        stages.append(StageReport("represent", len(surviving), len(records), counts))

        filtered = self.token_filter.run(records)
        rejections.extend(filtered.rejections)
        stages.append(
            StageReport(
                "tokenizer_filter",
                filtered.received,
                len(filtered.kept),
                filtered.counts_by_reason(),
            )
        )

        splits = split_records(
            filtered.kept,
            options.ratios,
            seed=options.seed,
            group_by_project=options.split_by_project,
        )

        removed_by_reason: Dict[str, int] = {}
        for rejection in rejections:
            key = rejection.reason.value
            removed_by_reason[key] = removed_by_reason.get(key, 0) + 1

        report = DatasetReport(
            dataset_name=options.dataset_name,
            raw_samples=raw_count,
            remaining=len(filtered.kept),
            duplicates=duplicate_count,
            stages=stages,
            removed_by_reason=removed_by_reason,
            token_stats=filtered.stats,
            split_counts={name: len(splits.get(name, [])) for name in SPLIT_NAMES},
            settings={
                "seed": options.seed,
                "ratios": options.ratios.as_dict(),
                "split_by_project": options.split_by_project,
                "dedup_strategy": options.dedup_strategy if options.deduplicate else "off",
                "token_counter": getattr(self.token_filter.counter, "name", "custom"),
                "max_input_tokens": self.token_filter.options.max_input_tokens,
                "max_output_tokens": self.token_filter.options.max_output_tokens,
                "input_representation": "IR4",
                "output_representation": "OR2",
            },
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

        log.info(
            "built %d records from %d raw samples (%d removed, %d duplicates)",
            report.remaining,
            report.raw_samples,
            report.removed,
            report.duplicates,
        )
        return BuildResult(splits=splits, rejections=rejections, report=report)

    def build_from_source(
        self,
        source: Union[str, Path, Iterable[Mapping[str, Any]]],
        loader_options: Optional[LoaderOptions] = None,
    ) -> BuildResult:
        """Load a corpus from ``source`` and build the dataset from it."""
        loader_options = loader_options or LoaderOptions()
        if self.options.max_examples and not loader_options.limit:
            loader_options = LoaderOptions(
                format=loader_options.format,
                field_map=loader_options.field_map,
                layout=loader_options.layout,
                encoding=loader_options.encoding,
                limit=self.options.max_examples,
                skip_invalid=loader_options.skip_invalid,
                infer_region=loader_options.infer_region,
            )
        return self.build(load_pairs(source, loader_options))


def build_dataset(
    source: Union[str, Path, Iterable[Mapping[str, Any]], Iterable[BugFixPair]],
    options: Optional[BuildOptions] = None,
    loader_options: Optional[LoaderOptions] = None,
    **builder_kwargs: Any,
) -> BuildResult:
    """One-call dataset build from a path, raw records, or ready-made pairs."""
    builder = DatasetBuilder(options, **builder_kwargs)
    items = list(source) if not isinstance(source, (str, Path)) else source
    if not isinstance(items, (str, Path)) and items and isinstance(items[0], BugFixPair):
        return builder.build(items)  # type: ignore[arg-type]
    return builder.build_from_source(items, loader_options)  # type: ignore[arg-type]
