"""Dataset acquisition, cleaning and splitting for Java program repair.

The pipeline turns a corpus of bug/fix pairs into JSONL training splits whose
rows are ``{"input": <IR4 prompt>, "output": <OR2 replacement>}``:

    loader.py            any corpus  -> BugFixPair (configurable field mapping)
    cleaner.py           drop non-single-function, oversized, test-only changes
    deduplicator.py      drop repeats before splitting
    tokenizer_filter.py  drop what exceeds the tokenizer budget
    dataset_builder.py   orchestrate, split, write JSONL, emit the report
    models.py            BugFixPair, TrainingRecord, DatasetReport

Typical use::

    from repairllama.data import DatasetBuilder, LoaderOptions

    builder = DatasetBuilder.from_config(cfg)
    result = builder.build_from_source("data/raw/megadiff.jsonl")
    result.write(cfg.paths.resolve("splits_dir"))
    print(result.report.render())

Nothing here downloads a corpus; loaders read local paths only.
"""

from repairllama.data.cleaner import (
    Cleaner,
    CleanerOptions,
    CleanResult,
    MethodSpan,
    changed_methods,
    clean_pairs,
    find_methods,
    looks_like_test,
    strip_comments,
)
from repairllama.data.dataset_builder import (
    BuildOptions,
    BuildResult,
    DatasetBuilder,
    SplitRatios,
    build_dataset,
    split_records,
)
from repairllama.data.deduplicator import (
    DedupResult,
    Deduplicator,
    deduplicate,
    fingerprint,
    normalize_code,
)
from repairllama.data.loader import (
    DirectoryLayout,
    FieldMap,
    LoaderOptions,
    load_pairs,
    load_records,
    record_to_pair,
    register_loader,
)
from repairllama.data.models import (
    BugFixPair,
    DataError,
    DatasetReport,
    Rejection,
    RejectionReason,
    StageReport,
    TokenStats,
    TrainingRecord,
)
from repairllama.data.tokenizer_filter import (
    HeuristicTokenCounter,
    HuggingFaceTokenCounter,
    TokenCounter,
    TokenizerFilter,
    TokenizerFilterOptions,
    filter_by_length,
    get_token_counter,
)

__all__ = [
    # models
    "BugFixPair",
    "TrainingRecord",
    "DataError",
    "Rejection",
    "RejectionReason",
    "StageReport",
    "TokenStats",
    "DatasetReport",
    # loader
    "FieldMap",
    "DirectoryLayout",
    "LoaderOptions",
    "load_pairs",
    "load_records",
    "record_to_pair",
    "register_loader",
    # cleaner
    "Cleaner",
    "CleanerOptions",
    "CleanResult",
    "clean_pairs",
    "MethodSpan",
    "find_methods",
    "changed_methods",
    "looks_like_test",
    "strip_comments",
    # deduplicator
    "Deduplicator",
    "DedupResult",
    "deduplicate",
    "fingerprint",
    "normalize_code",
    # tokenizer filter
    "TokenCounter",
    "HeuristicTokenCounter",
    "HuggingFaceTokenCounter",
    "TokenizerFilter",
    "TokenizerFilterOptions",
    "filter_by_length",
    "get_token_counter",
    # builder
    "DatasetBuilder",
    "BuildOptions",
    "BuildResult",
    "SplitRatios",
    "split_records",
    "build_dataset",
]
