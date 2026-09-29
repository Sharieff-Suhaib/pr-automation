# repairllama-java

Fine-tuned LLM program repair for Java, in the style of
[RepairLLaMA](https://arxiv.org/abs/2312.15698): a code LLM adapted with LoRA
on bug/fix pairs, prompted with a *code representation* that marks exactly
where the bug is, and evaluated by actually compiling and testing the patches
it produces.

**Status: trainable end to end; inference and evaluation still to come.** The
package layout, configuration schema, logging utilities, CLI, the IR4 × OR2 code
representation, the dataset pipeline, the model layer and LoRA fine-tuning are
implemented: `prepare-data` produces JSONL splits, `model-check` dry-runs the
model stack, and `train` fine-tunes the adapter and writes it to
`adapters/java-repair/`. Nothing is downloaded automatically. The remaining
stages (`localize`, `infer`, `patch`, `evaluate`) validate their
configuration, log the plan they would run, and exit with code `2`
(`not implemented`).

---

## Layout

```
repairllama-java/
├── models/                  base checkpoints (git-ignored)
├── adapters/                trained LoRA adapters (git-ignored)
│   └── java-repair/         one directory per language
├── data/
│   ├── raw/                 corpora as downloaded
│   ├── processed/           cleaned bug/fix pairs (JSONL)
│   └── splits/              train / val / test
├── src/repairllama/
│   ├── config.py            the configuration schema (dataclasses + YAML)
│   ├── cli.py               argparse entry points
│   ├── data/                corpus -> cleaned, deduped, split JSONL
│   │   ├── models.py        BugFixPair, TrainingRecord, DatasetReport
│   │   ├── loader.py        any corpus layout -> BugFixPair
│   │   ├── cleaner.py       single-function / size / test-only filtering
│   │   ├── deduplicator.py  exact, normalized and input-only dedup
│   │   ├── tokenizer_filter.py  tokenizer-aware length filtering
│   │   └── dataset_builder.py   orchestration, splitting, report
│   ├── representation/      IR4 prompt encoding / OR2 target decoding
│   │   ├── ir4.py           input: marked, commented-out region + <FILL_ME>
│   │   ├── or2.py           output: replacement code for that region only
│   │   └── builder.py       build_training_example / build_inference_input
│   ├── localization/        which region of the file to repair
│   ├── model/               checkpoint, tokenizer and adapter loading
│   │   ├── device.py        device + dtype resolution, memory reporting
│   │   ├── tokenizer.py     tokenizer, pad token, representation markers
│   │   ├── loader.py        find a checkpoint locally, load the base LM
│   │   ├── adapter.py       attach / load / save / unload LoRA (generic)
│   │   └── lora.py          the Java repair adapter (the paper's config)
│   ├── training/            supervised fine-tuning
│   │   ├── collator.py      IR4/OR2 rows -> masked batches
│   │   ├── metrics.py       validation loss, perplexity, accuracies
│   │   ├── checkpoint.py    resuming, and the final adapter artifact
│   │   └── trainer.py       the loop
│   ├── inference/           candidate patch generation
│   ├── patching/            splice a hunk back into source
│   ├── evaluation/          compile / test / score
│   └── utils/               logging, paths, seeding, IO
├── tests/                   mirrors the stage packages
├── configs/java_repair.yaml the default configuration
├── scripts/                 setup and smoke-check helpers
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

## Why the pipeline is shaped this way

Repair quality depends less on raw model size than on two things the pipeline
controls explicitly, so each gets its own stage:

**Fault localization** decides what goes in the prompt. `perfect` localization
(derived from the reference diff) is what you use while training and for upper
bound measurements; `stacktrace` and `spectrum` are the realistic settings.
Swapping strategies must not require touching the prompt code, so localization
emits a `BugRegion` and nothing more.

**Representation** decides how the buggy region is written down. The
RepairLLaMA study found the input/output encoding pair moves results more than
parameter count: whether the model sees the whole function or a marked hunk,
and whether it emits a whole function or just the replacement. Both are config
keys (`representation.input_format` / `output_format`), so a representation
ablation is a config sweep, not a code change.

### The implemented pair: IR4 × OR2

**IR4** (input) keeps the buggy code visible rather than deleting it. The
suspicious region is bracketed by marker comments, each buggy line is commented
out verbatim, and a `<FILL_ME>` token — indented to the region — marks where the
replacement goes:

```java
// suspicious region: lines 5-7 of 15
public class Grade {
    public String classify(int score) {
        if (score >= 90) {
            return "A";
        // buggy lines start here
        // } else if (score >= 80) {
            // return "C";
        // } else {
        // buggy lines end here
        <FILL_ME>
            if (score >= 70) {
```

**OR2** (output) is only the replacement for that region — never the enclosing
method:

```java
        } else if (score >= 80) {
            return "B";
        } else {
```

Two API calls cover both directions:

```python
from repairllama.representation import build_training_example, build_inference_input

example = build_training_example(buggy_src, fixed_src, 5, 7)  # prompt + target
request = build_inference_input(buggy_src, 5, 7)              # prompt only
```

Both return structured objects (`.prompt`, `.target`, `.original_region`,
`.region`, `.to_dict()`). For training, the fixed region is derived from the
fixed source by line diff, so callers supply only the buggy boundaries;
`fixed_start`/`fixed_end` override that when they are already known.

Three guarantees the tests pin down:

- **Line numbers are 1-based and inclusive.** `(4, 4)` is exactly line 4. Bad
  bounds — inverted, below 1, past the end, non-integer — raise
  `RepresentationError`; they are never clamped.
- **The Java code is never silently modified.** Context lines are emitted
  byte-for-byte, commenting is exactly invertible
  (`recover_region_from_prompt` reconstructs the original region from a
  prompt), and context dropped by the window is announced by an explicit
  `// ... N line(s) omitted ...` comment.
- **The target reconstructs the fix.** Splicing the target back over lines
  `start..end` reproduces the fixed source, for every fixture — including
  multi-line regions, deletions, tab-indented code, and regions touching the
  first or last line.

Edits that reach outside the marked region are reported
(`example.changed_outside_region`) rather than absorbed; `strict=True` rejects
those examples outright, since they usually indicate mislocalization.

The remaining stages are deliberately boring: generate *N* candidates, splice
each one back into the real file, then compile and test it. A patch only
counts once it builds and the test suite agrees.

## The dataset pipeline

`repairllama prepare-data` turns a corpus of bug/fix pairs into JSONL splits
whose rows are `{"input": <IR4 prompt>, "output": <OR2 replacement>}`:

```bash
repairllama prepare-data --input data/raw/megadiff.jsonl --output data/splits
repairllama prepare-data --input corpus/ --format directory --dry-run
```

Stages run in the order the paper describes, and each one reports what it
received and dropped:

| Stage | Drops |
| --- | --- |
| `clean` | identical pairs, blank sources, fixes spanning several methods or none, diffs outside `min/max_diff_lines`, test-only changes |
| `deduplicate` | repeats — before splitting, so nothing leaks across it |
| `represent` | pairs whose region cannot be rendered as IR4 × OR2 |
| `tokenizer_filter` | examples over `max_input_tokens` / `max_output_tokens` |

### Expected input format

Everything is normalised into one intermediate record before any processing:

```json
{
  "bug_id": "megadiff-1a2b3c",
  "buggy_code": "public int max(int a, int b) { ... }",
  "fixed_code": "public int max(int a, int b) { ... }",
  "suspicious_start": 4,
  "suspicious_end": 4,
  "file_path": "src/main/java/Foo.java",
  "project": "commons-lang",
  "metadata": {"commit": "..."}
}
```

Only `bug_id`, `buggy_code` and `fixed_code` are required. `buggy_code` and
`fixed_code` are the *same unit* before and after the fix — normally one Java
method, optionally with its enclosing class. When the suspicious region is
absent it is derived from the line diff (perfect localization); when `project`
is absent the file path, then the bug id, is used for split grouping.

### No assumed corpus layout

Megadiff has no single on-disk layout, so the loader is configured rather than
hard-coded. Three formats are built in — `jsonl`, `json` and `directory` — and
[`FieldMap`](src/repairllama/data/loader.py) maps a corpus's own key names onto
the canonical ones, including dotted paths into nested objects:

```python
from repairllama.data import FieldMap, LoaderOptions, load_pairs

pairs = load_pairs("corpus.jsonl", LoaderOptions(
    field_map=FieldMap(bug_id="commit.sha", buggy_code="before", fixed_code="after"),
))
```

Common aliases (`id`/`before`/`after`, `path`, `repo`, …) are accepted out of
the box, and unmapped keys are preserved as `metadata`. For a tree of per-bug
directories, `DirectoryLayout` configures the file names (fixed or glob-style),
recursion, where the bug id comes from, a regex for the project, and an
optional per-bug `metadata.json`. Anything else registers its own reader with
`register_loader(name, reader)` — no patching required. Loaders read local
paths only; nothing downloads a corpus.

### Token lengths without downloading a model

Length filtering is tokenizer-aware but does not require a checkpoint: the
default `HeuristicTokenCounter` approximates a code tokenizer (identifiers
split into subword-sized pieces, newlines and indentation counted) with no
dependencies, so `prepare-data` runs offline and deterministically. Set
`data.use_model_tokenizer: true` to count with the real tokenizer for
`model.base_model` instead; if it cannot be loaded, the build warns and falls
back rather than failing. Any object with `count(text) -> int` can be plugged
in.

### Determinism and leakage

Splitting depends only on the seed and the content — records are sorted by
`bug_id` before shuffling, so input order does not change the result. With
`data.split_by_project` (the default) every example sharing a project lands in
the same split, which keeps near-identical files out of both train and test.
The cost is that ratios become approximate: they can only be met to within one
project, and a corpus with very few projects may leave a small split empty (the
build warns when it does). Set `split_by_project: false` for exact record-level
ratios.

### The report (dataset)

Every build writes `report.json` and a readable `report.txt` beside the splits,
covering raw sample count, how many were removed and why, how many were
duplicates, how many remain, the token-length distribution (min/median/mean/
p90/p95/p99 plus a histogram) and the per-split counts. Dropped samples are
written to `rejected.jsonl` with the stage and reason, so any filtering
decision can be audited.

### Dependency discipline

`repairllama`, `repairllama.config` and `repairllama.cli` import nothing
heavier than PyYAML — `torch` and `transformers` are imported inside the
functions that need them. Config validation, path resolution and `--help` work
on a laptop with no ML stack installed, and there is a test that keeps it that
way.

## The model layer

CodeLLaMA-7B (or any causal LM) plus a PEFT LoRA adapter, loaded with two
rules the code enforces rather than assumes.

**Nothing is downloaded.** A 13 GB checkpoint appearing because a config file
names a repo id is a bad surprise, so `model.base_model` is resolved from local
sources only — a directory path, `<models_dir>/<name>`, or the HuggingFace
cache. If none of those has it, the error names every place it looked and gives
three ways forward (point the config at a checkpoint, download it once with
`huggingface-cli download`, or set `model.allow_download: true`).

**The base model is never modified.** Parameters are frozen the moment the
checkpoint is in memory, `attach_lora` freezes again before PEFT sees the model
and then verifies that *every* trainable tensor is an adapter tensor (a
`target_modules` entry matching nothing fails loudly instead of training
nothing), and `BaseWeightGuard` fingerprints base weights so a change can be
proven after the fact:

```python
from repairllama.model import BaseWeightGuard, LoRASettings, attach_lora, load_base_model

loaded = load_base_model(LoadOptions.from_config(cfg))   # frozen, on device
guard  = BaseWeightGuard.capture(loaded.model)
model  = attach_lora(loaded.model, LoRASettings.from_config(cfg.model.lora))
guard.verify(model)          # raises BaseWeightsModified if a base weight moved
base   = unload_adapter(model)   # the original model back, unmerged
```

Device and dtype are resolved explicitly. `auto` prefers CUDA, then Apple MPS,
then CPU, and picks bfloat16 on CUDA (float16 where bf16 is unsupported, and on
MPS), float32 on CPU. A *named* device that is not available raises rather than
silently downgrading — a training run should not quietly end up on the CPU —
while an explicit dtype is always honoured, with a note when the combination is
a poor one (`float16` on CPU, `bfloat16` on Metal).

### The Java repair adapter

[`model/lora.py`](src/repairllama/model/lora.py) pins the paper's reported LoRA
configuration, which is also the default in `configs/java_repair.yaml`:

| | |
| --- | --- |
| `r` | 8 |
| `alpha` | 16 |
| `dropout` | 0.05 |
| `target_modules` | `q_proj`, `v_proj` |

Adapting only the query and value projections is the original LoRA recipe —
about a tenth of the parameters that adapting all four attention projections
would touch. On CodeLlama-7B it trains roughly 4.2M of 6.7B parameters (~0.06%).

```python
from repairllama.model import (
    BaseWeightGuard, attach_java_repair_adapter, load_base_model,
    load_java_repair_adapter, save_adapter,
)

loaded = load_base_model(LoadOptions.from_config(cfg))   # frozen base
guard  = BaseWeightGuard.capture(loaded.model)
model  = attach_java_repair_adapter(loaded.model, guard=guard)
# ... training goes here ...
save_adapter(model, cfg.paths.resolve("adapters_dir"))   # adapters/java-repair/
```

`attach_java_repair_adapter` asserts four properties before returning, so a
misconfigured adapter fails immediately rather than after an hour of training
that changed nothing:

```
[PASS] base parameters are frozen             all base parameters have requires_grad=False
[PASS] LoRA parameters are trainable          16 adapter tensor(s), 16.4K parameters
[PASS] target modules are ['q_proj', 'v_proj'] adapted ['q_proj', 'v_proj']
[PASS] trainable parameters are a small fraction of the total  16,384 of 681,856 (2.40%, 42x smaller)
```

`repairllama model-check --attach-lora` runs the same checks from the command
line. Note the third one in particular: a `target_modules` entry that matches
nothing produces an adapter that trains *zero* parameters, silently, and only
this check catches it.

**The adapter is never merged into the base.** Merging writes the adapter into
the base weights and destroys the property that makes this cheap — one frozen
base serving many adapters. `save_adapter` writes adapter tensors only (plus a
`repairllama_adapter.json` sidecar recording language, base model and LoRA
settings), and can verify the base weights are untouched before writing when
given a `BaseWeightGuard`.

### One base, many languages

Adapters live one directory per language under `paths.adapters_dir`:

```
adapters/
    java-repair/        implemented
    python-repair/      planned
    cpp-repair/         planned
```

`ADAPTER_PROFILES` is that registry and `repairllama init` creates a directory
for every implemented language. Only Java exists today; `profile_for("python")`
raises with a message saying `python-repair/` is reserved but empty, rather
than silently producing a Java adapter under a Python name.

### The dry run

```bash
repairllama model-check                                   # uses the config
repairllama model-check --model /path/to/checkpoint --device cpu --dtype bfloat16
repairllama model-check --attach-lora                     # + a fresh adapter
repairllama model-check --tokenizer-only                  # no weights read
repairllama model-check --no-load                         # resolve paths only
```

It loads the tokenizer and model, prints total / trainable / frozen parameter
counts (grouped by dtype), the memory footprint and device memory, then runs a
verification table:

```
[PASS] tokenizer has a pad token          </s>
[PASS] <FILL_ME> is a single token        1 token(s)
[PASS] model is on the resolved device    cpu vs cpu
[PASS] model is in the resolved dtype     bfloat16 vs bfloat16
[PASS] base weights are frozen            0 trainable
[PASS] embeddings match the tokenizer     35 vs 35
```

Exit codes: `0` all checks passed, `1` a check or the configuration failed,
`3` no local checkpoint (with instructions). Nothing is trained and no weights
are written.

## Training

```bash
repairllama train                                   # config defaults
repairllama train --dry-run                         # set up, print the banner, stop
python scripts/train_java_repair.py --epochs 2 --lr 5e-4
python scripts/train_java_repair.py --resume auto   # newest checkpoint
```

Both entry points take the same settings; every one of them lives in
`configs/java_repair.yaml` and can be overridden with `--set key=value`.

### Defaults

| Setting | Value |
| --- | --- |
| `learning_rate` | 5e-4 |
| `lr_scheduler` | cosine |
| `epochs` | 2 |
| `optimizer` | AdamW (`adamw_torch`) |
| `max_length` | 1024 |
| LoRA | r=8, alpha=16, dropout=0.05, `q_proj` + `v_proj` |

Also configurable: `mixed_precision` (auto/no/fp16/bf16),
`gradient_accumulation_steps`, `batch_size`, `warmup_ratio`, `weight_decay`,
`max_grad_norm`, `gradient_checkpointing`, `seed`, the logging/eval/save
strategies and intervals, `save_total_limit`, `early_stopping_patience` and
`resume_from_checkpoint`.

### On reproducing the paper

**These defaults match the hyper-parameters the RepairLLaMA paper reports.
That is not the same as reproducing the paper's results**, which also depend
on the training corpus, the effective batch size, the hardware and details the
paper does not report. Nothing here has been run against the paper's setup.
Every run records what actually executed in `training_summary.json`, including
a `reproduction` block that lists any hyper-parameter differing from the
reported value and states plainly that reproduction is unverified.

### What a run prints

Before the first step:

```
Base model:            codellama/CodeLlama-7b-hf
Device / dtype:        cuda:0 / bfloat16
Base parameters:       6,738,415,616 (6.7B, frozen)
Trainable parameters:  4,194,304 (4.2M)
Trainable percentage:  0.0622%
Dataset size:          train 48,213, validation 2,678
Max sequence length:   1024 tokens
Batch size:            4 x 8 accumulation = 32 effective
Learning rate:         0.0005
Epochs:                2
LoRA configuration:    r=8, alpha=16, dropout=0.05, targets=['q_proj', 'v_proj']
```

### Guarantees

- **Only the adapter learns.** The base is frozen at load time and again before
  PEFT wraps it, the four LoRA assertions run before the first step, and a
  `BaseWeightGuard` verifies after the last one that no base weight moved — a
  run that somehow modified the base fails rather than saving.
- **Only the adapter is saved.** `adapters/java-repair/` gets adapter weights,
  the tokenizer (which matters, since `<FILL_ME>` was added to the vocabulary)
  and `training_summary.json`. No optimizer state, no base checkpoint, nothing
  merged.
- **The prompt is not a target.** The collator masks prompt tokens with `-100`,
  so the loss is computed on the replacement hunk only. Without that mask most
  of the gradient signal would go into reproducing buggy Java.
- **Resuming continues.** `--resume auto` takes the newest checkpoint;
  a path that does not exist is an error rather than a silent fresh start.

### Artifacts

```
outputs/training/<experiment>/checkpoint-<step>/   resumable, rotated, disposable
adapters/java-repair/
    adapter_model.safetensors    the deliverable
    adapter_config.json
    tokenizer.json, tokenizer_config.json
    repairllama_adapter.json     language, base model, LoRA settings
    training_summary.json        losses, settings, environment, reproduction
```

`training_summary.json` records the banner values, the loss curve, every
evaluation with perplexity and token/sequence accuracy, the best validation
loss, how many examples were truncated, the full settings, the library
versions and the GPU.

## Configuration

One file, [`configs/java_repair.yaml`](configs/java_repair.yaml), mirrors the
dataclasses in [`src/repairllama/config.py`](src/repairllama/config.py). Every
section validates itself on load, so a bad value fails immediately with the
offending key named, rather than an hour into training. Unknown keys are
rejected — a typo is an error, not a silently ignored setting.

Relative paths resolve against the project root, not your working directory.

Override any value from the command line:

```bash
repairllama train --set training.epochs=5 --set model.device=cpu
```

Values are parsed as YAML scalars, so `5` is an int and `false` is a bool.

## Installation

Requires Python 3.10–3.12 (torch has no 3.13+ wheels yet).

```bash
./scripts/setup_env.sh          # venv + editable install + dev extras
./scripts/setup_env.sh --full   # also installs torch/transformers/peft/trl
```

Or manually:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"         # add ".[train]" for the model stack
```

## CLI

```bash
repairllama --help
repairllama config show            # print the merged configuration
repairllama config validate        # exit 0 if the config is well formed
repairllama init                   # create data/, models/, outputs/ ...
```

| Command | Does | Implemented |
| --- | --- | --- |
| `config show/validate/path` | inspect the merged config | yes |
| `init` | create the directories the config names | yes |
| `prepare-data` | clean, dedupe, represent, split, write JSONL | yes |
| `model-check` | dry run: load, count parameters, verify device/dtype | yes |
| `localize` | run fault localization | no |
| `train` | fine-tune the LoRA adapter, save the artifact | yes |
| `infer` | sample candidate patches | no |
| `patch` | apply a candidate to a file | no |
| `evaluate` | compile + test candidates, score | no |

Global flags on every command: `-c/--config`, `-s/--set KEY=VALUE`
(repeatable), `--log-level`, `--no-log-file`.

Exit codes: `0` success, `1` usage or configuration error, `2` stage not
implemented yet, `3` no local model checkpoint.

## Tests

```bash
./scripts/run_tests.sh                 # structure check + pytest
python scripts/check_structure.py      # layout + imports only, no pytest
python -m pytest tests/test_config.py  # one file
```

`tests/` mirrors the stage packages. Two suites are real so far:

- `tests/representation/` — `test_ir4.py`, `test_or2.py`, `test_builder.py` and
  `test_scenarios.py` (one class per bug shape: one-line, multi-line, nested
  if/else, loops, region at the start, region at the end, malformed
  boundaries), over the Java fixtures in `samples.py`.
- `tests/data/` — one file per pipeline module, over the corpora built by
  `java_pairs.py`: every transformation, every rejection reason, split
  determinism, and the JSONL/report output.
- `tests/model/` and `tests/training/` — integration tests against a tiny Llama
  built locally (`tiny_checkpoint.py`), including real short training runs that
  check the base is unchanged, the artifact holds only the adapter, and
  `--resume` continues rather than restarting.

The remaining stage directories hold placeholder suites that assert the package
imports and document the cases they will cover.

## Next phases

1. **Localization** — perfect localization from the reference diff, then
   stack-trace and spectrum strategies feeding the IR4 region.
2. **Inference** — sampling candidate patches from the trained adapter.
3. **Patching** — hunk application and syntax validation.
4. **Evaluation** — sandboxed maven/gradle runs, plausible/correct metrics.
