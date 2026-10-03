from datasets import load_dataset, Dataset
from difflib import SequenceMatcher
import json
import re


# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "Harryxun/BugsInPy_data"

OUTPUT_FILE = "bugsinpy_ir4_or2.jsonl"

MIN_CHANGED_LINES = 1


# ============================================================
# LOAD DATASET
# ============================================================

dataset = load_dataset(DATASET_NAME)

data = dataset["train"]

print(data)
print("Number of examples:", len(data))


# ============================================================
# NORMALIZE SOURCE
# ============================================================

def normalize_source(code):
    code = code.replace("\r\n", "\n")
    code = code.replace("\r", "\n")

    # Convert non-breaking spaces to ordinary spaces
    code = code.replace("\u00a0", " ")

    return code


# ============================================================
# FIND CHANGED REGION
# ============================================================

def find_changed_region(buggy, fixed):
    """
    Compare buggy and fixed source line-by-line.

    Returns:

        buggy_start
        buggy_end
        fixed_start
        fixed_end

    using Python slicing conventions:

        lines[start:end]
    """

    buggy_lines = buggy.splitlines()
    fixed_lines = fixed.splitlines()

    matcher = SequenceMatcher(
        None,
        buggy_lines,
        fixed_lines,
        autojunk=False
    )

    opcodes = matcher.get_opcodes()

    changed = []

    for tag, i1, i2, j1, j2 in opcodes:

        if tag != "equal":
            changed.append(
                (i1, i2, j1, j2)
            )

    if not changed:
        return None

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We take the complete span from the first changed
    # location to the last changed location.
    #
    # This allows multi-location bugs to be represented
    # as ONE repair region.
    # --------------------------------------------------------

    buggy_start = min(x[0] for x in changed)
    buggy_end = max(x[1] for x in changed)

    fixed_start = min(x[2] for x in changed)
    fixed_end = max(x[3] for x in changed)

    return {
        "buggy_start": buggy_start,
        "buggy_end": buggy_end,
        "fixed_start": fixed_start,
        "fixed_end": fixed_end,
        "opcodes": opcodes
    }


# ============================================================
# CREATE IR4
# ============================================================

def make_ir4(buggy, region):

    lines = buggy.splitlines()

    start = region["buggy_start"]
    end = region["buggy_end"]

    prefix = lines[:start]
    buggy_region = lines[start:end]
    suffix = lines[end:]

    result = []

    # Unchanged code before repair region
    result.extend(prefix)

    # Buggy region marker
    result.append("# BUGGY CODE")

    # Preserve every original line exactly
    for line in buggy_region:

        if line == "":
            result.append("#")
        else:
            result.append("# " + line)

    # Model must generate this region
    result.append("<FILL_ME>")

    # Unchanged code after repair region
    result.extend(suffix)

    return "\n".join(result)
# ============================================================
# CREATE OR2
# ============================================================

def make_or2(fixed, region):
    """
    OR2 = fixed chunk corresponding to the localized
    buggy region.
    """

    fixed_lines = fixed.splitlines()

    start = region["fixed_start"]
    end = region["fixed_end"]

    fixed_region = fixed_lines[start:end]

    return "\n".join(fixed_region)


# ============================================================
# OPTIONAL: VERIFY RECONSTRUCTION
# ============================================================

def reconstruct_from_or2(buggy, fixed, region):
    """
    Replace buggy region with OR2.

    This should reconstruct the fixed source for ordinary
    localized patches.
    """

    buggy_lines = buggy.splitlines()
    fixed_lines = fixed.splitlines()

    b_start = region["buggy_start"]
    b_end = region["buggy_end"]

    f_start = region["fixed_start"]
    f_end = region["fixed_end"]

    or2 = fixed_lines[f_start:f_end]

    reconstructed = (
        buggy_lines[:b_start]
        + or2
        + buggy_lines[b_end:]
    )

    return "\n".join(reconstructed)


# ============================================================
# PROCESS DATASET
# ============================================================

records = []

skipped = 0
successful = 0

for idx, example in enumerate(data):

    buggy = normalize_source(example["buggy"])
    fixed = normalize_source(example["fixed"])

    # --------------------------------------------------------
    # Find changed region
    # --------------------------------------------------------

    region = find_changed_region(
        buggy,
        fixed
    )

    if region is None:

        skipped += 1
        continue

    # --------------------------------------------------------
    # Generate IR4
    # --------------------------------------------------------

    ir4 = make_ir4(
        buggy,
        region
    )

    # --------------------------------------------------------
    # Generate OR2
    # --------------------------------------------------------

    or2 = make_or2(
        fixed,
        region
    )

    # --------------------------------------------------------
    # Skip empty patches
    # --------------------------------------------------------

    if not or2.strip():

        skipped += 1
        continue

    # --------------------------------------------------------
    # Verify that replacing the buggy region with OR2
    # reconstructs the fixed code.
    # --------------------------------------------------------

    reconstructed = reconstruct_from_or2(
        buggy,
        fixed,
        region
    )

    # This check can fail for certain non-contiguous /
    # line-shift situations. We keep the example but mark it.
    reconstruction_ok = (
        reconstructed.strip()
        == fixed.strip()
    )

    # --------------------------------------------------------
    # Store metadata
    # --------------------------------------------------------

    record = {
        "project": example["project"],
        "bug_id": example["bug_id"],
        "language": "python",

        "input": ir4,
        "output": or2,

        "test_command": example["test_command"],

        "buggy_start": region["buggy_start"],
        "buggy_end": region["buggy_end"],

        "fixed_start": region["fixed_start"],
        "fixed_end": region["fixed_end"],

        "reconstruction_ok": reconstruction_ok
    }

    records.append(record)

    successful += 1


# ============================================================
# SAVE JSONL
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    for record in records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print("Original examples :", len(data))
print("Processed examples:", successful)
print("Skipped examples  :", skipped)

print()
print("Saved to:")
print(OUTPUT_FILE)