"""Code representations for Java program repair (IR4 x OR2).

The pair implemented here is the one the RepairLLaMA study found most
effective, and the two halves live in their own modules:

    ir4.py       input: buggy lines kept but commented out, bracketed by
                 markers, followed by a <FILL_ME> token in their place
    or2.py       output: only the replacement code for that region
    builder.py   the public API tying them together

Typical use::

    from repairllama.representation import build_training_example

    example = build_training_example(buggy_src, fixed_src, 12, 14)
    example.prompt   # IR4 input
    example.target   # OR2 target

Line numbers are 1-based and inclusive everywhere in this package, and the
Java source is never rewritten — see the fidelity contract in ``ir4``.
"""

from repairllama.representation.builder import (
    InferenceInput,
    RepresentationOptions,
    TrainingExample,
    build_inference_input,
    build_training_example,
    build_training_examples,
    decode_prediction,
    with_context,
)
from repairllama.representation.ir4 import (
    DEFAULT_FILL_TOKEN,
    IR4Options,
    RegionMetadata,
    RepresentationError,
    comment_line,
    describe_region,
    extract_region,
    recover_region_from_prompt,
    render_ir4,
    split_lines,
    uncomment_line,
    validate_region,
)
from repairllama.representation.or2 import (
    FixedRegion,
    OR2Options,
    OR2Prediction,
    align_fixed_region,
    build_or2_target,
    parse_or2_output,
    restore_indentation,
    splice_region,
)

__all__ = [
    # builder
    "build_training_example",
    "build_inference_input",
    "build_training_examples",
    "decode_prediction",
    "with_context",
    "RepresentationOptions",
    "TrainingExample",
    "InferenceInput",
    # ir4
    "IR4Options",
    "RegionMetadata",
    "RepresentationError",
    "DEFAULT_FILL_TOKEN",
    "split_lines",
    "validate_region",
    "describe_region",
    "extract_region",
    "render_ir4",
    "recover_region_from_prompt",
    "comment_line",
    "uncomment_line",
    # or2
    "OR2Options",
    "OR2Prediction",
    "FixedRegion",
    "align_fixed_region",
    "build_or2_target",
    "parse_or2_output",
    "restore_indentation",
    "splice_region",
]
