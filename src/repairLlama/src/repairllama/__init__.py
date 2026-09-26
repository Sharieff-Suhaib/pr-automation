"""RepairLLaMA-Java — fine-tuned LLM program repair for Java.

The package is organised as a pipeline; each stage lives in its own subpackage
and is reachable from the ``repairllama`` CLI:

    data           acquire and split bug/fix pairs
    representation encode a bug into model input / decode model output
    localization   find the buggy region to put in the prompt
    model          load base checkpoints, tokenizers and LoRA adapters
    training       supervised fine-tuning (not yet implemented)
    inference      sample candidate patches
    patching       splice a generated hunk back into the source file
    evaluation     compile/test candidate patches and score them
    utils          logging, paths, seeding, IO

Nothing here imports torch or transformers at module load time, so importing
``repairllama`` stays cheap in environments without a GPU stack.
"""

from repairllama.config import RepairConfig, load_config

__version__ = "0.1.0"

__all__ = ["RepairConfig", "load_config", "__version__"]
