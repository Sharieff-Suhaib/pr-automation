"""Reproducibility helpers.

Seeding torch/numpy is optional: they are imported lazily so the utility works
in a bare environment that only has the standard library installed.
"""

from __future__ import annotations

import os
import random
from typing import List

__all__ = ["set_seed"]


def set_seed(seed: int, deterministic: bool = False) -> List[str]:
    """Seed every RNG that is importable; return the names of those seeded."""
    seeded = ["python"]
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
        seeded.append("numpy")

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        seeded.append("torch")

    return seeded
