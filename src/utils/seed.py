"""
Reproducibility helper — call once at the top of every notebook/script.
"""
import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy, and torch (if installed) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch not installed yet is fine for Phase 0 (no model code runs here)
