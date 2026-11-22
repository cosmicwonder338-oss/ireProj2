"""Fixed seeds for python, numpy and (if installed) torch."""
import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG we use. `deterministic=True` also forces deterministic cuDNN (slower)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rng(seed: int) -> np.random.Generator:
    """Independent numpy Generator — use this for sensor noise / placement so it doesn't depend on
    how many global random draws happened earlier."""
    return np.random.default_rng(seed)
