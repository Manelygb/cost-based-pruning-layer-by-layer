"""
common/reproducibility.py

Reproducibility utilities: seed Python/NumPy/PyTorch and configure CuDNN determinism.

Logic kept identical to your original function.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_all(seed: int, use_cuda: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
