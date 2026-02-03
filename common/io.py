"""
common/io.py

I/O helpers that are reused across training, pruning, profiling, and inference.

This module intentionally stays small:
- safe loading of model state_dict files across PyTorch versions
- compact compatibility check between a checkpoint state_dict and a model state_dict
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import torch


def load_state_dict_weights_only(
    path: str,
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint that is expected to be a *pure* state_dict.

    PyTorch introduced `weights_only=True` for safer loading in some versions.
    This function uses it when available and falls back otherwise.

    Args:
        path: Path to a checkpoint file containing a state_dict.
        map_location: Passed to torch.load, e.g. "cpu", "cuda", or a torch.device.

    Returns:
        The loaded state_dict (mapping parameter/buffer names to tensors).
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older PyTorch versions do not support weights_only
        return torch.load(path, map_location=map_location)


def compare_state_dicts_compact(
    ckpt_sd: Dict[str, Any],
    model_sd: Dict[str, Any],
) -> Tuple[int, int, int, int]:
    """Compare two state_dict objects and return a compact mismatch summary.

    This is meant for logging/debugging before calling `load_state_dict`.

    Args:
        ckpt_sd: state_dict loaded from disk.
        model_sd: model.state_dict() of the current architecture.

    Returns:
        common: Number of keys present in both dicts.
        mismatch: Number of common keys whose tensor shapes differ.
        missing: Number of keys missing from ckpt compared to model.
        unexpected: Number of extra keys present in ckpt but not in model.
    """
    model_keys = set(model_sd.keys())
    ckpt_keys = set(ckpt_sd.keys())
    common_keys = model_keys & ckpt_keys

    mismatch = 0
    for k in common_keys:
        a = getattr(model_sd[k], "shape", None)
        b = getattr(ckpt_sd[k], "shape", None)
        if a is None or b is None:
            # non-tensor entries are unusual for state_dict; treat as mismatch
            mismatch += 1
        elif tuple(a) != tuple(b):
            mismatch += 1

    missing = len(model_keys - ckpt_keys)
    unexpected = len(ckpt_keys - model_keys)
    return len(common_keys), mismatch, missing, unexpected
