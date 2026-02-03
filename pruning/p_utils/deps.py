from __future__ import annotations

from typing import Dict

import torch


_PART_PREFIXES = {
    "encoder": ("inc.", "down1.", "down2.", "down3."),
    "bottleneck": ("down4.",),
    "decoder": ("up1.", "up2.", "up3.", "up4.", "outc."),
}


def _part_of(name: str) -> str:
    for part, prefs in _PART_PREFIXES.items():
        if any(name.startswith(p) for p in prefs):
            return part
    return "other"


def take_dep_snapshot(model: torch.nn.Module) -> Dict[str, int]:
    """
    Snapshot "capacity" per part as sum(out_channels) over Conv2d modules.
    Used to estimate dependency pruning (changes in later parts after earlier pruning).
    """
    snap = {"encoder": 0, "bottleneck": 0, "decoder": 0, "other": 0}
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            part = _part_of(name)
            snap[part] += int(m.out_channels)
    return snap


def dep_pruned_counts(base: Dict[str, int], cur: Dict[str, int]) -> Dict[str, int]:
    out = {}
    for k in ("encoder", "bottleneck", "decoder"):
        out[f"{k}_dep"] = max(0, int(base.get(k, 0)) - int(cur.get(k, 0)))
    return out
