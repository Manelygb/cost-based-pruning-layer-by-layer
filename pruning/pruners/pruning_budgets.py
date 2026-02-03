# pruning/pruners/budgets.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GroupBudget:
    prune_down_total: Optional[int] = None  # pooled encoder+bottleneck
    prune_encoder: int = 0
    prune_bottleneck: int = 0
    prune_decoder: int = 0

    def total(self) -> int:
        if self.prune_down_total is not None:
            return int(self.prune_down_total) + int(self.prune_decoder)
        return int(self.prune_encoder) + int(self.prune_bottleneck) + int(self.prune_decoder)


def make_group_budget(
    prune_down_total: Optional[int] = None,
    prune_encoder: int = 0,
    prune_bottleneck: int = 0,
    prune_decoder: int = 0,
) -> GroupBudget:
    if prune_down_total is not None:
        return GroupBudget(
            prune_down_total=int(prune_down_total),
            prune_encoder=0,
            prune_bottleneck=0,
            prune_decoder=int(prune_decoder),
        )
    return GroupBudget(
        prune_down_total=None,
        prune_encoder=int(prune_encoder),
        prune_bottleneck=int(prune_bottleneck),
        prune_decoder=int(prune_decoder),
    )


def resolve_budget_and_requested(
    scope: str,
    prune_channels: int,
    prune_encoder: int = 0,
    prune_bottleneck: int = 0,
    prune_decoder: int = 0,
    prune_down_total: Optional[int] = None,
) -> Tuple[Optional[GroupBudget], int, str]:
    """Resolve the pruning budget given a scope and user budgets."""
    sc = str(scope).lower().strip()

    if sc == "global":
        return None, int(prune_channels), "GLOBAL"

    if sc in ("encoder", "bottleneck", "decoder"):
        b = make_group_budget(
            prune_encoder=int(prune_channels) if sc == "encoder" else 0,
            prune_bottleneck=int(prune_channels) if sc == "bottleneck" else 0,
            prune_decoder=int(prune_channels) if sc == "decoder" else 0,
        )
        return b, int(b.total()), sc.upper()

    if sc == "custom":
        pe, pb, pd = int(prune_encoder), int(prune_bottleneck), int(prune_decoder)
        if pe + pb + pd <= 0:
            raise ValueError("custom scope requires at least one of --prune_encoder/--prune_bottleneck/--prune_decoder > 0")
        b = make_group_budget(prune_encoder=pe, prune_bottleneck=pb, prune_decoder=pd)
        return b, int(b.total()), "CUSTOM (encoder/bottleneck/decoder)"

    if sc == "pooled_down":
        if prune_down_total is None:
            raise ValueError("pooled_down scope requires --prune_down_total")
        d = int(prune_down_total)
        if d < 0:
            raise ValueError("pooled_down scope requires --prune_down_total >= 0")
        pd = int(prune_decoder)
        b = make_group_budget(prune_down_total=d, prune_decoder=pd)
        return b, int(b.total()), "POOLED-DOWN (encoder+bottleneck) + decoder"

    raise ValueError(f"Unknown scope: {scope}")
