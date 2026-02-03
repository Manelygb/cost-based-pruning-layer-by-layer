# pruning/pruners/custom_seq.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch.nn as nn


@dataclass
class DepSnapshot:
    """Minimal channel-dimension snapshot to measure dependency-induced pruning."""
    down4_in: int
    dec_deconv_in: int
    dec_deconv_out: int


def compute_targets_from_pct(total: int, enc_pct: float, bot_pct: float, dec_pct: float) -> Tuple[int, int, int]:
    total = int(total)
    if total < 0:
        raise ValueError("total budget must be >= 0")

    p = [float(enc_pct), float(bot_pct), float(dec_pct)]
    if all(0.0 <= x <= 1.0 for x in p) and sum(p) <= 1.01:
        p = [x * 100.0 for x in p]

    s = float(sum(p))
    if s <= 0:
        return 0, 0, 0

    p = [x / s for x in p]
    enc = int(total * p[0])
    bot = int(total * p[1])
    dec = int(total - enc - bot)
    return max(enc, 0), max(bot, 0), max(dec, 0)


def take_dep_snapshot(net: nn.Module) -> DepSnapshot:
    down4_in = int(net.down4.maxpool_conv[1].double_conv[0].weight.shape[1])

    dec_in = 0
    dec_out = 0
    for i in (1, 2, 3, 4):
        up = getattr(net, f"up{i}")
        dec_in += int(up.up.weight.shape[0])   # in_channels
        dec_out += int(up.up.weight.shape[1])  # out_channels

    return DepSnapshot(down4_in=down4_in, dec_deconv_in=dec_in, dec_deconv_out=dec_out)


def dep_pruned_counts(base: DepSnapshot, cur: DepSnapshot) -> dict:
    bot_dep = max(0, int(base.down4_in) - int(cur.down4_in))
    dec_dep = max(
        0,
        (int(base.dec_deconv_in) - int(cur.dec_deconv_in))
        + (int(base.dec_deconv_out) - int(cur.dec_deconv_out)),
    )
    return {"bottleneck_dep": int(bot_dep), "decoder_dep": int(dec_dep)}
