# pruning/pruners/taylor.py
from __future__ import annotations

import torch.nn as nn


def set_model_for_taylor(net: nn.Module, freeze_bn: bool = True) -> None:
    """Put model in train mode for Taylor, optionally freezing BN stats."""
    net.train()
    if freeze_bn:
        for m in net.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()
