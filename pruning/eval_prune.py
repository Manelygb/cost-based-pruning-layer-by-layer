# pruning/eval_prune.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from common.metrics import dice_coeff
from .p_utils.utils import batch


def _format_true_masks(true_masks: torch.Tensor) -> torch.Tensor:
    """Return target mask as LongTensor [B,H,W] with values in {0,1}."""
    if true_masks.ndim == 4 and true_masks.shape[1] == 1:
        true_masks = true_masks[:, 0, :, :]

    if true_masks.max() > 1:
        true_masks = (true_masks > 0).long()
    else:
        true_masks = true_masks.long()

    return true_masks


def eval_net(net, dataset, lendata: int, gpu: bool = False, batch_size: int = 8, is_loss: bool = False) -> float:
    """Evaluate using Dice (default) or loss.

    Supported outputs:
    - 2-class logits: [B,2,H,W] -> argmax for dice; CrossEntropyLoss for loss-mode
    - 1-channel logits: [B,1,H,W] or [B,H,W] -> sigmoid+threshold for dice; sigmoid+BCE for loss-mode
    """
    was_training = net.training
    net.eval()

    device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")
    net = net.to(device)

    criterion_bce = nn.BCELoss()

    tot = 0.0
    n_batches = 0

    with torch.no_grad(), tqdm(total=lendata) as progress_bar:
        for b in batch(dataset, batch_size):
            imgs = np.array([i[0] for i in b], dtype=np.float32)
            true_masks = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs).to(device=device, dtype=torch.float32)
            true_masks = torch.from_numpy(true_masks).to(device=device)

            true_masks = _format_true_masks(true_masks)  # [B,H,W] long

            out = net(imgs)
            if isinstance(out, (tuple, list)):
                out = out[0]

            if is_loss:
                if out.ndim == 4 and out.shape[1] == 2:
                    ce = nn.CrossEntropyLoss()
                    loss = ce(out, true_masks)
                    tot += float(loss.item())
                    progress_bar.set_postfix(CE=float(loss.item()))
                else:
                    if out.ndim == 4 and out.shape[1] == 1:
                        out = out[:, 0, :, :]
                    prob = torch.sigmoid(out)
                    loss = criterion_bce(prob, true_masks.float())
                    tot += float(loss.item())
                    progress_bar.set_postfix(BCE=float(loss.item()))
            else:
                if out.ndim == 4 and out.shape[1] == 2:
                    pred = torch.argmax(out, dim=1)  # [B,H,W]
                else:
                    if out.ndim == 4 and out.shape[1] == 1:
                        out = out[:, 0, :, :]
                    prob = torch.sigmoid(out)
                    pred = (prob > 0.5).long()

                dice = float(dice_coeff(pred.float(), true_masks.float()).item())
                tot += dice
                progress_bar.set_postfix(DICE=dice)

            n_batches += 1
            progress_bar.update(len(b))

    if was_training:
        net.train()

    return float(tot / max(1, n_batches))
