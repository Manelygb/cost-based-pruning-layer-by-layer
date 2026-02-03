"""
common/metrics.py

Reusable metric primitives (model-agnostic).

Contains:
- Dice coefficient (batch + per-sample autograd function) used in segmentation work.
- AverageMeter utility (simple running average).
"""

from __future__ import annotations

from typing import Optional

import torch


class DiceCoeff(torch.autograd.Function):
    """Dice coefficient for a single example (autograd-enabled).

    Notes:
    - Expects `input` and `target` to be tensors of the same shape.
    - For segmentation, you typically pass binary masks or probabilities in [0, 1].
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-4
        ctx.save_for_backward(input, target)

        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        union = input.sum() + target.sum() + eps

        ctx.inter = inter
        ctx.union = union
        return (2.0 * inter + eps) / union

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, target = ctx.saved_tensors
        inter = ctx.inter
        union = ctx.union

        grad_input = grad_target = None

        if ctx.needs_input_grad[0]:
            # d/dx ( (2*<x,y> + eps) / (sum(x)+sum(y)+eps) )
            grad_input = grad_output * 2.0 * (target * union - inter) / (union * union)

        # target is treated as constant
        return grad_input, grad_target


def dice_coeff(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Dice coefficient for a batch.

    Args:
        input: Tensor of shape (N, ...) where N is batch size.
        target: Tensor of shape (N, ...) matching input.

    Returns:
        Scalar tensor: average Dice across the batch.
    """
    if input.shape != target.shape:
        raise ValueError(f"dice_coeff: shape mismatch input={tuple(input.shape)} target={tuple(target.shape)}")

    # scalar accumulator on the same device/dtype as input
    acc = input.new_zeros(())
    n = int(input.shape[0])

    for i in range(n):
        acc = acc + DiceCoeff.apply(input[i], target[i])

    return acc / max(n, 1)


class AverageMeter:
    """Track running average of scalar values."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, num_samples: int = 1) -> None:
        """Update meter.

        Args:
            val: Value to accumulate (typically an average over num_samples).
            num_samples: Number of samples represented by `val`.
        """
        self.count += int(num_samples)
        self.sum += float(val) * int(num_samples)
        self.avg = self.sum / max(self.count, 1)
