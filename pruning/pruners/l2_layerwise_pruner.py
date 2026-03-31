# pruning/pruners/l2_layerwise_pruner.py
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from common.measurements import ceil_to_multiple, floor_to_multiple
from .base_pruner import Pruner


class L2LayerwisePruner(Pruner):
    """Layer-by-layer structured pruning using L2 norm criterion.

    Instead of Taylor criterion (which needs forward+backward passes and gradient
    accumulation), this pruner scores each output channel by the L2 norm of its
    filter weights: score(c) = ||W_c||_2.

    Channels with the smallest L2 norm are considered least important and are
    pruned first, layer by layer.

    Layer-by-layer means each layer gets its own per-layer budget derived from
    the global/scope budget proportionally (by current out_channels share), rather
    than ranking all channels globally and letting the lowest-scoring ones win
    regardless of which layer they come from.

    This avoids the common failure mode of global ranking where one large layer
    loses almost all its channels while small layers are untouched.

    Args:
        net: the UNet model.
        flops_reg: kept for interface compatibility, not used in L2 scoring.
        scope: same as FriendlyBudgetPruner ("global", "encoder", "bottleneck",
               "decoder", "custom", "pooled_down", "custom_seq").
        include_inc: whether inc.* counts as encoder.
        budget: GroupBudget for custom/pooled_down scopes.
        min_keep_ch: minimum output channels to keep per conv.
        min_keep_ratio: minimum keep ratio per conv (0 disables).
        friendly_enabled: round per-layer budgets to a multiple of friendly_multiple.
        friendly_multiple: the channel multiple to round to.
        friendly_round: "up" or "down" rounding for friendly alignment.
        distribution: how to distribute the global budget across layers.
            "proportional" (default) — each layer's budget is proportional to
                its share of total out_channels in the eligible scope.
            "uniform" — each eligible layer gets the same number of channels pruned.
    """

    def __init__(
        self,
        net: nn.Module,
        flops_reg: float = 0.0,          # kept for interface compat, unused
        scope: str = "global",
        include_inc: bool = True,
        budget=None,                      # Optional[GroupBudget]
        min_keep_ch: int = 1,
        min_keep_ratio: float = 0.0,
        friendly_enabled: bool = True,
        friendly_multiple: int = 8,
        friendly_round: str = "up",
        distribution: str = "proportional",
        **kwargs,
    ):
        # Call Pruner.__init__ but then immediately remove the hooks it registered,
        # since L2 pruning is purely weight-based (no activations/gradients needed).
        super().__init__(net, flops_reg, min_keep_ch=min_keep_ch, min_keep_ratio=min_keep_ratio)

        self.scope = str(scope).lower().strip()
        self.include_inc = bool(include_inc)
        self.budget = budget
        self.friendly_enabled = bool(friendly_enabled)
        self.friendly_multiple = int(max(1, friendly_multiple))
        self.friendly_round = str(friendly_round).lower().strip()
        self.distribution = str(distribution).lower().strip()

        self._last_selected: List[Tuple[float, int, int]] = []
        self._effective_prune_last: Optional[int] = None

        # L2 scores: computed on demand from weights, not from activations.
        # Dict[layer_idx -> Tensor of shape [out_channels]]
        self._l2_scores: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Hook override — L2 pruner does not need forward/backward hooks.
    # We still register the module lists (convs, BNs, deconvs) which
    # base_pruner._register_hooks() builds as a side effect, so we keep
    # calling it but strip the actual hook handles afterward.
    # ------------------------------------------------------------------
    def _register_hooks(self) -> None:
        """Register module lists only; skip activation/gradient hooks."""
        for name, module in self.net.named_modules():
            if isinstance(module, nn.Conv2d):
                # Do NOT register forward/backward hooks — no activations needed.
                self.convs.append(module)
                self.conv_names.append(name)
            elif isinstance(module, nn.ConvTranspose2d):
                self.deconvs[name] = module
            elif isinstance(module, nn.BatchNorm2d):
                self.BNs.append(module)

    # ------------------------------------------------------------------
    # L2 score computation — purely weight-based, no data passes needed.
    # ------------------------------------------------------------------
    def compute_l2_scores(self) -> None:
        """Compute L2 norm of each output channel's filter for all eligible convs.

        score(layer, c) = ||W[c, :, :, :]||_2

        Results stored in self._l2_scores[layer_idx].
        """
        self._l2_scores = {}
        for layer_idx, conv in enumerate(self.convs):
            name = self.conv_names[layer_idx]
            if name.startswith("outc."):
                continue
            # W shape: [out_channels, in_channels, kH, kW]
            w = conv.weight.data.detach().float()
            # Flatten all dims except out_channels, then compute norm per channel
            scores = w.view(w.shape[0], -1).norm(p=2, dim=1)  # [out_channels]
            self._l2_scores[layer_idx] = scores.cpu()

    # ------------------------------------------------------------------
    # compute_rank is a no-op for L2 pruner — scores come from weights.
    # Kept so callers that call compute_rank() don't crash.
    # ------------------------------------------------------------------
    def compute_rank(self) -> None:
        """No-op: L2 pruner scores channels from weights, not activations."""
        self.clear_cache()

    # ------------------------------------------------------------------
    # calc_flops is kept for interface compat (pipeline calls it on i==0).
    # Returns empty list since L2 doesn't use FLOPs regularization.
    # ------------------------------------------------------------------
    def calc_flops(self) -> List[float]:
        self.flops = [0.0] * max(0, len(self.convs) - 1)
        return self.flops

    # ------------------------------------------------------------------
    # Scope helpers (mirrors SafeBudgetPruner)
    # ------------------------------------------------------------------
    def _group_of_layer(self, layer_idx: int) -> str:
        name = self.conv_names[layer_idx] if layer_idx < len(self.conv_names) else ""
        if name.startswith("outc."):
            return "other"
        if name.startswith("inc."):
            return "encoder" if self.include_inc else "other"
        if name.startswith(("down1.", "down2.", "down3.")):
            return "encoder"
        if name.startswith("down4."):
            return "bottleneck"
        if name.startswith(("up1.", "up2.", "up3.", "up4.")):
            return "decoder"
        return "other"

    def _is_down_layer(self, layer_idx: int) -> bool:
        return self._group_of_layer(layer_idx) in ("encoder", "bottleneck")

    def _eligible_layer_idxs(self, scope_override: Optional[str] = None) -> List[int]:
        """Return layer indices eligible under current scope."""
        scope = str(scope_override or self.scope).lower().strip()
        eligible = []
        for li in range(len(self.convs)):
            name = self.conv_names[li]
            if name.startswith("outc."):
                continue
            g = self._group_of_layer(li)
            if scope == "global":
                if g != "other":
                    eligible.append(li)
            elif scope in ("encoder", "bottleneck", "decoder"):
                if g == scope:
                    eligible.append(li)
            elif scope in ("custom", "pooled_down"):
                if g in ("encoder", "bottleneck", "decoder"):
                    eligible.append(li)
            else:
                if g != "other":
                    eligible.append(li)
        return eligible

    # ------------------------------------------------------------------
    # Per-layer budget distribution
    # ------------------------------------------------------------------
    def _layer_max_prune(self, layer_idx: int) -> int:
        """Maximum channels we are allowed to prune from this layer."""
        conv = self.convs[layer_idx] if layer_idx < len(self.convs) else None
        if conv is None:
            return 0
        name = self.conv_names[layer_idx]
        if name.startswith("outc."):
            return 0

        out_ch = int(conv.out_channels)
        min_keep_by_ratio = int(math.ceil(self.min_keep_ratio * out_ch)) if self.min_keep_ratio > 0 else 0
        min_keep = max(int(self.min_keep_ch), int(min_keep_by_ratio), 1)
        min_keep = min(min_keep, out_ch)
        max_prune = max(0, out_ch - min_keep)

        if self.friendly_enabled and self.friendly_multiple > 1 and out_ch >= self.friendly_multiple:
            max_prune = floor_to_multiple(max_prune, self.friendly_multiple)

        return int(max_prune)

    def _distribute_budget(self, layers: List[int], total_budget: int) -> Dict[int, int]:
        """Distribute total_budget channels across layers.

        Returns Dict[layer_idx -> channels_to_prune_from_this_layer].
        """
        if not layers or total_budget <= 0:
            return {}

        layer_max = {li: self._layer_max_prune(li) for li in layers}
        cap_total = sum(layer_max.values())
        if cap_total <= 0:
            return {}

        effective_budget = min(int(total_budget), int(cap_total))

        if self.distribution == "uniform":
            base = effective_budget // len(layers)
            remainder = effective_budget - base * len(layers)
            raw = {li: base + (1 if i < remainder else 0) for i, li in enumerate(layers)}
        else:
            # proportional: weight by out_channels
            out_chs = {li: int(self.convs[li].out_channels) for li in layers}
            total_ch = float(sum(out_chs.values()))
            raw = {}
            allocated = 0
            for i, li in enumerate(layers):
                if i == len(layers) - 1:
                    raw[li] = effective_budget - allocated
                else:
                    share = int(round(effective_budget * out_chs[li] / total_ch))
                    raw[li] = share
                    allocated += share

        # Apply friendly rounding and cap at layer_max
        result = {}
        for li in layers:
            alloc = int(raw.get(li, 0))
            alloc = min(alloc, int(layer_max[li]))
            alloc = max(alloc, 0)
            if self.friendly_enabled and self.friendly_multiple > 1:
                if self.friendly_round == "down":
                    alloc = floor_to_multiple(alloc, self.friendly_multiple)
                else:
                    alloc = min(
                        ceil_to_multiple(alloc, self.friendly_multiple),
                        int(layer_max[li])
                    )
            result[li] = max(0, int(alloc))

        return result

    # ------------------------------------------------------------------
    # Channel selection — layer-by-layer L2 lowest first
    # ------------------------------------------------------------------
    def _select_from_layer(self, layer_idx: int, n: int) -> List[Tuple[float, int, int]]:
        """Select n lowest-L2 channels from a single layer."""
        if n <= 0:
            return []
        if layer_idx not in self._l2_scores:
            return []

        scores = self._l2_scores[layer_idx]  # [out_channels]
        n = min(int(n), int(scores.numel()))
        if n <= 0:
            return []

        vals, idxs = torch.topk(scores, n, largest=False)
        return [(float(vals[i].item()), int(layer_idx), int(idxs[i].item())) for i in range(n)]

    def _rank_channels(self, prune_channels: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select channels layer-by-layer using L2 norm scores."""

        # Ensure L2 scores are fresh
        if not self._l2_scores:
            self.compute_l2_scores()

        chosen: List[Tuple[float, int, int]] = []

        if self.scope in ("custom", "pooled_down"):
            if self.budget is None:
                raise ValueError("Budget scope selected but budget is None.")
            b = self.budget

            if b.prune_down_total is not None:
                down_layers = [li for li in self._eligible_layer_idxs("global") if self._is_down_layer(li)]
                dec_layers = [li for li in self._eligible_layer_idxs("global") if self._group_of_layer(li) == "decoder"]
                down_budgets = self._distribute_budget(sorted(down_layers), int(b.prune_down_total))
                dec_budgets = self._distribute_budget(sorted(dec_layers), int(b.prune_decoder))
                for li, n in {**down_budgets, **dec_budgets}.items():
                    chosen += self._select_from_layer(li, n)
            else:
                enc_layers = [li for li in self._eligible_layer_idxs("global") if self._group_of_layer(li) == "encoder"]
                bot_layers = [li for li in self._eligible_layer_idxs("global") if self._group_of_layer(li) == "bottleneck"]
                dec_layers = [li for li in self._eligible_layer_idxs("global") if self._group_of_layer(li) == "decoder"]
                enc_budgets = self._distribute_budget(sorted(enc_layers), int(b.prune_encoder))
                bot_budgets = self._distribute_budget(sorted(bot_layers), int(b.prune_bottleneck))
                dec_budgets = self._distribute_budget(sorted(dec_layers), int(b.prune_decoder))
                for li, n in {**enc_budgets, **bot_budgets, **dec_budgets}.items():
                    chosen += self._select_from_layer(li, n)

        else:
            layers = self._eligible_layer_idxs()
            layer_budgets = self._distribute_budget(sorted(layers), int(prune_channels))
            self._effective_prune_last = int(sum(layer_budgets.values()))
            for li, n in layer_budgets.items():
                chosen += self._select_from_layer(li, n)

        # Dedup (shouldn't be needed for layer-by-layer but kept for safety)
        seen = set()
        dedup: List[Tuple[float, int, int]] = []
        for s, l, c in chosen:
            key = (int(l), int(c))
            if key in seen:
                continue
            seen.add(key)
            dedup.append((float(s), int(l), int(c)))

        self._last_selected = dedup
        self._effective_prune_last = len(dedup)

        layer_tensor = torch.tensor([l for (_, l, _) in dedup], dtype=torch.long, device="cpu")
        chan_tensor = torch.tensor([c for (_, _, c) in dedup], dtype=torch.long, device="cpu")
        return layer_tensor, chan_tensor

    # ------------------------------------------------------------------
    # preview_selection — mirrors FriendlyBudgetPruner interface
    # ------------------------------------------------------------------
    def preview_selection(self, requested_prune: int):
        self.compute_l2_scores()
        return self._rank_channels(requested_prune)

    # ------------------------------------------------------------------
    # Logging — adapted for L2 stats
    # ------------------------------------------------------------------
    def log_rank_stats(self, log, max_layers: int = 200) -> None:
        if not self._l2_scores:
            self.compute_l2_scores()

        log.info("=" * 92)
        log.info("L2 NORM STATS (per layer)")
        log.info("=" * 92)

        for li in sorted(self._l2_scores.keys())[:max_layers]:
            name = self.conv_names[li] if li < len(self.conv_names) else f"layer{li}"
            conv = self.convs[li] if li < len(self.convs) else None
            out_ch = int(conv.out_channels) if conv is not None else -1
            g = self._group_of_layer(li)

            scores = self._l2_scores[li]
            mn = float(scores.min().item())
            md = float(scores.median().item())
            mx = float(scores.max().item())
            std = float(scores.std().item()) if scores.numel() > 1 else 0.0

            log.info(
                f"  - {name:40s} | group={g:10s} | out_ch={out_ch:4d} | "
                f"L2(min/med/max/std)={mn:.4f}/{md:.4f}/{mx:.4f}/{std:.4f}"
            )
        log.info("=" * 92)

    def log_selection_distribution(self, log, selected: List[Tuple[float, int, int]]) -> None:
        by_conv: Dict[str, int] = {}
        by_group: Dict[str, int] = {"encoder": 0, "bottleneck": 0, "decoder": 0, "other": 0}

        for _, li, _ in selected:
            name = self.conv_names[li] if li < len(self.conv_names) else f"layer{li}"
            by_conv[name] = by_conv.get(name, 0) + 1
            g = self._group_of_layer(li)
            by_group[g] = by_group.get(g, 0) + 1

        log.info("=" * 92)
        log.info("SELECTION DISTRIBUTION")
        log.info("=" * 92)
        log.info(f"Group counts: {json.dumps(by_group, sort_keys=True)}")

        top = sorted(by_conv.items(), key=lambda x: x[1], reverse=True)[:25]
        log.info("Top convs by selected channels:")
        for n, c in top:
            log.info(f"  - {n:40s}: {c}")
        log.info("=" * 92)
