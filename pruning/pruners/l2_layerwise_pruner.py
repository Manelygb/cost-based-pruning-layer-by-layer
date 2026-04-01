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

    Friendly rounding is applied ONCE to the total budget before distribution,
    not per-layer (which causes either inflation or starvation of small layers).

    Args:
        net: the UNet model.
        flops_reg: kept for interface compatibility, not used in L2 scoring.
        scope: same as FriendlyBudgetPruner ("global", "encoder", "bottleneck",
               "decoder", "custom", "pooled_down", "custom_seq").
        include_inc: whether inc.* counts as encoder.
        budget: GroupBudget for custom/pooled_down scopes.
        min_keep_ch: minimum output channels to keep per conv.
        min_keep_ratio: minimum keep ratio per conv (0 disables).
        friendly_enabled: snap final total to a multiple of friendly_multiple.
        friendly_multiple: the channel multiple to snap to.
        friendly_round: "up" or "down" for the total snapping.
        distribution: how to distribute the budget across layers.
            "proportional" (default) — proportional to each layer's out_channels.
            "uniform"                — equal share per eligible layer.
    """

    def __init__(
        self,
        net: nn.Module,
        flops_reg: float = 0.0,
        scope: str = "global",
        include_inc: bool = True,
        budget=None,
        min_keep_ch: int = 1,
        min_keep_ratio: float = 0.0,
        friendly_enabled: bool = True,
        friendly_multiple: int = 8,
        friendly_round: str = "up",
        distribution: str = "proportional",
        **kwargs,
    ):
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
        self._l2_scores: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Hook override — L2 pruner does not need forward/backward hooks.
    # ------------------------------------------------------------------
    def _register_hooks(self) -> None:
        for name, module in self.net.named_modules():
            if isinstance(module, nn.Conv2d):
                self.convs.append(module)
                self.conv_names.append(name)
            elif isinstance(module, nn.ConvTranspose2d):
                self.deconvs[name] = module
            elif isinstance(module, nn.BatchNorm2d):
                self.BNs.append(module)

    # ------------------------------------------------------------------
    # L2 score computation
    # ------------------------------------------------------------------
    def compute_l2_scores(self) -> None:
        """Compute L2 norm of each output channel's filter for all eligible convs.

        score(layer, c) = ||W[c, :, :, :]||_2
        """
        self._l2_scores = {}
        for layer_idx, conv in enumerate(self.convs):
            name = self.conv_names[layer_idx]
            if name.startswith("outc."):
                continue
            w = conv.weight.data.detach().float()
            scores = w.view(w.shape[0], -1).norm(p=2, dim=1)  # [out_channels]
            self._l2_scores[layer_idx] = scores.cpu()

    def compute_rank(self) -> None:
        """No-op: L2 pruner scores channels from weights, not activations."""
        self.clear_cache()

    def calc_flops(self) -> List[float]:
        self.flops = [0.0] * max(0, len(self.convs) - 1)
        return self.flops

    # ------------------------------------------------------------------
    # Scope helpers
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
    # Friendly rounding — applied ONCE to the total budget
    # ------------------------------------------------------------------
    def _snap_total(self, total: int) -> int:
        """Snap total budget to friendly_multiple, applied once before distribution."""
        if not self.friendly_enabled or self.friendly_multiple <= 1:
            return int(total)
        if self.friendly_round == "down":
            return floor_to_multiple(int(total), self.friendly_multiple)
        return ceil_to_multiple(int(total), self.friendly_multiple)

    # ------------------------------------------------------------------
    # Per-layer max prune (hard cap, no rounding)
    # ------------------------------------------------------------------
    def _layer_max_prune(self, layer_idx: int) -> int:
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
        return max(0, out_ch - min_keep)

    # ------------------------------------------------------------------
    # Budget distribution — raw proportional/uniform, NO per-layer rounding
    # ------------------------------------------------------------------
    def _distribute_budget(self, layers: List[int], total_budget: int) -> Dict[int, int]:
        """Distribute total_budget across layers without any per-layer rounding.

        Friendly rounding must already have been applied to total_budget before
        calling this. Per-layer allocations are plain integers, capped at each
        layer's hard max.
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
            # proportional by out_channels
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

        # Cap at layer max only — no rounding
        result = {}
        for li in layers:
            alloc = int(raw.get(li, 0))
            alloc = min(alloc, int(layer_max[li]))
            alloc = max(alloc, 0)
            result[li] = alloc

        return result

    # ------------------------------------------------------------------
    # Channel selection — layer-by-layer, lowest L2 first
    # ------------------------------------------------------------------
    def _select_from_layer(self, layer_idx: int, n: int) -> List[Tuple[float, int, int]]:
        if n <= 0 or layer_idx not in self._l2_scores:
            return []
        scores = self._l2_scores[layer_idx]
        n = min(int(n), int(scores.numel()))
        if n <= 0:
            return []
        vals, idxs = torch.topk(scores, n, largest=False)
        return [(float(vals[i].item()), int(layer_idx), int(idxs[i].item())) for i in range(n)]

    def _rank_channels(self, prune_channels: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._l2_scores:
            self.compute_l2_scores()

        chosen: List[Tuple[float, int, int]] = []

        if self.scope in ("custom", "pooled_down"):
            if self.budget is None:
                raise ValueError("Budget scope selected but budget is None.")
            b = self.budget

            if b.prune_down_total is not None:
                down_layers = sorted([li for li in self._eligible_layer_idxs("global") if self._is_down_layer(li)])
                dec_layers  = sorted([li for li in self._eligible_layer_idxs("global") if self._group_of_layer(li) == "decoder"])
                down_budgets = self._distribute_budget(down_layers, self._snap_total(int(b.prune_down_total)))
                dec_budgets  = self._distribute_budget(dec_layers,  self._snap_total(int(b.prune_decoder)))
                for li, n in {**down_budgets, **dec_budgets}.items():
                    chosen += self._select_from_layer(li, n)
            else:
                enc_layers = sorted([li for li in self._eligible_layer_idxs("global") if self._group_of_layer(li) == "encoder"])
                bot_layers = sorted([li for li in self._eligible_layer_idxs("global") if self._group_of_layer(li) == "bottleneck"])
                dec_layers = sorted([li for li in self._eligible_layer_idxs("global") if self._group_of_layer(li) == "decoder"])
                enc_budgets = self._distribute_budget(enc_layers, self._snap_total(int(b.prune_encoder)))
                bot_budgets = self._distribute_budget(bot_layers, self._snap_total(int(b.prune_bottleneck)))
                dec_budgets = self._distribute_budget(dec_layers, self._snap_total(int(b.prune_decoder)))
                for li, n in {**enc_budgets, **bot_budgets, **dec_budgets}.items():
                    chosen += self._select_from_layer(li, n)

        else:
            # global / single-part scopes
            # Step 1: snap total once
            snapped_total = self._snap_total(int(prune_channels))

            # Step 2: distribute raw (no per-layer rounding)
            layers = self._eligible_layer_idxs()
            layer_budgets = self._distribute_budget(sorted(layers), snapped_total)

            # Step 3: select lowest-L2 channels per layer
            for li, n in layer_budgets.items():
                chosen += self._select_from_layer(li, n)

        # Dedup
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
        chan_tensor  = torch.tensor([c for (_, _, c) in dedup], dtype=torch.long, device="cpu")
        return layer_tensor, chan_tensor

    # ------------------------------------------------------------------
    # preview_selection
    # ------------------------------------------------------------------
    def preview_selection(self, requested_prune: int):
        self.compute_l2_scores()
        return self._rank_channels(requested_prune)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log_rank_stats(self, log, max_layers: int = 200) -> None:
        if not self._l2_scores:
            self.compute_l2_scores()

        log.info("=" * 92)
        log.info("L2 NORM STATS (per layer)")
        log.info("=" * 92)

        for li in sorted(self._l2_scores.keys())[:max_layers]:
            name  = self.conv_names[li] if li < len(self.conv_names) else f"layer{li}"
            conv  = self.convs[li] if li < len(self.convs) else None
            out_ch = int(conv.out_channels) if conv is not None else -1
            g     = self._group_of_layer(li)
            scores = self._l2_scores[li]

            mn  = float(scores.min().item())
            md  = float(scores.median().item())
            mx  = float(scores.max().item())
            std = float(scores.std().item()) if scores.numel() > 1 else 0.0

            log.info(
                f"  - {name:40s} | group={g:10s} | out_ch={out_ch:4d} | "
                f"L2(min/med/max/std)={mn:.4f}/{md:.4f}/{mx:.4f}/{std:.4f}"
            )
        log.info("=" * 92)

    def log_selection_distribution(self, log, selected: List[Tuple[float, int, int]]) -> None:
        by_conv: Dict[str, int]  = {}
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