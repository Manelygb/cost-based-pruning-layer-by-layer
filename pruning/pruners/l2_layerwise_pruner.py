# pruning/pruners/l2_layerwise_pruner.py
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from common.measurements import ceil_to_multiple, floor_to_multiple
from .base_pruner import Pruner


# ---------------------------------------------------------------------------
# Pruning budget table — layer_name → cost_share% (from profiling notebook).
# Layers absent from this table receive 0 allocation (skipped).
# outc.conv is never pruned regardless.
#
# These cost_share values drive the same idea as the old per-part split
# (encoder≈24 % / bottleneck≈7 % / decoder≈69 %) but at layer granularity:
# a layer's share of the total prune budget == its share of total FLOPs.
# ---------------------------------------------------------------------------
_COST_SHARE_TABLE: Dict[str, float] = {
    # ── decoder (rank 1-4, cost_share 8.847% each × 4 = 35.4%) ──────────────
    "up4.conv.double_conv.0":              8.847247,
    "up1.conv.double_conv.0":             8.847247,
    "up2.conv.double_conv.0":             8.847247,
    "up3.conv.double_conv.0":             8.847247,
    # ── encoder + bottleneck + decoder (rank 5-13, 4.424% each × 9 = 39.8%) ─
    "inc.double_conv.3":                  4.423623,
    "down2.maxpool_conv.1.double_conv.3": 4.423623,
    "down1.maxpool_conv.1.double_conv.3": 4.423623,
    "up4.conv.double_conv.3":             4.423623,
    "up2.conv.double_conv.3":             4.423623,
    "down3.maxpool_conv.1.double_conv.3": 4.423623,
    "down4.maxpool_conv.1.double_conv.3": 4.423623,
    "up1.conv.double_conv.3":             4.423623,
    "up3.conv.double_conv.3":             4.423623,
    # ── decoder ConvTranspose2d (rank 14-17, 3.932% each × 4 = 15.7%) ───────
    "up1.up":                             3.932110,
    "up2.up":                             3.932110,
    "up4.up":                             3.932110,
    "up3.up":                             3.932110,
    # ── encoder + bottleneck (rank 18-21, 2.212% each × 4 = 8.8%) ───────────
    "down1.maxpool_conv.1.double_conv.0": 2.211812,
    "down2.maxpool_conv.1.double_conv.0": 2.211812,
    "down4.maxpool_conv.1.double_conv.0": 2.211812,
    "down3.maxpool_conv.1.double_conv.0": 2.211812,
    # ── inc input conv (rank 22, 0.207%) ────────────────────────────────────
    "inc.double_conv.0":                  0.207357,
    # outc.conv intentionally absent → never pruned
}


class L2LayerwisePruner(Pruner):
    """Layer-by-layer structured pruning using L2 norm criterion.

    Budget distribution
    -------------------
    Each eligible layer receives a share of the total prune budget that is
    proportional to its ``cost_share%`` from ``_COST_SHARE_TABLE``.  This
    mirrors the old per-part split (encoder / bottleneck / decoder) but at
    layer granularity — layers that account for more FLOPs get a larger slice
    of the pruning budget.

    Layers absent from ``_COST_SHARE_TABLE`` receive **zero** allocation and
    are skipped.  ``outc.conv`` is never pruned regardless.

    Friendly rounding is applied **once** to the total budget before
    distribution, not per-layer.

    Args:
        net: the UNet model.
        flops_reg: kept for interface compatibility, not used in L2 scoring.
        scope: same as FriendlyBudgetPruner
               (``"global"``, ``"encoder"``, ``"bottleneck"``, ``"decoder"``,
               ``"custom"``, ``"pooled_down"``, ``"custom_seq"``).
               Scope still gates *which* layers are candidates; cost-share
               weights determine the split *among* candidates.
        include_inc: whether ``inc.*`` counts as encoder.
        budget: GroupBudget for custom/pooled_down scopes.
        min_keep_ch: minimum output channels to keep per conv.
        min_keep_ratio: minimum keep ratio per conv (0 disables).
        friendly_enabled: snap final total to a multiple of friendly_multiple.
        friendly_multiple: the channel multiple to snap to.
        friendly_round: ``"up"`` or ``"down"`` for the total snapping.
        distribution: ``"cost_share"`` (default, table-driven) |
                      ``"proportional"`` (by out_channels, legacy) |
                      ``"uniform"`` (equal share per layer, legacy).
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
        distribution: str = "cost_share",
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
    # Cost-share weight lookup
    # ------------------------------------------------------------------
    def _cost_share_weight(self, layer_idx: int) -> float:
        """Return the cost_share% for this layer, or 0.0 if not in the table."""
        name = self.conv_names[layer_idx] if layer_idx < len(self.conv_names) else ""
        return _COST_SHARE_TABLE.get(name, 0.0)

    # ------------------------------------------------------------------
    # Budget distribution
    # ------------------------------------------------------------------
    def _distribute_budget(self, layers: List[int], total_budget: int) -> Dict[int, int]:
        """Distribute total_budget across layers.

        ``distribution == "cost_share"`` (default):
            Each layer's raw allocation = total_budget × (layer_cost_share /
            sum_of_cost_shares_for_eligible_layers).  Layers not in
            _COST_SHARE_TABLE receive weight 0 and are skipped entirely.

        ``distribution == "proportional"`` (legacy):
            Proportional to each layer's current out_channels.

        ``distribution == "uniform"`` (legacy):
            Equal share per eligible layer.

        Friendly rounding must already have been applied to total_budget.
        Per-layer allocations are plain integers, capped at each layer's hard max.
        """
        if not layers or total_budget <= 0:
            return {}

        layer_max = {li: self._layer_max_prune(li) for li in layers}
        cap_total = sum(layer_max.values())
        if cap_total <= 0:
            return {}

        effective_budget = min(int(total_budget), int(cap_total))

        # ── cost_share distribution ───────────────────────────────────────────
        if self.distribution == "cost_share":
            weights = {li: self._cost_share_weight(li) for li in layers}
            total_weight = sum(weights.values())

            if total_weight <= 0.0:
                # No layer in scope has a cost_share entry → fall back to
                # proportional so the run doesn't silently prune nothing.
                import warnings
                warnings.warn(
                    "L2LayerwisePruner: none of the eligible layers have a "
                    "cost_share entry in _COST_SHARE_TABLE; falling back to "
                    "proportional distribution.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self.distribution = "proportional"
            else:
                raw: Dict[int, int] = {}
                allocated = 0
                # Sort for determinism; handle the last layer's remainder
                eligible_with_weight = [(li, weights[li]) for li in sorted(layers) if weights[li] > 0.0]
                skipped              = [li for li in sorted(layers) if weights[li] <= 0.0]

                for i, (li, w) in enumerate(eligible_with_weight):
                    if i == len(eligible_with_weight) - 1:
                        raw[li] = effective_budget - allocated
                    else:
                        share = int(round(effective_budget * w / total_weight))
                        raw[li] = share
                        allocated += share

                for li in skipped:
                    raw[li] = 0

                result = {}
                for li in layers:
                    alloc = min(int(raw.get(li, 0)), int(layer_max[li]))
                    result[li] = max(alloc, 0)
                return result

        # ── uniform distribution (legacy) ────────────────────────────────────
        if self.distribution == "uniform":
            base      = effective_budget // len(layers)
            remainder = effective_budget - base * len(layers)
            raw = {li: base + (1 if i < remainder else 0) for i, li in enumerate(layers)}

        # ── proportional distribution (legacy) ───────────────────────────────
        else:
            out_chs    = {li: int(self.convs[li].out_channels) for li in layers}
            total_ch   = float(sum(out_chs.values()))
            raw        = {}
            allocated  = 0
            for i, li in enumerate(sorted(layers)):
                if i == len(layers) - 1:
                    raw[li] = effective_budget - allocated
                else:
                    share  = int(round(effective_budget * out_chs[li] / total_ch))
                    raw[li] = share
                    allocated += share

        result = {}
        for li in layers:
            alloc = min(int(raw.get(li, 0)), int(layer_max[li]))
            result[li] = max(alloc, 0)
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
            snapped_total = self._snap_total(int(prune_channels))
            layers        = self._eligible_layer_idxs()
            layer_budgets = self._distribute_budget(sorted(layers), snapped_total)

            for li, n in layer_budgets.items():
                chosen += self._select_from_layer(li, n)

        # Dedup
        seen  = set()
        dedup: List[Tuple[float, int, int]] = []
        for s, l, c in chosen:
            key = (int(l), int(c))
            if key in seen:
                continue
            seen.add(key)
            dedup.append((float(s), int(l), int(c)))

        self._last_selected       = dedup
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
            name   = self.conv_names[li] if li < len(self.conv_names) else f"layer{li}"
            conv   = self.convs[li] if li < len(self.convs) else None
            out_ch = int(conv.out_channels) if conv is not None else -1
            g      = self._group_of_layer(li)
            scores = self._l2_scores[li]
            share  = _COST_SHARE_TABLE.get(name, 0.0)

            mn  = float(scores.min().item())
            md  = float(scores.median().item())
            mx  = float(scores.max().item())
            std = float(scores.std().item()) if scores.numel() > 1 else 0.0

            log.info(
                f"  - {name:44s} | group={g:10s} | out_ch={out_ch:4d} | "
                f"cost_share={share:6.3f}% | "
                f"L2(min/med/max/std)={mn:.4f}/{md:.4f}/{mx:.4f}/{std:.4f}"
            )
        log.info("=" * 92)

    def log_selection_distribution(self, log, selected: List[Tuple[float, int, int]]) -> None:
        by_conv:  Dict[str, int] = {}
        by_group: Dict[str, int] = {"encoder": 0, "bottleneck": 0, "decoder": 0, "other": 0}

        for _, li, _ in selected:
            name = self.conv_names[li] if li < len(self.conv_names) else f"layer{li}"
            by_conv[name]  = by_conv.get(name, 0) + 1
            g              = self._group_of_layer(li)
            by_group[g]    = by_group.get(g, 0) + 1

        log.info("=" * 92)
        log.info("SELECTION DISTRIBUTION")
        log.info("=" * 92)
        log.info(f"Group counts: {json.dumps(by_group, sort_keys=True)}")

        top = sorted(by_conv.items(), key=lambda x: x[1], reverse=True)[:25]
        log.info("Top convs by selected channels:")
        for n, c in top:
            cost = _COST_SHARE_TABLE.get(n, 0.0)
            log.info(f"  - {n:44s}: {c:4d} channels  (cost_share={cost:.3f}%)")
        log.info("=" * 92)
