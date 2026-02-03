# pruning/pruners/safe.py
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .pruning_budgets import GroupBudget
from .base_pruner import Pruner


class SafeBudgetPruner(Pruner):
    """Adds scope filtering + rank transforms + debug logging (logic preserved)."""

    def __init__(
        self,
        net: nn.Module,
        flops_reg: float,
        scope: str = "global",
        include_inc: bool = True,
        budget: Optional[GroupBudget] = None,
        rank_transform: str = "abs",  # "abs" | "square" | "none"
        rank_overfetch: int = 5,
        max_candidates_per_layer: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(net, flops_reg, **kwargs)
        self.scope = str(scope).lower().strip()
        self.include_inc = bool(include_inc)
        self.budget = budget
        self.rank_transform = str(rank_transform).lower().strip()
        self.rank_overfetch = max(1, int(rank_overfetch))
        self.max_candidates_per_layer = max_candidates_per_layer
        self._last_selected: List[Tuple[float, int, int]] = []

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

    def _apply_rank_transform(self, t: torch.Tensor) -> torch.Tensor:
        if self.rank_transform == "abs":
            return t.abs()
        if self.rank_transform == "square":
            return t.pow(2)
        return t

    def _norm_flops(self) -> List[float]:
        if not self.flops:
            return []
        s = float(sum(self.flops))
        if s == 0.0:
            s = 1.0
        return [float(x) / s for x in self.flops]

    def _build_rank_items(self) -> List[Tuple[float, int, int]]:
        items: List[Tuple[float, int, int]] = []
        flops_norm = self._norm_flops()

        for layer_idx, ranks in self.ranks.items():
            name = self.conv_names[layer_idx] if layer_idx < len(self.conv_names) else ""
            if name.startswith("outc."):
                continue

            taylor = ranks.detach()
            if taylor.is_cuda:
                taylor = taylor.cpu()

            taylor = self._apply_rank_transform(taylor)
            denom = torch.sqrt(torch.sum(taylor ** 2)) + 1e-12
            taylor = taylor / denom

            flops_pen = float(flops_norm[layer_idx]) * float(self.flops_reg) if layer_idx < len(flops_norm) else 0.0

            for ch in range(int(taylor.numel())):
                score = float(taylor[ch].item()) + float(flops_pen)
                items.append((score, int(layer_idx), int(ch)))

        return items

    def _filter_items_by_scope(self, items: List[Tuple[float, int, int]]) -> List[Tuple[float, int, int]]:
        if self.scope == "global":
            return items

        allowed_groups = {"encoder", "bottleneck", "decoder"}
        kept = [(s, l, c) for (s, l, c) in items if self._group_of_layer(l) in allowed_groups]

        if self.scope in ("encoder", "bottleneck", "decoder"):
            kept = [(s, l, c) for (s, l, c) in kept if self._group_of_layer(l) == self.scope]
        # custom/pooled_down handled later
        return kept

    @staticmethod
    def _select_k_lowest(items: List[Tuple[float, int, int]], k: int) -> List[Tuple[float, int, int]]:
        if k <= 0 or not items:
            return []
        k = min(int(k), len(items))
        return sorted(items, key=lambda x: x[0])[:k]

    def log_rank_stats(self, log, max_layers: int = 200) -> None:
        log.info("=" * 92)
        log.info("RANK STATS (per layer)")
        log.info("=" * 92)

        layer_idxs = sorted(self.ranks.keys())[:max_layers]
        flops_norm = self._norm_flops()

        for li in layer_idxs:
            name = self.conv_names[li] if li < len(self.conv_names) else f"layer{li}"
            if name.startswith("outc."):
                continue

            conv = self.convs[li] if li < len(self.convs) else None
            out_ch = int(conv.out_channels) if conv is not None else -1
            g = self._group_of_layer(li)

            r = self.ranks[li].detach()
            r_cpu = r.cpu() if r.is_cuda else r
            neg_frac = float((r_cpu < 0).float().mean().item()) if r_cpu.numel() > 0 else 0.0

            rt = self._apply_rank_transform(r_cpu)
            mn = float(rt.min().item()) if rt.numel() > 0 else 0.0
            md = float(rt.median().item()) if rt.numel() > 0 else 0.0
            mx = float(rt.max().item()) if rt.numel() > 0 else 0.0
            fp = float(flops_norm[li]) if li < len(flops_norm) else 0.0

            log.info(
                f"  - {name:40s} | group={g:10s} | out_ch={out_ch:4d} | "
                f"neg%={neg_frac*100:6.2f} | rank(min/med/max)={mn:.3e}/{md:.3e}/{mx:.3e} | flops_w={fp:.4f}"
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

    def preview_selection(self, requested_prune: int):
        return self._rank_channels(requested_prune)

    def _rank_channels(self, prune_channels: int):
        items_all = self._build_rank_items()
        items = self._filter_items_by_scope(items_all)

        if not items:
            self._last_selected = []
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        of = int(self.rank_overfetch)

        # -------- Budget scopes --------
        if self.scope in ("custom", "pooled_down"):
            if self.budget is None:
                raise ValueError("Budget scope selected but budget is None.")
            b = self.budget
            chosen: List[Tuple[float, int, int]] = []

            def _k(req: int, pool_len: int) -> int:
                if req <= 0 or pool_len <= 0:
                    return 0
                return min(pool_len, int(req) * of)

            if b.prune_down_total is not None:
                down_items = [(s, l, c) for (s, l, c) in items if self._is_down_layer(l)]
                dec_items = [(s, l, c) for (s, l, c) in items if self._group_of_layer(l) == "decoder"]
                chosen += self._select_k_lowest(down_items, _k(int(b.prune_down_total), len(down_items)))
                chosen += self._select_k_lowest(dec_items, _k(int(b.prune_decoder), len(dec_items)))
            else:
                enc_items = [(s, l, c) for (s, l, c) in items if self._group_of_layer(l) == "encoder"]
                bot_items = [(s, l, c) for (s, l, c) in items if self._group_of_layer(l) == "bottleneck"]
                dec_items = [(s, l, c) for (s, l, c) in items if self._group_of_layer(l) == "decoder"]
                chosen += self._select_k_lowest(enc_items, _k(int(b.prune_encoder), len(enc_items)))
                chosen += self._select_k_lowest(bot_items, _k(int(b.prune_bottleneck), len(bot_items)))
                chosen += self._select_k_lowest(dec_items, _k(int(b.prune_decoder), len(dec_items)))

            chosen = sorted(chosen, key=lambda x: x[0])

        else:
            # -------- Global / single-part scopes --------
            k = int(prune_channels) * of
            items_sorted = sorted(items, key=lambda x: x[0])

            if self.max_candidates_per_layer is None:
                chosen = items_sorted[: min(k, len(items_sorted))]
            else:
                cap = int(self.max_candidates_per_layer)
                per_layer = {}
                chosen = []
                for it in items_sorted:
                    _, li, _ = it
                    per_layer[li] = per_layer.get(li, 0)
                    if per_layer[li] >= cap:
                        continue
                    chosen.append(it)
                    per_layer[li] += 1
                    if len(chosen) >= min(k, len(items_sorted)):
                        break

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
        layer_tensor = torch.tensor([l for (_, l, _) in dedup], dtype=torch.long, device="cpu")
        chan_tensor = torch.tensor([c for (_, _, c) in dedup], dtype=torch.long, device="cpu")
        return layer_tensor, chan_tensor
