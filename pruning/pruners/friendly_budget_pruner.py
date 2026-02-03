# pruning/pruners/friendly.py
from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from common.measurements import ceil_to_multiple, floor_to_multiple
from .safe_budget_pruner import SafeBudgetPruner


class FriendlyBudgetPruner(SafeBudgetPruner):
    """Friendly selection: prune in blocks of M channels per conv (logic preserved)."""

    def __init__(
        self,
        net: nn.Module,
        flops_reg: float,
        friendly_enabled: bool = True,
        friendly_multiple: int = 8,
        friendly_round: str = "up",  # "up" -> prune at least requested; "down" -> prune at most requested
        **kwargs,
    ):
        super().__init__(net, flops_reg, **kwargs)
        self.friendly_enabled = bool(friendly_enabled)
        self.friendly_multiple = int(max(1, friendly_multiple))
        self.friendly_round = str(friendly_round).lower().strip()
        self._effective_prune_last: Optional[int] = None

    def _layer_sorted_candidates(self) -> Dict[int, List[Tuple[float, int]]]:
        flops_norm = self._norm_flops()
        out: Dict[int, List[Tuple[float, int]]] = {}

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

            scores = (taylor + float(flops_pen)).tolist()
            pairs = [(float(scores[i]), int(i)) for i in range(len(scores))]
            pairs.sort(key=lambda x: x[0])
            out[int(layer_idx)] = pairs

        return out

    def _eligible_layers_for_scope(self, layer_candidates: Dict[int, List[Tuple[float, int]]]) -> List[int]:
        if self.scope == "global":
            return sorted(layer_candidates.keys())

        allowed_groups = {"encoder", "bottleneck", "decoder"}
        layers = [li for li in layer_candidates.keys() if self._group_of_layer(li) in allowed_groups]

        if self.scope in ("encoder", "bottleneck", "decoder"):
            layers = [li for li in layers if self._group_of_layer(li) == self.scope]
        return sorted(layers)

    def _layer_max_prune(self, layer_idx: int) -> int:
        conv = self.convs[layer_idx] if layer_idx < len(self.convs) else None
        if conv is None:
            return 0

        out_ch = int(conv.out_channels)
        name = self.conv_names[layer_idx] if layer_idx < len(self.conv_names) else ""
        if name.startswith("outc."):
            return 0

        min_keep_ratio = float(getattr(self, "min_keep_ratio", 0.0))
        min_keep_ch = int(getattr(self, "min_keep_ch", 0))

        min_keep_by_ratio = int(math.ceil(min_keep_ratio * out_ch)) if min_keep_ratio > 0 else 0
        min_keep = max(int(min_keep_ch), int(min_keep_by_ratio))
        min_keep = min(min_keep, out_ch)
        max_prune = max(0, out_ch - min_keep)

        if self.friendly_enabled and self.friendly_multiple > 1 and out_ch >= self.friendly_multiple:
            max_prune = floor_to_multiple(max_prune, self.friendly_multiple)

        return int(max_prune)

    @staticmethod
    def _select_blocks_greedy(
        layer_candidates: Dict[int, List[Tuple[float, int]]],
        layer_max_prune: Dict[int, int],
        target_prune: int,
        block: int,
    ) -> List[Tuple[float, int, int]]:
        if target_prune <= 0 or block <= 0:
            return []

        per_layer = {}
        for li, pairs in layer_candidates.items():
            cap = int(layer_max_prune.get(li, 0))
            if cap <= 0:
                continue
            cap = min(cap, len(pairs))
            if cap <= 0:
                continue

            use = pairs[:cap]
            scores = [p[0] for p in use]
            chans = [p[1] for p in use]

            n_full = cap // block
            if n_full <= 0:
                continue

            ps = [0.0]
            for s in scores[: n_full * block]:
                ps.append(ps[-1] + float(s))

            per_layer[li] = {"scores": scores, "chans": chans, "n_full_blocks": n_full, "ps": ps, "bidx": 0}

        if not per_layer:
            return []

        def block_cost(li: int, bidx: int) -> float:
            st = bidx * block
            en = (bidx + 1) * block
            ps = per_layer[li]["ps"]
            return float(ps[en] - ps[st])

        heap = []
        for li in per_layer.keys():
            heapq.heappush(heap, (block_cost(li, 0), li))

        selected: List[Tuple[float, int, int]] = []
        need = int(target_prune)

        while heap and len(selected) < need:
            _, li = heapq.heappop(heap)
            d = per_layer[li]
            bidx = int(d["bidx"])
            if bidx >= int(d["n_full_blocks"]):
                continue

            st = bidx * block
            en = (bidx + 1) * block

            for j in range(st, en):
                selected.append((float(d["scores"][j]), int(li), int(d["chans"][j])))
                if len(selected) >= need:
                    break

            d["bidx"] = bidx + 1
            if d["bidx"] < int(d["n_full_blocks"]) and len(selected) < need:
                heapq.heappush(heap, (block_cost(li, d["bidx"]), li))

        return selected

    def _effective_target(self, requested: int) -> int:
        if not self.friendly_enabled or self.friendly_multiple <= 1:
            return int(requested)

        if self.friendly_round == "down":
            return floor_to_multiple(int(requested), int(self.friendly_multiple))
        return ceil_to_multiple(int(requested), int(self.friendly_multiple))

    def _rank_channels(self, prune_channels: int):
        if not self.friendly_enabled or self.friendly_multiple <= 1:
            return super()._rank_channels(prune_channels)

        M = int(self.friendly_multiple)
        requested = int(prune_channels)
        target = self._effective_target(requested)
        self._effective_prune_last = int(target)

        layer_candidates_all = self._layer_sorted_candidates()

        def build_for_layers(layers: List[int], tgt: int) -> List[Tuple[float, int, int]]:
            lc = {li: layer_candidates_all[li] for li in layers if li in layer_candidates_all}
            lmax = {li: self._layer_max_prune(li) for li in layers}

            cap_total = int(sum(lmax.values()))
            if cap_total <= 0:
                return []

            tgt_eff = min(int(tgt), int(cap_total))
            tgt_eff = floor_to_multiple(tgt_eff, M)
            return self._select_blocks_greedy(lc, lmax, tgt_eff, M)

        chosen: List[Tuple[float, int, int]] = []

        if self.scope in ("custom", "pooled_down"):
            if self.budget is None:
                raise ValueError("Budget scope selected but budget is None.")
            b = self.budget

            if b.prune_down_total is not None:
                down_layers = [li for li in layer_candidates_all.keys() if self._is_down_layer(li)]
                dec_layers = [li for li in layer_candidates_all.keys() if self._group_of_layer(li) == "decoder"]

                chosen += build_for_layers(sorted(down_layers), self._effective_target(int(b.prune_down_total)))
                chosen += build_for_layers(sorted(dec_layers), self._effective_target(int(b.prune_decoder)))
            else:
                enc_layers = [li for li in layer_candidates_all.keys() if self._group_of_layer(li) == "encoder"]
                bot_layers = [li for li in layer_candidates_all.keys() if self._group_of_layer(li) == "bottleneck"]
                dec_layers = [li for li in layer_candidates_all.keys() if self._group_of_layer(li) == "decoder"]

                chosen += build_for_layers(sorted(enc_layers), self._effective_target(int(b.prune_encoder)))
                chosen += build_for_layers(sorted(bot_layers), self._effective_target(int(b.prune_bottleneck)))
                chosen += build_for_layers(sorted(dec_layers), self._effective_target(int(b.prune_decoder)))
        else:
            layers = self._eligible_layers_for_scope(layer_candidates_all)
            chosen = build_for_layers(layers, target)

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
