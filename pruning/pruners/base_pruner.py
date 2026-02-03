# pruning/pruners/core.py
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class Pruner:
    """Base Taylor channel pruner with your original dependency logic preserved.

    Notes:
    - Hook order assumes the same UNet module naming as your current codebase.
    - This class prunes Conv2d + BN + ConvTranspose2d (decoder upsamplers).
    - Selection strategy is implemented via `_rank_channels()` (overridden by subclasses).
    """

    def __init__(self, net: nn.Module, flops_reg: float, min_keep_ch: int = 1, min_keep_ratio: float = 0.0):
        self.net = net.eval()
        self.flops_reg = float(flops_reg)

        self.min_keep_ch = int(min_keep_ch)
        self.min_keep_ratio = float(min_keep_ratio)

        self.clear_rank()
        self.clear_modules()
        self.clear_cache()
        self._register_hooks()

    # ----------------------------
    # State
    # ----------------------------
    def clear_rank(self) -> None:
        self.ranks: Dict[int, torch.Tensor] = {}
        self.flops: List[float] = []

    def clear_modules(self) -> None:
        self.convs: List[nn.Conv2d] = []
        self.conv_names: List[str] = []
        self.BNs: List[nn.BatchNorm2d] = []
        self.deconvs: Dict[str, nn.ConvTranspose2d] = {}  # e.g. {"up1.up": module, ...}

    def clear_cache(self) -> None:
        self.activation_maps: List[torch.Tensor] = []
        self.gradients: List[torch.Tensor] = []

    # ----------------------------
    # Hooks + Taylor ranks
    # ----------------------------
    def _register_hooks(self) -> None:
        def forward_hook_fn(_module, _inputs, output):
            self.activation_maps.append(output.detach())

        def backward_hook_fn(_module, _grad_input, grad_output):
            self.gradients.append(grad_output[0].detach())

        for name, module in self.net.named_modules():
            if isinstance(module, nn.Conv2d):
                if name != "outc.conv":
                    module.register_forward_hook(forward_hook_fn)
                    module.register_full_backward_hook(backward_hook_fn)
                self.convs.append(module)
                self.conv_names.append(name)

            elif isinstance(module, nn.ConvTranspose2d):
                self.deconvs[name] = module

            elif isinstance(module, nn.BatchNorm2d):
                self.BNs.append(module)

    def compute_rank(self) -> None:
        # Your original ordering behavior
        self.gradients.reverse()

        for layer, act in enumerate(self.activation_maps):
            taylor = (act * self.gradients[layer]).mean(dim=(2, 3)).abs().mean(dim=0)  # [C]
            if layer not in self.ranks:
                self.ranks[layer] = taylor
            else:
                self.ranks[layer] = 0.9 * self.ranks[layer] + 0.1 * taylor

        self.clear_cache()

    # ----------------------------
    # Ranking (base: original topk implementation)
    # Subclasses override this.
    # ----------------------------
    def _rank_channels(self, prune_channels: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (layer_idxs, channel_idxs), both CPU LongTensor of length <= prune_channels."""
        total_rank_list: List[torch.Tensor] = []
        channel_layers_list: List[int] = []
        layer_channels_list: List[int] = []

        s = float(sum(self.flops)) if self.flops else 1.0
        if s == 0.0:
            s = 1.0
        self.flops[:] = [float(x) / s for x in self.flops]

        for layer, ranks in self.ranks.items():
            taylor = ranks.detach()
            if taylor.is_cuda:
                taylor = taylor.cpu()

            denom = torch.sqrt(torch.sum(taylor ** 2)) + 1e-12
            taylor = taylor / denom

            flops_pen = float(self.flops[layer]) * float(self.flops_reg)
            total_rank_list.append(taylor + flops_pen)

            c = int(taylor.shape[0])
            channel_layers_list.extend([int(layer)] * c)
            layer_channels_list.extend(range(c))

        if not total_rank_list:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        channel_layers = torch.tensor(channel_layers_list, dtype=torch.long, device="cpu")
        layer_channels = torch.tensor(layer_channels_list, dtype=torch.long, device="cpu")
        total_rank = torch.cat(total_rank_list, dim=0).to(device="cpu")

        k = min(int(prune_channels), int(total_rank.numel()))
        _, sorted_indices = torch.topk(total_rank, k, largest=False)
        sorted_indices = sorted_indices.to(device="cpu")

        return channel_layers[sorted_indices], layer_channels[sorted_indices]

    # ----------------------------
    # Indices bookkeeping
    # ----------------------------
    def create_indices(self):
        chans = [(list(range(c.weight.shape[1])), list(range(c.weight.shape[0]))) for c in self.convs]
        inchans, outchans = list(zip(*chans))
        return list(inchans), list(outchans)

    def _name_to_conv_idx(self) -> Dict[str, int]:
        return {n: i for i, n in enumerate(self.conv_names)}

    @torch.no_grad()
    def _prune_deconv(self, deconv: nn.ConvTranspose2d, in_keep: list, out_keep: list) -> None:
        # ConvTranspose2d weight: [in_channels, out_channels, kH, kW]
        w = deconv.weight.data
        deconv.weight.data = w[in_keep, ...][:, out_keep, ...]
        if deconv.bias is not None:
            deconv.bias.data = deconv.bias.data[out_keep]

    def _min_keep_for_layer(self, layer_idx: int, outchans: list) -> int:
        cur = len(outchans[layer_idx])
        ratio_min = int(math.ceil(cur * self.min_keep_ratio)) if self.min_keep_ratio > 0 else 0
        return max(self.min_keep_ch, ratio_min, 1)

    # ----------------------------
    # Apply pruning (logic unchanged)
    # ----------------------------
    def pruning(self, prune_channels: int) -> None:
        sorted_channel_layers, sorted_layer_channels = self._rank_channels(prune_channels)
        inchans, outchans = self.create_indices()

        requested = int(prune_channels)
        applied = 0
        skipped_minkeep = 0
        skipped_missing = 0

        # ---- your original dependency logic (unchanged) ----
        for i in range(len(sorted_channel_layers)):
            if applied >= requested:
                break

            cl = int(sorted_channel_layers[i])
            lc = int(sorted_layer_channels[i])

            if lc not in outchans[cl]:
                skipped_missing += 1
                continue

            min_keep = self._min_keep_for_layer(cl, outchans)
            if len(outchans[cl]) <= min_keep:
                skipped_minkeep += 1
                continue

            res = True if cl in [1, 3, 5, 7] else False
            offset = True if cl in [9, 11, 13, 15] else False

            if offset:
                mapping = {9: 7, 11: 5, 13: 3, 15: 1}
                top = self.convs[mapping[cl]].weight.shape[0]
                try:
                    inchans[cl + 1].remove(top + lc)
                except ValueError:
                    pass
            else:
                try:
                    inchans[cl + 1].remove(lc)
                except ValueError:
                    pass

            if res:
                try:
                    inchans[-(cl + 2)].remove(lc)
                except ValueError:
                    pass

            try:
                outchans[cl].remove(lc)
                applied += 1
            except ValueError:
                skipped_missing += 1
                continue

        if applied < requested:
            print(
                f"[Pruner] Requested prune={requested}, applied={applied}. "
                f"Skipped(min_keep)={skipped_minkeep}, skipped(already_removed)={skipped_missing}. "
                f"min_keep_ch={self.min_keep_ch}, min_keep_ratio={self.min_keep_ratio}"
            )

        # -----------------------------------------------------------------------------
        # FORCE DECODER CONCAT CONSISTENCY (unchanged)
        # -----------------------------------------------------------------------------
        name_to_idx = self._name_to_conv_idx()

        def _force_up_conv0_in(up_conv0_name: str, skip_out_name: str):
            idx_conv0 = name_to_idx[up_conv0_name]
            idx_skip = name_to_idx[skip_out_name]

            keep_skip = outchans[idx_skip]
            half = self.convs[idx_conv0].weight.shape[1] // 2
            inchans[idx_conv0] = keep_skip + [half + k for k in keep_skip]

        _force_up_conv0_in("up1.conv.double_conv.0", "down3.maxpool_conv.1.double_conv.3")
        _force_up_conv0_in("up2.conv.double_conv.0", "down2.maxpool_conv.1.double_conv.3")
        _force_up_conv0_in("up3.conv.double_conv.0", "down1.maxpool_conv.1.double_conv.3")
        _force_up_conv0_in("up4.conv.double_conv.0", "inc.double_conv.3")

        for li in range(len(outchans)):
            if len(outchans[li]) < 1:
                raise RuntimeError(f"[Pruner] Invalid pruning result: layer {li} ({self.conv_names[li]}) has 0 out channels.")

        # ---- prune Conv2d params ----
        for i, c in enumerate(self.convs):
            self.convs[i].weight.data = c.weight.data[outchans[i], ...][:, inchans[i], ...]
            if c.bias is not None:
                self.convs[i].bias.data = c.bias.data[outchans[i]]

        # ---- prune BN params ----
        for i, bn in enumerate(self.BNs):
            self.BNs[i].running_mean.data = bn.running_mean.data[outchans[i]]
            self.BNs[i].running_var.data = bn.running_var.data[outchans[i]]
            self.BNs[i].weight.data = bn.weight.data[outchans[i]]
            self.BNs[i].bias.data = bn.bias.data[outchans[i]]

        # ---- prune ConvTranspose2d (decoder upsamplers) ----
        name_to_idx = self._name_to_conv_idx()

        idx_x1 = name_to_idx["inc.double_conv.3"]
        idx_x2 = name_to_idx["down1.maxpool_conv.1.double_conv.3"]
        idx_x3 = name_to_idx["down2.maxpool_conv.1.double_conv.3"]
        idx_x4 = name_to_idx["down3.maxpool_conv.1.double_conv.3"]
        idx_x5 = name_to_idx["down4.maxpool_conv.1.double_conv.3"]

        idx_u1 = name_to_idx["up1.conv.double_conv.3"]
        idx_u2 = name_to_idx["up2.conv.double_conv.3"]
        idx_u3 = name_to_idx["up3.conv.double_conv.3"]

        if "up1.up" in self.deconvs:
            self._prune_deconv(self.deconvs["up1.up"], in_keep=outchans[idx_x5], out_keep=outchans[idx_x4])
        if "up2.up" in self.deconvs:
            self._prune_deconv(self.deconvs["up2.up"], in_keep=outchans[idx_u1], out_keep=outchans[idx_x3])
        if "up3.up" in self.deconvs:
            self._prune_deconv(self.deconvs["up3.up"], in_keep=outchans[idx_u2], out_keep=outchans[idx_x2])
        if "up4.up" in self.deconvs:
            self._prune_deconv(self.deconvs["up4.up"], in_keep=outchans[idx_u3], out_keep=outchans[idx_x1])

    def channel_save(self, path: str) -> None:
        n = self.net
        chans = []

        chans.append(n.inc.double_conv[0].weight.shape[0])  # mid
        chans.append(n.inc.double_conv[3].weight.shape[0])  # out

        for d in [n.down1, n.down2, n.down3, n.down4]:
            c0 = d.maxpool_conv[1].double_conv[0]
            c1 = d.maxpool_conv[1].double_conv[3]
            chans.append(c0.weight.shape[0])
            chans.append(c1.weight.shape[0])

        for u in [n.up1, n.up2, n.up3, n.up4]:
            chans.append(u.up.weight.shape[0])  # deep_in
            chans.append(u.conv.double_conv[0].weight.shape[0])  # mid
            chans.append(u.conv.double_conv[3].weight.shape[0])  # out

        assert len(chans) == 22, f"Expected 22 integers, got {len(chans)}: {chans}"

        with open(path, "w") as f:
            for c in chans:
                f.write(f"{int(c)}\n")

    def calc_flops(self) -> List[float]:
        """Same convention as before: H*W*KH*KW*Cin (no *Cout), excluding outc."""
        self.flops = []
        for i, c in enumerate(self.convs[:-1]):
            H, W = self.gradients[i].shape[2:]
            _, I, KH, KW = c.weight.shape
            self.flops.append(float(H * W * KH * KW * I))
        return self.flops
