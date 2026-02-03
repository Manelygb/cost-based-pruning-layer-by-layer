"""
common/measurements.py

Centralized measurement utilities used across the project.

Scope (no model-specific training logic):
- lightweight formatting helpers (params/FLOPs display)
- parameter counting and model-size estimation
- per-part parameter/FLOPs aggregation (inc/down*/up*/outc/other)
- FLOPs estimation utilities (two conventions kept as-is)
- inference latency benchmarking (CPU/GPU)

Important:
- Logic is kept consistent with your original codebase (same conventions).
- FLOPs convention used in pruning code: H * W * KH * KW * Cin (NO * Cout).
"""

from __future__ import annotations

import io
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


# ----------------------------
# Formatting helpers
# ----------------------------
def fmt_int(n: int) -> str:
    return f"{int(n):,}"


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def fmt_mflops(n: int) -> str:
    return f"{(float(n) / 1e6):.3f} MFLOPs  ({fmt_int(n)})"


def fmt_gflops(n: int) -> str:
    return f"{(float(n) / 1e9):.3f} GFLOPs  ({fmt_int(n)})"


def _ratio(pv: float, bv: float) -> float:
    if bv == 0:
        return float("nan")
    return pv / bv


def _factor(bv: float, pv: float) -> float:
    if pv == 0:
        return float("inf")
    return bv / pv


def ceil_to_multiple(x: int, m: int) -> int:
    if m <= 0:
        return int(x)
    x = int(x)
    return int(((x + m - 1) // m) * m)


def floor_to_multiple(x: int, m: int) -> int:
    if m <= 0:
        return int(x)
    x = int(x)
    return int((x // m) * m)


# ----------------------------
# FLOPs (legacy list) - kept as-is
# Convention: H*W*KH*KW*Cin  (NO *Cout), and skips outc.conv
# ----------------------------
def flops_count(net: nn.Module, size: Tuple[int, int]) -> List[int]:
    """Legacy FLOPs list used by older scripts.

    Notes:
    - Keeps the original "is_warp" toggling logic unchanged.
    - Returns a list of per-conv FLOPs (excluding 'outc.conv').
    """
    H, W = size
    flops: List[int] = []
    is_warp = True

    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d) and name != "outc.conv":
            O, I, KH, KW = module.weight.shape  # noqa: F841  (O kept for readability)
            if is_warp:
                if name[:2] == "up":
                    H *= 2
                    W *= 2
                elif name[:4] == "down":
                    H //= 2
                    W //= 2

            flops.append(int(H * W * KH * KW * I))
            is_warp = not is_warp

    return flops


# ----------------------------
# Params / size
# ----------------------------
def count_params(net: nn.Module) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def size_mb_from_state_dict(sd: dict) -> float:
    """Approximate serialized size (MB) of a state_dict using torch.save."""
    buf = io.BytesIO()
    torch.save(sd, buf)
    return float(buf.getbuffer().nbytes) / (1024.0 * 1024.0)


# ----------------------------
# Per-part aggregation
# ----------------------------
_PART_KEYS: List[str] = ["inc", "down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4", "outc", "other"]


def _part_from_name(name: str) -> str:
    """Map a module name like 'down2.maxpool_conv.1.double_conv.0' -> 'down2'."""
    prefix = name.split(".", 1)[0] if "." in name else name
    return prefix if prefix in _PART_KEYS else "other"


def params_by_part(net: nn.Module) -> Dict[str, int]:
    parts: Dict[str, int] = {k: 0 for k in _PART_KEYS}
    for n, p in net.named_parameters():
        parts[_part_from_name(n)] += int(p.numel())
    return parts


def summarize_flops_by_part(flops_named: List[Tuple[str, int]]) -> Dict[str, int]:
    parts: Dict[str, int] = {k: 0 for k in _PART_KEYS}
    for name, fl in flops_named:
        parts[_part_from_name(name)] += int(fl)
    return parts


def conv_out_channels_by_name(net: nn.Module) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            out[name] = int(m.out_channels)
    return out


# ----------------------------
# FLOPs after pruning (no grads): hook-based
# Convention: H*W*KH*KW*Cin  (NO *Cout)
# ----------------------------
@torch.no_grad()
def conv_flops_named_no_grad(net: nn.Module, x: torch.Tensor) -> List[Tuple[str, int]]:
    """Compute per-conv FLOPs using a forward pass to infer output H,W.

    Returns:
        List of (conv_name, flops) for every Conv2d discovered via named_modules().
    """
    net.eval()

    convs: List[nn.Conv2d] = []
    names: List[str] = []
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            convs.append(m)
            names.append(name)

    out_hw: Dict[str, Tuple[int, int]] = {}

    def _hook(name_: str):
        def fn(_module, _inputs, output):
            if isinstance(output, torch.Tensor) and output.ndim == 4:
                out_hw[name_] = (int(output.shape[2]), int(output.shape[3]))
        return fn

    hooks = [m.register_forward_hook(_hook(name)) for name, m in zip(names, convs)]
    _ = net(x)
    for h in hooks:
        h.remove()

    report: List[Tuple[str, int]] = []
    for name, m in zip(names, convs):
        if name not in out_hw:
            continue
        H, W = out_hw[name]
        _, Cin, KH, KW = m.weight.shape
        fl = int(H * W * KH * KW * Cin)
        report.append((name, fl))
    return report


# ----------------------------
# Inference benchmarking (avg/std in ms)
# ----------------------------
@torch.no_grad()
def benchmark_inference_avg_ms(
    net: nn.Module,
    device: torch.device,
    input_hw: Tuple[int, int],
    batch_size: int = 1,
    warmup: int = 50,
    runs: int = 200,
    use_amp: bool = False,
    channels_last: bool = False,
    threads: int = 0,
) -> Dict[str, float]:
    """Benchmark inference latency.

    Returns:
        {"avg_ms": ..., "std_ms": ..., "runs": ..., "warmup": ...}

    Notes:
    - For CUDA, uses cuda Events (ms).
    - For CPU, uses perf_counter (ms).
    - `threads` applies only for CPU if > 0.
    """
    if device.type == "cpu" and threads and threads > 0:
        try:
            torch.set_num_threads(int(threads))
        except Exception:
            pass

    net = net.to(device)
    net.eval()

    x = torch.zeros(
        (int(batch_size), 3, int(input_hw[0]), int(input_hw[1])),
        device=device,
        dtype=torch.float32,
    )

    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
        try:
            net = net.to(memory_format=torch.channels_last)
        except Exception:
            pass

    # Warmup + timing
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

        for _ in range(int(warmup)):
            with torch.cuda.amp.autocast(enabled=bool(use_amp)):
                _ = net(x)
        torch.cuda.synchronize()

        times: List[float] = []
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        for _ in range(int(runs)):
            starter.record()
            with torch.cuda.amp.autocast(enabled=bool(use_amp)):
                _ = net(x)
            ender.record()
            torch.cuda.synchronize()
            times.append(float(starter.elapsed_time(ender)))  # ms

    else:
        for _ in range(int(warmup)):
            _ = net(x)

        times = []
        for _ in range(int(runs)):
            t0 = time.perf_counter()
            _ = net(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    arr = np.array(times, dtype=np.float64) if len(times) else np.array([float("nan")], dtype=np.float64)
    return {
        "avg_ms": float(np.nanmean(arr)),
        "std_ms": float(np.nanstd(arr)),
        "runs": float(runs),
        "warmup": float(warmup),
    }
