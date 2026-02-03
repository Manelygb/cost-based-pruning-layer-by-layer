"""
common/reporting.py

Logging/reporting utilities. This module only formats and logs results;
it does not compute metrics.

It depends on helpers from common.measurements (formatting + ratio helpers).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from common.measurements import (
    _factor,
    _ratio,
    fmt_gflops,
    fmt_int,
    fmt_mflops,
    pct,
)


def _print_flops_per_conv(log, title: str, flops_named: List[Tuple[str, int]]) -> int:
    log.info(title)
    total = 0
    for name, fl in flops_named:
        log.info(f"  - {name:40s}: {fmt_mflops(fl)}")
        total += int(fl)
    log.info(f"  => SUM: {fmt_gflops(total)}")
    return int(total)


def _print_final_summary(log, baseline: Dict, pruned: Dict) -> None:
    log.info("=" * 92)
    log.info("FINAL SUMMARY (Baseline vs Pruned+Finetuned)")
    log.info("=" * 92)
    log.info(f"{'Metric':12s} | {'Baseline':18s} | {'Pruned':18s} | {'Delta':22s} | {'Pruned/Baseline':16s}")
    log.info("-" * 92)

    rows = [
        (
            "Dice",
            f"{baseline['dice']:.6f}",
            f"{pruned['dice']:.6f}",
            f"{(pruned['dice'] - baseline['dice']):+.6f}",
            pct(_ratio(pruned["dice"], baseline["dice"])),
        ),
        (
            "Params",
            fmt_int(baseline["params"]),
            fmt_int(pruned["params"]),
            f"{pct(_ratio(pruned['params'] - baseline['params'], baseline['params']))} ({(pruned['params'] - baseline['params']):+d})",
            pct(_ratio(pruned["params"], baseline["params"])),
        ),
        (
            "Size (MB)",
            f"{baseline['size_mb']:.3f}",
            f"{pruned['size_mb']:.3f}",
            f"{pct(_ratio(pruned['size_mb'] - baseline['size_mb'], baseline['size_mb']))} ({(pruned['size_mb'] - baseline['size_mb']):+.3f})",
            pct(_ratio(pruned["size_mb"], baseline["size_mb"])),
        ),
        (
            "FLOPs",
            f"{baseline['flops_g']:.3f}G",
            f"{pruned['flops_g']:.3f}G",
            f"{pct(_ratio(pruned['flops_g'] - baseline['flops_g'], baseline['flops_g']))} ({(pruned['flops_g'] - baseline['flops_g']):+.3f}G)",
            pct(_ratio(pruned["flops_g"], baseline["flops_g"])),
        ),
    ]

    for r in rows:
        log.info(f"{r[0]:12s} | {r[1]:18s} | {r[2]:18s} | {r[3]:22s} | {r[4]:16s}")

    log.info("-" * 92)
    log.info("Compression factors (Baseline / Pruned):")
    log.info(f"  - Params: {_factor(baseline['params'], pruned['params']):.2f}× smaller")
    log.info(f"  - Size  : {_factor(baseline['size_mb'], pruned['size_mb']):.2f}× smaller")
    log.info(f"  - FLOPs : {_factor(baseline['flops_g'], pruned['flops_g']):.2f}× smaller")
    log.info("=" * 92)


def _print_part_compare(log, title: str, base_parts: Dict, pr_parts: Dict, unit: str) -> None:
    keys = ["inc", "down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4", "outc", "other"]
    log.info("=" * 92)
    log.info(title)
    log.info("=" * 92)
    log.info(f"{'Part':10s} | {'Baseline':20s} | {'Pruned':20s} | {'Delta':20s} | {'Pruned/B':12s}")
    log.info("-" * 92)

    for k in keys:
        b = float(base_parts.get(k, 0.0))
        p = float(pr_parts.get(k, 0.0))
        ratio = (p / b) if b > 0 else float("nan")
        delta = p - b

        if unit == "flops":
            b_s = f"{(b / 1e9):.3f}G"
            p_s = f"{(p / 1e9):.3f}G"
            d_s = f"{(delta / 1e9):+.3f}G"
        else:
            b_s = fmt_int(int(b))
            p_s = fmt_int(int(p))
            d_s = f"{int(delta):+d}"

        log.info(f"{k:10s} | {b_s:20s} | {p_s:20s} | {d_s:20s} | {pct(ratio):12s}")

    log.info("-" * 92)


# Public names (kept to match your original scripts)
def print_flops_per_conv(log, title: str, flops_named: List[Tuple[str, int]]) -> int:
    return _print_flops_per_conv(log, title, flops_named)


def print_final_summary(log, baseline: Dict, pruned: Dict) -> None:
    _print_final_summary(log, baseline, pruned)


def print_part_compare(log, title: str, base_parts: Dict, pr_parts: Dict, unit: str) -> None:
    _print_part_compare(log, title, base_parts, pr_parts, unit)
