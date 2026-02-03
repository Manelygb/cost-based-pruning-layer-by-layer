"""
prune.py

Entrypoint script for Taylor channel pruning (global / per-part / custom budgets),
with optional friendly alignment and optional CPU/GPU benchmarking.

Keep this file at the same level as train.py / predict.py.
All heavy logic lives in pruning/prune_pipeline.py.
"""
from __future__ import annotations

from pruning.prune_cli import parse_args
from pruning.prune_pipeline import run_pruning


def main() -> None:
    args = parse_args()
    run_pruning(args)


if __name__ == "__main__":
    main()
