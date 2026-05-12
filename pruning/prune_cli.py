from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Identity / paths
    p.add_argument("-n", "--name", default="final", help="run name (folder name)")
    p.add_argument("--save_root", default="cost_profiling_based_pruning_taylor", help="root output folder")
    p.add_argument("--data_root", default="data", help="folder containing train/ and train_masks/")
    p.add_argument("--splitfile", default="pruning/data/trainval.json", help="train/val split json path")
    p.add_argument("-l", "--load", default="weights/unet_carvana_scale0.5_epoch2.pth", help="checkpoint path (state_dict)")
    p.add_argument("-c", "--channel_txt", default="configs/model_channels_baseline.txt", help="channel txt (baseline channels)")

    # Device
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="training/pruning device")
    p.add_argument("-g", "--gpu", action="store_true", help="alias for --device cuda")

    # Data / eval
    p.add_argument("-s", "--scale", type=float, default=0.5, help="downscale factor")
    p.add_argument("-b", "--batch_size", type=int, default=2, help="batch size")
    p.add_argument("--quiet_split", type=int, default=0, help="1 to silence split logging")

    # Taylor criterion
    p.add_argument("-t", "--taylor_batches", type=int, default=500, help="mini-batches for Taylor criterion")
    p.add_argument("--taylor_shuffle", type=int, default=0, help="shuffle in Taylor batches (0/1)")
    p.add_argument("--freeze_bn_taylor", type=int, default=1, help="1 freeze BN during Taylor, 0 keep BN train")
    p.add_argument("--rank_transform", default="abs", help="abs|square|none (default abs)")
    p.add_argument("--rank_overfetch", type=int, default=5, help="overfetch factor (compat)")
    p.add_argument("--max_candidates_per_layer", type=int, default=0, help="compat; 0 disables")

    # Pruning scope / budgets
    p.add_argument("--scope", default="global", help="global|encoder|bottleneck|decoder|custom|pooled_down|custom_seq")
    p.add_argument("-p", "--prune_channels", type=int, default=300, help="channels to remove (global/single-part)")

    # for scope=custom
    p.add_argument("--prune_encoder", type=int, default=0, help="custom: encoder budget")
    p.add_argument("--prune_bottleneck", type=int, default=0, help="custom: bottleneck budget")
    p.add_argument("--prune_decoder", type=int, default=0, help="custom: decoder budget")

    # for scope=pooled_down
    p.add_argument("--prune_down_total", type=int, default=-1, help="pooled_down: pooled encoder+bottleneck budget (>=0 enables)")

    # for scope=custom_seq (pct args)
    p.add_argument("--prune_encoder_pct", type=float, default=0.0, help="custom_seq: encoder share (e.g., 20 for 20%)")
    p.add_argument("--prune_bottleneck_pct", type=float, default=0.0, help="custom_seq: bottleneck share (e.g., 60 for 60%)")
    p.add_argument("--prune_decoder_pct", type=float, default=0.0, help="custom_seq: decoder share (e.g., 20 for 20%)")

    # More pruning controls
    p.add_argument("--include_inc", type=int, default=1, help="1 include inc.* in encoder selection, 0 exclude inc.*")
    p.add_argument("--min_keep_ch", type=int, default=1, help="minimum out_channels to keep per conv")
    p.add_argument("--min_keep_ratio", type=float, default=0.0, help="minimum keep ratio per conv (0 disables)")

    # Friendly pruning controls
    p.add_argument("--friendly", type=int, default=1, help="1 enable friendly pruning, 0 disable")
    p.add_argument("--friendly_round", default="up", choices=["up", "down"], help="round requested prune to multiple")
    p.add_argument("--friendly_multiple_cpu", type=int, default=16, help="friendly multiple for CPU runs")
    p.add_argument("--friendly_multiple_gpu", type=int, default=16, help="friendly multiple for CUDA runs")

    # Regularization
    p.add_argument("-f", "--flops", dest="flops_reg", type=float, default=0.001, help="FLOPs regularizer weight")

    # Finetuning
    p.add_argument("-r", "--lr", type=float, default=0.001, help="finetune lr")
    p.add_argument("-i", "--iters", type=int, default=1500, help="finetune iterations")
    p.add_argument("-e", "--epochs", type=int, default=3, help="compat: epochs, 0 uses iters only")

    # Baseline eval
    p.add_argument("--eval_baseline", type=int, default=0, help="1 eval baseline, 0 skip")
    p.add_argument("--baseline_dice", type=float, default=0.989935, help="cached baseline dice if baseline eval is skipped")

    # Diagnostics
    p.add_argument("--print_flops", type=int, default=0, help="1 print per-conv FLOPs tables")

    # Benchmarking
    p.add_argument("--print_table", type=int, default=1, help="print final summaries (1=yes, 0=no)")
    p.add_argument("--bench", action="store_true", help="compute CPU/GPU inference time and report speedup (default off)")
    p.add_argument("--bench_batch", type=int, default=1, help="benchmark batch size")
    p.add_argument("--bench_warmup", type=int, default=5, help="benchmark warmup iters")
    p.add_argument("--bench_runs", type=int, default=50, help="benchmark timed runs")
    p.add_argument("--bench_amp", type=int, default=0, help="1 enable AMP for CUDA benchmark")
    p.add_argument("--bench_channels_last", type=int, default=0, help="1 use channels_last for benchmark")
    p.add_argument("--bench_threads", type=int, default=0, help="CPU threads for benchmark (0 = default)")

    #layer by layer
    #p.add_argument("--l2_distribution", default="proportional", choices=["proportional", "uniform"])
    p.add_argument("--l2_distribution", default="cost_share", choices=["proportional", "uniform", "cost_share"])
    p.add_argument("--pruning_method", default="taylor", choices=["taylor", "l2", "bn_gamma", "fpgm"])
    p.add_argument("--fpgm_dist_metric", default="l2", choices=["l2", "cosine"])

    # Repro
    p.add_argument("--seed", type=int, default=42, help="random seed")

    args = p.parse_args()
    if args.gpu:
        args.device = "cuda"
    return args
