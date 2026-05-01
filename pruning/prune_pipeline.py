from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from .models.unet_flexible import UNet

from .p_utils.load import get_ids, get_imgs_and_masks
from .p_utils.utils import split_train_val, batch
from .p_utils.deps import take_dep_snapshot, dep_pruned_counts

from common.run_logging import get_logger, get_save_dir
from common.reproducibility import seed_all
from common.io import compare_state_dicts_compact, load_state_dict_weights_only
from common.measurements import (
    ceil_to_multiple,
    floor_to_multiple,
    count_params,
    size_mb_from_state_dict,
    params_by_part,
    conv_flops_named_no_grad,
    summarize_flops_by_part,
    benchmark_inference_avg_ms,
)
from common.reporting import (
    print_flops_per_conv,
    print_part_compare,
    print_final_summary,
)

from .finetune import finetune
from .eval_prune import eval_net

from .pruners.taylor_pruning import set_model_for_taylor
from .pruners.custom_sequential_pruning import compute_targets_from_pct
from .pruners.pruning_budgets import resolve_budget_and_requested
from .pruners.friendly_budget_pruner import FriendlyBudgetPruner
from .pruners.l2_layerwise_pruner import L2LayerwisePruner
from .pruners.bn_gamma_layerwise_pruner import BNGammaLayerwisePruner
from .pruners.fpgm_layerwise_pruner import FPGMLayerwisePruner

# Weight-based methods need no data passes for scoring
_WEIGHT_BASED_METHODS = ("l2", "bn_gamma", "fpgm")
_ALL_METHODS = ("taylor",) + _WEIGHT_BASED_METHODS


def _select_device(device_str: str) -> torch.device:
    s = str(device_str).lower().strip()
    if s == "cpu":
        return torch.device("cpu")
    if s == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _log_run_config(log, payload: Dict[str, Any]) -> None:
    log.info("Args: {}".format(json.dumps(payload, indent=2)))


def run_pruning(args) -> None:
    data_root = osp.expanduser(args.data_root)
    dir_img = osp.join(data_root, "train") + "/"
    dir_mask = osp.join(data_root, "train_masks") + "/"
    splitfile = osp.expanduser(args.splitfile)

    if not osp.isdir(dir_img):
        raise FileNotFoundError(f"train/ not found: {dir_img} (use --data_root)")
    if not osp.isdir(dir_mask):
        raise FileNotFoundError(f"train_masks/ not found: {dir_mask} (use --data_root)")

    device = _select_device(args.device)
    seed_all(int(args.seed), use_cuda=(device.type == "cuda"))

    # Pruning method
    pruning_method = str(getattr(args, "pruning_method", "taylor")).lower().strip()
    assert pruning_method in _ALL_METHODS, \
        f"Unknown pruning_method: {pruning_method}. Choose from {_ALL_METHODS}"
    is_weight_based = pruning_method in _WEIGHT_BASED_METHODS

    # Resolve scope + budgets
    prune_down_total = None if int(args.prune_down_total) < 0 else int(args.prune_down_total)
    scope_norm = str(args.scope).lower().strip()

    if scope_norm == "custom_seq":
        budget = None
        requested_prune_user = int(args.prune_channels)
        scope_desc = "CUSTOM_SEQ (total + pct, sequential)"
    else:
        budget, requested_prune_user, scope_desc = resolve_budget_and_requested(
            scope=scope_norm,
            prune_channels=int(args.prune_channels),
            prune_encoder=int(args.prune_encoder),
            prune_bottleneck=int(args.prune_bottleneck),
            prune_decoder=int(args.prune_decoder),
            prune_down_total=prune_down_total,
        )

    # Friendly settings
    friendly_enabled = bool(int(args.friendly) == 1)
    friendly_multiple = int(args.friendly_multiple_gpu if device.type == "cuda" else args.friendly_multiple_cpu)
    friendly_round = str(args.friendly_round).lower().strip()

    # Output folder + logger
    runname = str(args.name)
    save_path = osp.join(str(args.save_root), runname)
    save_dir = get_save_dir(save_path, runname, training=False)
    log = get_logger(save_dir, runname)

    # Effective requested prune (logging only)
    requested_prune_effective = int(requested_prune_user)
    if friendly_enabled and friendly_multiple > 1:
        if friendly_round == "down":
            requested_prune_effective = floor_to_multiple(requested_prune_user, friendly_multiple)
        else:
            requested_prune_effective = ceil_to_multiple(requested_prune_user, friendly_multiple)

    _log_run_config(log, {
        "name": runname,
        "save_dir": save_dir,
        "device": device.type,
        "data_root": data_root,
        "pruning_method": pruning_method,
        "scale": float(args.scale),
        "batch_size": int(args.batch_size),
        "taylor_batches": int(args.taylor_batches),
        "taylor_shuffle": int(args.taylor_shuffle),
        "freeze_bn_taylor": int(args.freeze_bn_taylor),
        "seed": int(args.seed),
        "scope": args.scope,
        "scope_desc": scope_desc,
        "requested_prune_user": int(requested_prune_user),
        "requested_prune_effective": int(requested_prune_effective),
        "budget": None if budget is None else {
            "prune_down_total": budget.prune_down_total,
            "prune_encoder": budget.prune_encoder,
            "prune_bottleneck": budget.prune_bottleneck,
            "prune_decoder": budget.prune_decoder,
            "total": budget.total(),
        },
        "friendly_enabled": int(friendly_enabled),
        "friendly_multiple": int(friendly_multiple),
        "friendly_round": friendly_round,
        "min_keep_ch": int(args.min_keep_ch),
        "min_keep_ratio": float(args.min_keep_ratio),
        "flops_reg": float(args.flops_reg),
        "finetune_iters": int(args.iters),
        "finetune_lr": float(args.lr),
        "eval_baseline": int(args.eval_baseline),
        "bench": bool(args.bench),
    })

    # ----------------------------
    # Train/val split
    # ----------------------------
    if not osp.exists(splitfile):
        os.makedirs(osp.dirname(splitfile), exist_ok=True)
        ids = get_ids(dir_img)
        iddataset = split_train_val(ids, 0.2, splitfile)
        if int(args.quiet_split) != 1:
            log.info(f"New split dataset -> {splitfile}")
    else:
        with open(splitfile) as f:
            iddataset = json.load(f)
        if int(args.quiet_split) != 1:
            log.info(f"Load split dataset -> {splitfile}")

    train_ids = iddataset["train"]
    val_ids = iddataset["val"]

    # ----------------------------
    # Model init + load
    # ----------------------------
    channel_txt = osp.expanduser(args.channel_txt)
    if not osp.exists(channel_txt):
        raise FileNotFoundError(f"channel_txt not found: {channel_txt} (use --channel_txt)")

    net = UNet(n_channels=3, n_classes=2, f_channels=channel_txt).to(device)
    log.info(f"Built model using channel file: {channel_txt}")

    if args.load and osp.exists(args.load):
        ckpt_sd = load_state_dict_weights_only(args.load, map_location=device)
        common, mismatch, missing, unexpected = compare_state_dicts_compact(ckpt_sd, net.state_dict())
        log.info(
            f"Checkpoint key/shape check: common={common}, mismatch={mismatch}, "
            f"missing={missing}, unexpected={unexpected}"
        )
        net.load_state_dict(ckpt_sd, strict=False)
        log.info(f"Loaded checkpoint: {args.load}")
    else:
        log.info(f"Checkpoint not found: {args.load} (continuing)")

    # ----------------------------
    # Baseline metrics
    # ----------------------------
    baseline_params_total = count_params(net)
    baseline_size_mb = size_mb_from_state_dict(net.state_dict())
    baseline_param_parts = params_by_part(net)

    if int(args.eval_baseline) == 1:
        log.info("Evaluating BASELINE model...")
        val_gen = get_imgs_and_masks(val_ids, dir_img, dir_mask, args.scale)
        baseline_dice = eval_net(net, val_gen, len(val_ids), (device.type == "cuda"), args.batch_size)
        seed_all(int(args.seed), use_cuda=(device.type == "cuda"))
        log.info(f"BASELINE Dice: {baseline_dice:.6f}")
    else:
        baseline_dice = float(args.baseline_dice)
        log.info(f"Skipping baseline eval; using cached baseline dice={baseline_dice:.6f}")

    baseline_metrics: Dict[str, Any] = {
        "dice": float(baseline_dice),
        "params": int(baseline_params_total),
        "size_mb": float(baseline_size_mb),
    }

    # ----------------------------
    # Build pruner
    # ----------------------------
    max_cand = int(args.max_candidates_per_layer)
    if max_cand <= 0:
        max_cand = None

    common_pruner_kwargs = dict(
        scope=scope_norm,
        include_inc=bool(int(args.include_inc)),
        budget=budget,
        min_keep_ch=int(args.min_keep_ch),
        min_keep_ratio=float(args.min_keep_ratio),
        friendly_enabled=friendly_enabled,
        friendly_multiple=int(friendly_multiple),
        friendly_round=friendly_round,
    )

    # distribution arg shared by all layerwise methods
    layerwise_distribution = str(getattr(args, "l2_distribution", "proportional"))

    if pruning_method == "l2":
        log.info("Pruning method: L2 LAYERWISE (weight norm, no data passes required)")
        pruner = L2LayerwisePruner(
            net,
            flops_reg=0.0,
            distribution=layerwise_distribution,
            **common_pruner_kwargs,
        )
    elif pruning_method == "bn_gamma":
        log.info("Pruning method: BN GAMMA LAYERWISE (|γ|-based, no data passes required)")
        pruner = BNGammaLayerwisePruner(
            net,
            flops_reg=0.0,
            distribution=layerwise_distribution,
            **common_pruner_kwargs,
        )
    elif pruning_method == "fpgm":
        fpgm_dist_metric = str(getattr(args, "fpgm_dist_metric", "l2"))
        log.info(f"Pruning method: FPGM LAYERWISE (geometric median, dist={fpgm_dist_metric}, no data passes required)")
        pruner = FPGMLayerwisePruner(
            net,
            flops_reg=0.0,
            distribution=layerwise_distribution,
            dist_metric=fpgm_dist_metric,
            **common_pruner_kwargs,
        )
    else:
        log.info("Pruning method: TAYLOR (gradient-based)")
        pruner = FriendlyBudgetPruner(
            net,
            float(args.flops_reg),
            rank_transform=str(args.rank_transform),
            rank_overfetch=int(args.rank_overfetch),
            max_candidates_per_layer=max_cand,
            **common_pruner_kwargs,
        )

    criterion = nn.CrossEntropyLoss()
    taylor_shuffle = bool(int(args.taylor_shuffle))

    baseline_flops_named = None
    baseline_flops_sum = None
    input_hw: Optional[Tuple[int, int]] = None

    # ----------------------------
    # Taylor batch loop (Taylor only)
    # ----------------------------
    def _run_taylor_batches(stage_label: str) -> None:
        nonlocal baseline_flops_named, baseline_flops_sum, input_hw

        if pruning_method != "taylor":
            return

        pruner.clear_cache()
        pruner.clear_rank()
        set_model_for_taylor(net, freeze_bn=bool(int(args.freeze_bn_taylor)))

        train_gen_local = get_imgs_and_masks(train_ids, dir_img, dir_mask, args.scale)

        with tqdm(
            total=int(args.taylor_batches) * int(args.batch_size),
            desc=f"Taylor ({stage_label})"
        ) as progress_bar:
            for i, b0 in enumerate(batch(train_gen_local, args.batch_size, shuffle=taylor_shuffle)):
                net.zero_grad(set_to_none=True)

                imgs_np = np.array([x[0] for x in b0], dtype=np.float32)
                masks_np = np.array([x[1] for x in b0])

                imgs = torch.from_numpy(imgs_np).to(device=device, dtype=torch.float32)

                true_masks = torch.from_numpy(masks_np).to(device=device)
                if true_masks.ndim == 4 and true_masks.shape[1] == 1:
                    true_masks = true_masks[:, 0, :, :]
                true_masks = (true_masks > 0).long() if true_masks.max() > 1 else true_masks.long()

                if input_hw is None:
                    input_hw = (int(imgs.shape[2]), int(imgs.shape[3]))

                out = net(imgs)
                loss = criterion(out, true_masks)
                loss.backward()

                if i == 0:
                    flops_list = pruner.calc_flops()
                    if baseline_flops_named is None:
                        names = pruner.conv_names[:-1]
                        baseline_flops_named = list(zip(names, flops_list))
                        baseline_flops_sum = int(sum(flops_list))
                        if int(args.print_flops) == 1:
                            print_flops_per_conv(log, "FLOPs before pruning (per conv):", baseline_flops_named)

                pruner.compute_rank()
                progress_bar.update(len(b0))

                if (i + 1) >= int(args.taylor_batches):
                    log.info(f"Finished Taylor criterion computation ({stage_label})")
                    break

    # For weight-based methods: infer input_hw + baseline FLOPs from dummy forward
    def _infer_input_hw_for_weight_based() -> None:
        nonlocal input_hw, baseline_flops_named, baseline_flops_sum
        if input_hw is not None:
            return
        gen = get_imgs_and_masks(train_ids[:1], dir_img, dir_mask, args.scale)
        for img, _ in gen:
            h, w = int(img.shape[1]), int(img.shape[2])
            input_hw = (h, w)
            break
        dummy = torch.zeros((1, 3, input_hw[0], input_hw[1]), device=device, dtype=torch.float32)
        named = conv_flops_named_no_grad(net, dummy)
        named_no_outc = [(n, f) for (n, f) in named if n != "outc.conv"]
        baseline_flops_named = named_no_outc
        baseline_flops_sum = int(sum(f for _, f in named_no_outc))
        if int(args.print_flops) == 1:
            print_flops_per_conv(log, "FLOPs before pruning (per conv):", baseline_flops_named)

    def _compute_weight_scores(stage_label: str = "") -> None:
        """Compute scores for the active weight-based method."""
        suffix = f" ({stage_label})" if stage_label else ""
        if pruning_method == "l2":
            log.info(f"Computing L2 scores{suffix} from model weights...")
            pruner.compute_l2_scores()
        elif pruning_method == "bn_gamma":
            log.info(f"Computing BN gamma scores{suffix} from BN parameters...")
            pruner.compute_gamma_scores()
        elif pruning_method == "fpgm":
            log.info(f"Computing FPGM redundancy scores{suffix} from model weights...")
            pruner.compute_fpgm_scores()

    # ----------------------------
    # Pruning stage(s)
    # ----------------------------
    applied_total = 0

    if scope_norm == "custom_seq":
        target_enc, target_bot, target_dec = compute_targets_from_pct(
            int(args.prune_channels),
            float(args.prune_encoder_pct),
            float(args.prune_bottleneck_pct),
            float(args.prune_decoder_pct),
        )

        log.info("=" * 92)
        log.info("CUSTOM_SEQ TARGETS")
        log.info("=" * 92)
        log.info(
            f"Total target: {int(args.prune_channels)} | "
            f"encoder={target_enc} | bottleneck={target_bot} | decoder={target_dec}"
        )

        if is_weight_based:
            _infer_input_hw_for_weight_based()

        base_dep = take_dep_snapshot(net)

        def _stage(stage_scope: str, stage_target: int, stage_label: str) -> int:
            if stage_target <= 0:
                log.info(f"Stage {stage_label}: skipped (target=0)")
                return 0

            pruner.scope = stage_scope
            pruner.budget = None

            _run_taylor_batches(stage_label)  # no-op for weight-based

            if is_weight_based:
                _compute_weight_scores(stage_label)

            pruner.log_rank_stats(log)
            _ = pruner.preview_selection(int(stage_target))
            pruner.log_selection_distribution(log, pruner._last_selected)

            to_apply = int(stage_target)
            if friendly_enabled and int(friendly_multiple) > 1 and getattr(pruner, "_effective_prune_last", None) is not None:
                to_apply = int(pruner._effective_prune_last)

            pruner.pruning(int(to_apply))
            log.info(
                f"Stage {stage_label}: requested={stage_target}, applied={to_apply} "
                f"(friendly-adjusted={to_apply != stage_target})"
            )
            return int(to_apply)

        applied_total += _stage("encoder", int(target_enc), "encoder")

        dep = dep_pruned_counts(base_dep, take_dep_snapshot(net))
        bot_dep = int(dep.get("bottleneck_dep", 0))

        remaining_bot = max(0, int(target_bot) - int(bot_dep))
        if remaining_bot <= 0:
            log.info("Stage bottleneck: skipped (target satisfied by dependencies)")
        else:
            applied_total += _stage("bottleneck", int(remaining_bot), "bottleneck")

        dep = dep_pruned_counts(base_dep, take_dep_snapshot(net))
        dec_dep = int(dep.get("decoder_dep", 0))

        remaining_dec = max(0, int(target_dec) - int(dec_dep))
        if remaining_dec <= 0:
            log.info("Stage decoder: skipped (target satisfied by dependencies)")
        else:
            applied_total += _stage("decoder", int(remaining_dec), "decoder")

        if input_hw is None or baseline_flops_sum is None or baseline_flops_named is None:
            raise RuntimeError("Could not infer input HW / baseline FLOPs (train set empty?)")

        baseline_metrics["flops_g"] = float(baseline_flops_sum) / 1e9
        log.info(f"Custom_seq pruning complete: total_applied={applied_total}")

    else:
        # ----------------------------
        # Single pass
        # ----------------------------
        if is_weight_based:
            _infer_input_hw_for_weight_based()
            _compute_weight_scores()
        else:
            _run_taylor_batches("single_pass")

        if input_hw is None or baseline_flops_sum is None or baseline_flops_named is None:
            raise RuntimeError("Could not infer input HW / baseline FLOPs (train set empty?)")

        baseline_metrics["flops_g"] = float(baseline_flops_sum) / 1e9

        pruner.log_rank_stats(log)
        _ = pruner.preview_selection(int(requested_prune_user))
        pruner.log_selection_distribution(log, pruner._last_selected)

        to_apply = int(requested_prune_user)
        if friendly_enabled and friendly_multiple > 1:
            if getattr(pruner, "_effective_prune_last", None) is not None:
                to_apply = int(pruner._effective_prune_last)
            else:
                to_apply = int(requested_prune_effective)

        pruner.pruning(int(to_apply))
        applied_total = int(to_apply)
        log.info(f"Pruning complete: requested removal={applied_total}")

    # ----------------------------
    # Save & rebuild pruned architecture
    # ----------------------------
    save_pruned_pth = osp.join(save_dir, "Pruned.pth")
    save_pruned_txt = osp.join(save_dir, "pruned_channels.txt")
    torch.save(net.state_dict(), save_pruned_pth)
    pruner.channel_save(save_pruned_txt)
    log.info(f"Saved pruned weights : {save_pruned_pth}")
    log.info(f"Saved pruned channels: {save_pruned_txt}")

    net_pr = UNet(n_channels=3, n_classes=2, f_channels=save_pruned_txt).to(device)
    net_pr.load_state_dict(load_state_dict_weights_only(save_pruned_pth, map_location=device), strict=False)
    net = net_pr

    # ----------------------------
    # Post-prune metrics (params/flops)
    # ----------------------------
    pruned_params_total = count_params(net)
    pruned_size_mb = size_mb_from_state_dict(net.state_dict())
    pruned_param_parts = params_by_part(net)

    dummy = torch.zeros((1, 3, input_hw[0], input_hw[1]), device=device, dtype=torch.float32)
    pruned_flops_named = conv_flops_named_no_grad(net, dummy)
    pruned_flops_named_no_outc = [(n, f) for (n, f) in pruned_flops_named if n != "outc.conv"]
    pruned_flops_sum = int(sum(v for _, v in pruned_flops_named_no_outc))

    pruned_metrics: Dict[str, Any] = {
        "dice": float("nan"),
        "params": int(pruned_params_total),
        "size_mb": float(pruned_size_mb),
        "flops_g": float(pruned_flops_sum) / 1e9,
    }

    if int(args.print_flops) == 1:
        print_flops_per_conv(log, "FLOPs after pruning (per conv):", pruned_flops_named_no_outc)

    # ----------------------------
    # Finetune
    # ----------------------------
    log.info("=" * 92)
    log.info("FINETUNE")
    log.info("=" * 92)

    optimizer = optim.SGD(net.parameters(), lr=float(args.lr), momentum=0.9, weight_decay=0.0005)
    save_finetuned = osp.join(save_dir, "Finetuned.pth")

    finetune(
        net,
        optimizer,
        criterion,
        train_ids,
        log,
        save_finetuned,
        int(args.iters),
        args.epochs,
        int(args.batch_size),
        (device.type == "cuda"),
        float(args.scale),
        dir_img=dir_img,
        dir_mask=dir_mask,
        valset=val_ids,
        dir_img_val=dir_img,
        dir_mask_val=dir_mask,
        eval_fn=eval_net,
    )
    log.info(f"Saved finetuned weights: {save_finetuned}")

    # Note: net already holds the best weights (loaded inside finetune())
    log.info("Evaluating FINETUNED model...")
    val_gen_eval = get_imgs_and_masks(val_ids, dir_img, dir_mask, args.scale)
    finetuned_dice = eval_net(net, val_gen_eval, len(val_ids), (device.type == "cuda"), args.batch_size)
    seed_all(int(args.seed), use_cuda=(device.type == "cuda"))
    pruned_metrics["dice"] = float(finetuned_dice)
    log.info(f"FINETUNED Dice: {finetuned_dice:.6f}")

    # ----------------------------
    # Reporting
    # ----------------------------
    log.info("=" * 92)
    log.info("SUMMARY")
    log.info("=" * 92)

    if int(args.print_table) == 1:
        print_final_summary(log, baseline_metrics, pruned_metrics)

        base_flops_parts = summarize_flops_by_part(baseline_flops_named)
        pr_flops_parts = summarize_flops_by_part(pruned_flops_named_no_outc)

        print_part_compare(
            log,
            "PER-PART FLOPs (Baseline vs Pruned+Finetuned)",
            base_flops_parts,
            pr_flops_parts,
            unit="flops",
        )

        print_part_compare(
            log,
            "PER-PART Params (Baseline vs Pruned+Finetuned)",
            baseline_param_parts,
            pruned_param_parts,
            unit="params",
        )

    # ----------------------------
    # Optional benchmarking
    # ----------------------------
    if bool(args.bench):
        log.info("=" * 92)
        log.info("INFERENCE BENCHMARK (Optional)")
        log.info("=" * 92)

        bench_batch = int(args.bench_batch)
        bench_warmup = int(args.bench_warmup)
        bench_runs = int(args.bench_runs)
        bench_amp = bool(int(args.bench_amp) == 1)
        bench_cl = bool(int(args.bench_channels_last) == 1)
        bench_threads = int(args.bench_threads)

        net_cpu = UNet(n_channels=3, n_classes=2, f_channels=save_pruned_txt).cpu()
        net_cpu.load_state_dict(load_state_dict_weights_only(save_finetuned, map_location="cpu"), strict=False)
        cpu_stats = benchmark_inference_avg_ms(
            net_cpu,
            device=torch.device("cpu"),
            input_hw=input_hw,
            batch_size=bench_batch,
            warmup=bench_warmup,
            runs=bench_runs,
            use_amp=False,
            channels_last=bench_cl,
            threads=bench_threads,
        )

        gpu_stats = None
        if torch.cuda.is_available():
            net_gpu = UNet(n_channels=3, n_classes=2, f_channels=save_pruned_txt).cuda()
            net_gpu.load_state_dict(load_state_dict_weights_only(save_finetuned, map_location="cuda"), strict=False)
            gpu_stats = benchmark_inference_avg_ms(
                net_gpu,
                device=torch.device("cuda"),
                input_hw=input_hw,
                batch_size=bench_batch,
                warmup=bench_warmup,
                runs=bench_runs,
                use_amp=bench_amp,
                channels_last=bench_cl,
                threads=0,
            )

        log.info(
            f"CPU avg={cpu_stats['avg_ms']:.3f} ms (std={cpu_stats['std_ms']:.3f}) | "
            f"runs={bench_runs} warmup={bench_warmup}"
        )
        if gpu_stats is not None:
            log.info(f"GPU avg={gpu_stats['avg_ms']:.3f} ms (std={gpu_stats['std_ms']:.3f}) | amp={int(bench_amp)}")
            if (cpu_stats["avg_ms"] > 0) and (gpu_stats["avg_ms"] > 0):
                speedup = float(cpu_stats["avg_ms"]) / float(gpu_stats["avg_ms"])
                log.info(f"Speedup (CPU/GPU): {speedup:.2f}×")
        else:
            log.info("GPU benchmark skipped (CUDA not available).")