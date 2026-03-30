# pruning/finetune.py
from __future__ import annotations

import os
import os.path as osp

import numpy as np
import torch
from tqdm import tqdm

from .p_utils.utils import batch
from .p_utils.load import get_imgs_and_masks
from common.metrics import AverageMeter
from common.measurements import flops_count


def _ensure_trailing_sep(p: str) -> str:
    p = osp.expanduser(p)
    return p if p.endswith(os.sep) else p + os.sep


def _resolve_data_dirs() -> tuple[str, str]:
    """Legacy resolution for backwards compatibility (env -> ~/work/carvana -> data/)."""
    env_img = os.environ.get("DATA_TRAIN_DIR")
    env_msk = os.environ.get("DATA_MASK_DIR")
    if env_img and env_msk:
        return _ensure_trailing_sep(env_img), _ensure_trailing_sep(env_msk)

    cand_img = osp.expanduser("~/work/carvana/train")
    cand_msk = osp.expanduser("~/work/carvana/train_masks")
    if osp.isdir(cand_img) and osp.isdir(cand_msk):
        return _ensure_trailing_sep(cand_img), _ensure_trailing_sep(cand_msk)

    return _ensure_trailing_sep("data/train/"), _ensure_trailing_sep("data/train_masks/")


def finetune(
    net,
    optimizer,
    criterion,
    trainset,
    log,
    path,
    iters: int = 100,
    epochs: int | None = None,
    batch_size: int = 2,
    gpu: bool = True,
    scale: float = 0.5,
    dir_img: str | None = None,
    dir_mask: str | None = None,
    # --- val args (optional) ---
    valset=None,
    dir_img_val: str | None = None,
    dir_mask_val: str | None = None,
    eval_fn=None,
) -> None:
    """Fine-tune on Carvana-style dataset generators.

    Notes:
    - Keeps original CE training logic: logits [B,2,H,W] + targets [B,H,W] long.
    - If dir_img/dir_mask are not provided, falls back to legacy resolution.
    - If valset + dir_img_val + dir_mask_val + eval_fn are all provided,
      val Dice is computed after every epoch and the best checkpoint is saved.
      Otherwise behaviour is identical to the original (save last epoch).
    """
    net.train()
    loss_meter = AverageMeter()

    if dir_img is None or dir_mask is None:
        dir_img, dir_mask = _resolve_data_dirs()
    else:
        dir_img, dir_mask = _ensure_trailing_sep(dir_img), _ensure_trailing_sep(dir_mask)

    log.info(f"[finetune] Using dir_img={dir_img} dir_mask={dir_mask}")

    use_val = (valset is not None and dir_img_val is not None
               and dir_mask_val is not None and eval_fn is not None)

    if use_val:
        log.info(f"[finetune] Val-based best-checkpoint saving ENABLED ({len(valset)} val samples)")
    else:
        log.info("[finetune] Val args not provided — will save last-epoch checkpoint (original behaviour)")

    # If epochs is None -> run "iters" mini-batches total (1 epoch loop wrapper kept).
    if epochs is None:
        epochs = 1
    else:
        iters = 0  # disabled when epochs explicitly set

    use_cuda = bool(gpu and torch.cuda.is_available())

    best_dice = -1.0
    best_path = path + ".best.pth"

    for e in range(int(epochs)):
        net.train()
        loss_meter = AverageMeter()

        train = get_imgs_and_masks(trainset, dir_img, dir_mask, scale)

        with tqdm(total=len(trainset)) as progress_bar:
            for i, b in enumerate(batch(train, batch_size)):
                imgs = np.array([x[0] for x in b], dtype=np.float32)
                true_masks = np.array([x[1] for x in b])

                imgs = torch.from_numpy(imgs)             # [B,3,H,W] float32
                true_masks = torch.from_numpy(true_masks) # [B,1,H,W] or [B,H,W]

                # Target for CrossEntropyLoss: [B,H,W] long in {0,1}
                if true_masks.ndim == 4 and true_masks.shape[1] == 1:
                    true_masks = true_masks[:, 0, :, :]

                if true_masks.max() > 1:
                    true_masks = (true_masks > 0).long()
                else:
                    true_masks = true_masks.long()

                if use_cuda:
                    imgs = imgs.cuda(non_blocking=True)
                    true_masks = true_masks.cuda(non_blocking=True)

                masks_pred = net(imgs)  # [B,2,H,W] logits

                if not masks_pred.is_floating_point():
                    masks_pred = masks_pred.float()

                loss = criterion(masks_pred, true_masks)
                loss_meter.update(float(loss.item()), int(imgs.shape[0]))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                progress_bar.update(int(imgs.shape[0]))
                progress_bar.set_postfix(epoch=e, loss=loss_meter.avg)

                if i == 0 and e == 0:
                    try:
                        log.info(f"FLOPs after pruning:\n{flops_count(net, imgs.shape[2:])}")
                    except Exception as ex:
                        log.info(f"[finetune] FLOPs count failed: {ex}")

                if iters and (i + 1) >= int(iters):
                    break

        log.info(f"Epoch {e + 1} finished | avg train loss: {loss_meter.avg:.6f}")

        if use_val:
            _dir_img_val = _ensure_trailing_sep(dir_img_val)
            _dir_mask_val = _ensure_trailing_sep(dir_mask_val)
            val_gen = get_imgs_and_masks(valset, _dir_img_val, _dir_mask_val, scale)
            val_dice = eval_fn(net, val_gen, len(valset), gpu, batch_size)
            net.train()  # restore train mode after eval

            log.info(f"Epoch {e +1 } | val Dice: {val_dice:.6f} ")

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(net.state_dict(), best_path)
                log.info(f" New best! Saved checkpoint (dice={best_dice:.6f})")

    log.info("Finished finetuning")
    log.info(f"Finetuned loss: {loss_meter.avg}")

    if use_val:
        if osp.exists(best_path):
            os.replace(best_path, path)
            log.info(f"Best val Dice={best_dice:.6f} — saved best checkpoint to {path}")
        else:
            torch.save(net.state_dict(), path)
            log.info(f"Warning: best checkpoint file missing; saved last epoch to {path}")

        state = torch.load(path, map_location="cuda" if use_cuda else "cpu", weights_only=True)
        net.load_state_dict(state)
        log.info("Loaded best weights back into model")
    else:
        torch.save(net.state_dict(), path)
        log.info(f"Saving finetuned to {path}...")