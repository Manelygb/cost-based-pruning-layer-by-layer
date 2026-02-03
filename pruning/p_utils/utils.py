# pruning/p_utils/utils.py
from __future__ import annotations

import json
import os
import random
from typing import Iterable, Iterator, List

import numpy as np


def get_square(img: np.ndarray, pos: int) -> np.ndarray:
    """Extract left (pos=0) or right (pos=1) square from an HxWxC array."""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    return img[:, -h:]


def split_img_into_squares(img: np.ndarray):
    return get_square(img, 0), get_square(img, 1)


def hwc_to_chw(img: np.ndarray) -> np.ndarray:
    return np.transpose(img, axes=[2, 0, 1])


def resize_and_crop(pilimg, scale: float = 0.5, final_height: int | None = None) -> np.ndarray:
    """Resize PIL image by scale and center-crop height to final_height if provided."""
    w, h = pilimg.size
    newW, newH = int(w * scale), int(h * scale)

    diff = 0 if final_height is None else (newH - int(final_height))

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)


def batch(iterable: Iterable, batch_size: int, shuffle: bool = True) -> Iterator[List]:
    """Yield lists of length batch_size from iterable.
    Note: if shuffle=True, shuffles elements *within each yielded batch*.
    """
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            if shuffle:
                random.shuffle(b)
            yield b
            b = []

    if b:
        if shuffle:
            random.shuffle(b)
        yield b


def split_train_val(dataset, val_percent: float = 0.05, splitfile: str = "data/trainval.json"):
    """Shuffle dataset and split into train/val; save split to JSON."""
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * float(val_percent))
    if n <= 0:
        n = max(1, min(length, 1)) if length > 0 else 0

    random.shuffle(dataset)
    data = {"train": dataset[:-n] if n > 0 else dataset, "val": dataset[-n:] if n > 0 else []}

    os.makedirs(os.path.dirname(splitfile) or ".", exist_ok=True)
    with open(splitfile, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return data


def normalize(x: np.ndarray) -> np.ndarray:
    return x / 255.0


def merge_masks(img1: np.ndarray, img2: np.ndarray, full_w: int) -> np.ndarray:
    h = img1.shape[0]
    new = np.zeros((h, full_w), np.float32)
    new[:, : full_w // 2 + 1] = img1[:, : full_w // 2 + 1]
    new[:, full_w // 2 + 1 :] = img2[:, -(full_w // 2 - 1) :]
    return new


def rle_encode(mask_image: np.ndarray) -> np.ndarray:
    """Run-length encoding for binary masks."""
    pixels = mask_image.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs
