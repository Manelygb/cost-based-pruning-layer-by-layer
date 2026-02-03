# pruning/p_utils/load.py
from __future__ import annotations

import os
from typing import Iterable, Iterator, Tuple

import numpy as np
from PIL import Image

from .utils import get_square, hwc_to_chw, normalize, resize_and_crop


def get_ids(dir_path: str) -> list[str]:
    """Return list of sample ids (filenames without extension) from a directory."""
    out = []
    for f in os.listdir(dir_path):
        base, ext = os.path.splitext(f)
        if ext.lower() == ".jpg":
            out.append(base)
    return out


def split_ids(ids: Iterable[str], n: int = 2) -> list[Tuple[str, int]]:
    """Split each id into n tiles: returns (id, pos) for pos in [0..n-1]."""
    return [(id_, i) for id_ in ids for i in range(n)]


def to_cropped_imgs(ids: Iterable[Tuple[str, int]], dir_path: str, suffix: str, scale: float) -> Iterator[np.ndarray]:
    """Yield cropped numpy images from (id,pos) tuples."""
    for id_, pos in ids:
        p = os.path.join(dir_path, f"{id_}{suffix}")
        im = resize_and_crop(Image.open(p), scale=scale)
        yield get_square(im, pos)


def get_imgs_and_masks(ids: Iterable[Tuple[str, int]], dir_img: str, dir_mask: str, scale: float):
    """Yield (img, mask) pairs."""
    imgs = to_cropped_imgs(ids, dir_img, ".jpg", scale)
    imgs = map(hwc_to_chw, imgs)
    imgs = map(normalize, imgs)

    masks = to_cropped_imgs(ids, dir_mask, "_mask.gif", scale)
    return zip(imgs, masks)


def get_full_img_and_mask(id_: str, dir_img: str, dir_mask: str):
    p_img = os.path.join(dir_img, f"{id_}.jpg")
    p_msk = os.path.join(dir_mask, f"{id_}_mask.gif")
    im = Image.open(p_img)
    mask = Image.open(p_msk)
    return np.array(im), np.array(mask)
