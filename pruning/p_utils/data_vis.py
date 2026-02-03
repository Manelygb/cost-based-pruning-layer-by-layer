# pruning/p_utils/data_vis.py
from __future__ import annotations

import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask) -> None:
    """Quick visualization helper for a single image and mask."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].set_title("Input image")
    axes[0].imshow(img)
    axes[0].axis("off")

    axes[1].set_title("Output mask")
    axes[1].imshow(mask)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
