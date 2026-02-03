# pruning/p_utils/crf.py
from __future__ import annotations

import numpy as np

try:
    import pydensecrf.densecrf as dcrf
except Exception:  # optional dependency
    dcrf = None


def dense_crf(img: np.ndarray, output_probs: np.ndarray, n_iters: int = 5) -> np.ndarray:
    """DenseCRF refinement for a binary segmentation probability map.

    Args:
        img: HxWx3 uint8/float image.
        output_probs: HxW probability of class=1 (float in [0,1]).
        n_iters: CRF inference iterations.

    Returns:
        HxW int mask in {0,1}.
    """
    if dcrf is None:
        raise ImportError("pydensecrf is not installed. Install it to use dense_crf().")

    h, w = int(output_probs.shape[0]), int(output_probs.shape[1])

    probs = np.expand_dims(output_probs, 0)
    probs = np.append(1.0 - probs, probs, axis=0)  # [2,H,W]

    d = dcrf.DenseCRF2D(w, h, 2)

    U = -np.log(np.clip(probs, 1e-8, 1.0))
    U = U.reshape((2, -1))
    d.setUnaryEnergy(np.ascontiguousarray(U))

    d.addPairwiseGaussian(sxy=20, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=np.ascontiguousarray(img), compat=10)

    Q = d.inference(int(n_iters))
    return np.argmax(np.array(Q), axis=0).reshape((h, w)).astype(np.int64)
