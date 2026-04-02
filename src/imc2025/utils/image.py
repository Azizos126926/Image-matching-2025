from __future__ import annotations

import numpy as np
import kornia as K
import torch


def load_torch_image(fname: str, device: torch.device | str = "cpu") -> torch.Tensor:
    return K.io.load_image(fname, K.io.ImageLoadType.RGB32, device=device)[None, ...]


def pad_to_square(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        height, width, channels = image.shape
        max_dim = max(height, width)
        canvas = np.zeros((max_dim, max_dim, channels), dtype=image.dtype)
        canvas[:height, :width, :] = image
        return canvas

    if image.ndim == 2:
        height, width = image.shape
        max_dim = max(height, width)
        canvas = np.zeros((max_dim, max_dim), dtype=image.dtype)
        canvas[:height, :width] = image
        return canvas

    raise ValueError(f"Unsupported image rank: {image.ndim}")
