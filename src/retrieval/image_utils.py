# -*- coding: utf-8 -*-
"""图像输入统一转为 PIL RGB。"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from PIL import Image
except ModuleNotFoundError as e:  # pragma: no cover
    raise ImportError("image_utils requires pillow") from e


def pil_from_any(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3 and arr.shape[2] >= 3:
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr[:, :, :3]).convert("RGB")
        raise TypeError(f"unsupported ndarray shape for image: {arr.shape}")
    raise TypeError(f"unsupported image type: {type(image)}")
