from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def build_pyramid(base_image: np.ndarray) -> Dict[float, np.ndarray]:
    return {
        1.0: base_image,
        0.5: cv2.resize(base_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
        0.25: cv2.resize(base_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA),
    }


class PyramidCache:
    def __init__(self) -> None:
        self._cache: Dict[int, Dict[float, np.ndarray]] = {}

    def get(self, well: str, base_image: np.ndarray) -> Dict[float, np.ndarray]:
        del well
        key = id(base_image)
        if key in self._cache:
            return self._cache[key]

        pyramid = build_pyramid(base_image)
        self._cache[key] = pyramid
        return pyramid
