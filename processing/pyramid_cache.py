from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from src.gui.rendering.cache.hybrid_cache import HybridPartitionCache


def build_pyramid(base_image: np.ndarray) -> Dict[float, np.ndarray]:
    return {
        1.0: base_image,
        0.5: cv2.resize(base_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
        0.25: cv2.resize(base_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA),
    }


class PyramidCache:
    def __init__(self) -> None:
        self._levels = HybridPartitionCache("microscope_pyramid_levels")
        self._well_to_base_id: dict[str, int] = {}

    def get(self, well: str, base_image: np.ndarray) -> Dict[float, np.ndarray]:
        base_id = id(base_image)
        previous_base_id = self._well_to_base_id.get(well)
        if previous_base_id is not None and previous_base_id != base_id:
            self._levels.invalidate_where(
                lambda entry: self._is_well_base_entry(entry.normalized_key, well=well, base_id=previous_base_id)
            )
        self._well_to_base_id[well] = base_id

        keys = {
            1.0: ("mic_pyramid", well, base_id, 1.0),
            0.5: ("mic_pyramid", well, base_id, 0.5),
            0.25: ("mic_pyramid", well, base_id, 0.25),
        }
        cached: Dict[float, np.ndarray] = {}
        for scale, key in keys.items():
            value = self._levels.get(key)
            if value is None:
                cached = {}
                break
            cached[scale] = value
        if cached:
            return cached

        pyramid = build_pyramid(base_image)
        for scale, image in pyramid.items():
            self._levels.put(keys[float(scale)], image)
        return pyramid

    @staticmethod
    def _is_well_base_entry(normalized_key: object, *, well: str, base_id: int) -> bool:
        if not isinstance(normalized_key, dict):
            return False
        tuple_items = normalized_key.get("__tuple__")
        if not isinstance(tuple_items, list) or len(tuple_items) < 4:
            return False
        return tuple_items[0] == "mic_pyramid" and tuple_items[1] == well and tuple_items[2] == base_id
