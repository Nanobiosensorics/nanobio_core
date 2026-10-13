from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..epic_cardio.defs import WELL_NAMES


def default_border_filter() -> Dict[str, int]:
    return {
        "top": 0,
        "bottom": 0,
        "left": 0,
        "right": 0,
    }


def default_border_filter_per_well() -> Dict[str, Dict[str, int]]:
    return {well_name: default_border_filter() for well_name in WELL_NAMES}


def sanitize_border_filter(border: Any) -> Dict[str, int]:
    if not isinstance(border, dict):
        return default_border_filter()
    defaults = default_border_filter()
    return {
        "top": max(0, int(border.get("top", defaults["top"]) or 0)),
        "bottom": max(0, int(border.get("bottom", defaults["bottom"]) or 0)),
        "left": max(0, int(border.get("left", defaults["left"]) or 0)),
        "right": max(0, int(border.get("right", defaults["right"]) or 0)),
    }


def normalize_border_filter_per_well(data: Any) -> Dict[str, Dict[str, int]]:
    normalized = default_border_filter_per_well()
    if not isinstance(data, dict):
        return normalized
    for well_name in WELL_NAMES:
        normalized[well_name] = sanitize_border_filter(data.get(well_name, {}))
    return normalized


def _legacy_global_border(localization_params: dict) -> Dict[str, int]:
    if "filter_border" in localization_params:
        return sanitize_border_filter(localization_params.get("filter_border"))
    width = max(0, int(localization_params.get("filter_border_width", 0) or 0))
    return {
        "top": width,
        "bottom": width,
        "left": width,
        "right": width,
    }


def extract_border_filter_per_well(localization_params: dict) -> Dict[str, Dict[str, int]]:
    per_well = localization_params.get("filter_border_per_well")
    if isinstance(per_well, dict):
        return normalize_border_filter_per_well(per_well)
    legacy = _legacy_global_border(localization_params)
    return {well_name: dict(legacy) for well_name in WELL_NAMES}


def border_filter_for_well(localization_params: dict, well_name: str) -> Dict[str, int]:
    per_well = extract_border_filter_per_well(localization_params)
    return dict(per_well.get(well_name, default_border_filter()))


def set_border_filter_for_well(
    localization_params: dict,
    well_name: str,
    border: Any,
) -> Dict[str, Dict[str, int]]:
    per_well = extract_border_filter_per_well(localization_params)
    if well_name in WELL_NAMES:
        per_well[well_name] = sanitize_border_filter(border)
    localization_params["filter_border_per_well"] = per_well
    return per_well


def set_border_filter_for_all(localization_params: dict, border: Any) -> Dict[str, Dict[str, int]]:
    clean = sanitize_border_filter(border)
    per_well = {well_name: dict(clean) for well_name in WELL_NAMES}
    localization_params["filter_border_per_well"] = per_well
    return per_well


def reset_border_filter_per_well(localization_params: dict) -> Dict[str, Dict[str, int]]:
    per_well = default_border_filter_per_well()
    localization_params["filter_border_per_well"] = per_well
    return per_well


def allowed_pixel_mask_for_well(localization_params: dict, well_name: str, shape=(80, 80)) -> np.ndarray:
    """Combine rectangular trims with an optional saved active-sensor mask.

    Active masks use one binary string per sensor row; ``1`` denotes a usable
    pixel after the measurement-specific inward margin has been applied.
    """
    height, width = shape
    border = border_filter_for_well(localization_params, well_name)
    yy, xx = np.indices(shape)
    allowed = (
        (yy >= border['top']) & (yy < height - border['bottom'])
        & (xx >= border['left']) & (xx < width - border['right'])
    )
    record = localization_params.get('filter_active_area_per_well', {}).get(well_name)
    if record is not None:
        rows = record.get('rows', [])
        if len(rows) != height or any(
            not isinstance(row, str) or len(row) != width or set(row) - {'0', '1'} for row in rows
        ):
            raise ValueError(f'{well_name}: active-area filter must contain {height} binary rows of width {width}')
        allowed &= np.array([[value == '1' for value in row] for row in rows], dtype=bool)
    return allowed


def points_inside_allowed_area(points: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Test x/y sensor coordinates without clipping out-of-frame points inward."""
    points = np.asarray(points)
    keep = np.zeros(len(points), dtype=bool)
    if not len(points):
        return keep
    height, width = allowed.shape
    valid = (
        np.isfinite(points[:, :2]).all(axis=1)
        & (points[:, 0] >= 0) & (points[:, 0] <= width - 1)
        & (points[:, 1] >= 0) & (points[:, 1] <= height - 1)
    )
    indices = np.flatnonzero(valid)
    xy = np.rint(points[indices, :2]).astype(int)
    keep[indices] = allowed[xy[:, 1], xy[:, 0]]
    return keep
