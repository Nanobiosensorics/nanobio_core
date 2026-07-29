from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..epic_cardio.data_correction import correct_well
from ..epic_cardio.defs import WELL_NAMES


@dataclass(frozen=True)
class MicroscopeCellImageData:
    label: int
    image: np.ndarray
    overlay_crop: Optional[np.ndarray]
    overlay_vmin: float
    overlay_vmax: float
    focused_contour: np.ndarray
    strategy_contour: np.ndarray
    crop_bounds: Tuple[int, int, int, int]


def extract_signal_lines(
    ptss_selected: np.ndarray,
    pre_cube: np.ndarray,
    raw_cube: np.ndarray,
    well_pixel_sets: Iterable[np.ndarray] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    lines_selected: List[np.ndarray] = []
    lines_integrated: List[np.ndarray] = []
    raw_lines_selected: List[np.ndarray] = []
    pixel_sets = list(well_pixel_sets or [])
    use_pixel_sets = len(pixel_sets) == int(ptss_selected.shape[0]) and len(pixel_sets) > 0

    for idx in range(ptss_selected.shape[0]):
        if use_pixel_sets:
            coords = np.asarray(pixel_sets[idx], dtype=np.int32)
            if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
                coords = np.asarray([[ptss_selected[idx, 0], ptss_selected[idx, 1]]], dtype=np.int32)
            xs = np.clip(coords[:, 0], 0, pre_cube.shape[2] - 1)
            ys = np.clip(coords[:, 1], 0, pre_cube.shape[1] - 1)
            pre_values = pre_cube[:, ys, xs]
            raw_values = raw_cube[:, ys, xs]
            lines_selected.append(np.max(pre_values, axis=1))
            lines_integrated.append(np.sum(pre_values, axis=1))
            raw_lines_selected.append(np.max(raw_values, axis=1))
            continue

        line = pre_cube[:, ptss_selected[idx, 1], ptss_selected[idx, 0]]
        lines_selected.append(line)
        lines_integrated.append(line)
        raw_lines_selected.append(raw_cube[:, ptss_selected[idx, 1], ptss_selected[idx, 0]])

    return (
        np.asarray(lines_selected),
        np.asarray(lines_integrated),
        np.asarray(raw_lines_selected),
        use_pixel_sets,
    )


def build_foreground_pixel_set(
    pre_cube: np.ndarray,
    foreground_threshold: float,
    neighborhood_size: int,
) -> np.ndarray:
    if pre_cube.ndim != 3 or pre_cube.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    from scipy import ndimage

    preview_image = np.max(np.asarray(pre_cube, dtype=np.float32), axis=0)
    data_max = np.asarray(ndimage.maximum_filter(preview_image, int(neighborhood_size)), dtype=np.float32)
    data_min = np.asarray(ndimage.minimum_filter(preview_image, int(neighborhood_size)), dtype=np.float32)
    contrast = data_max - data_min
    ys, xs = np.nonzero(contrast > float(foreground_threshold))
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack((xs, ys)).astype(np.int32, copy=False)


def combine_pixel_sets(pixel_sets: Iterable[np.ndarray] | None) -> np.ndarray:
    if pixel_sets is None:
        return np.empty((0, 2), dtype=np.int32)

    merged: List[np.ndarray] = []
    for coords in pixel_sets:
        arr = np.asarray(coords, dtype=np.int32)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
            continue
        merged.append(arr)

    if not merged:
        return np.empty((0, 2), dtype=np.int32)

    return np.unique(np.vstack(merged), axis=0).astype(np.int32, copy=False)


def aggregate_signal_for_pixel_set(
    pre_cube: np.ndarray,
    pixel_set: np.ndarray,
) -> np.ndarray:
    coords = np.asarray(pixel_set, dtype=np.int32)
    if pre_cube.ndim != 3 or pre_cube.size == 0 or coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)

    xs = np.clip(coords[:, 0], 0, pre_cube.shape[2] - 1)
    ys = np.clip(coords[:, 1], 0, pre_cube.shape[1] - 1)
    values = np.asarray(pre_cube[:, ys, xs], dtype=np.float32)
    return np.sum(values, axis=1, dtype=np.float32)


def build_microscope_cell_image_data(
    microscope_image: np.ndarray,
    microscope_mask: np.ndarray,
    centroid: np.ndarray,
    label: int,
    crop_size: int,
    *,
    epic_image: Optional[np.ndarray] = None,
    aligned_rect: Optional[Tuple[float, float, float, float]] = None,
    strategy_pixel_set: Optional[np.ndarray] = None,
    overlay_limits: Optional[Tuple[float, float]] = None,
) -> Optional[MicroscopeCellImageData]:
    image = np.asarray(microscope_image)
    mask = np.asarray(microscope_mask, dtype=np.int32)
    center = np.asarray(centroid, dtype=np.float32).reshape(-1)
    if (
        image.ndim not in {2, 3}
        or mask.ndim != 2
        or image.shape[:2] != mask.shape
        or image.size == 0
        or center.size < 2
        or not np.all(np.isfinite(center[:2]))
    ):
        return None

    size = max(1, int(crop_size))
    x = int(round(float(center[0])))
    y = int(round(float(center[1])))
    half_low = size // 2
    half_high = size - half_low
    y0 = max(0, y - half_low)
    y1 = min(mask.shape[0], y + half_high)
    x0 = max(0, x - half_low)
    x1 = min(mask.shape[1], x + half_high)
    if x1 <= x0 or y1 <= y0:
        return None

    if image.ndim == 3:
        crop_image = np.mean(image[y0:y1, x0:x1, :3], axis=2).astype(np.float32, copy=False)
    else:
        crop_image = np.asarray(image[y0:y1, x0:x1], dtype=np.float32)
    crop_mask = mask[y0:y1, x0:x1]
    focused_contour = segment_contour_from_region(crop_mask == int(label))
    strategy_contour = np.zeros(crop_mask.shape, dtype=bool)
    overlay_crop = None
    overlay_vmin, overlay_vmax = 0.0, 1.0

    epic = np.asarray(epic_image, dtype=np.float32) if epic_image is not None else None
    if (
        epic is not None
        and epic.ndim == 2
        and epic.size > 0
        and aligned_rect is not None
        and len(aligned_rect) == 4
    ):
        rect_x, rect_y, rect_w, rect_h = (float(value) for value in aligned_rect)
        if rect_w > 0.0 and rect_h > 0.0:
            if overlay_limits is None:
                overlay_vmin = float(np.min(epic))
                overlay_vmax = float(np.max(epic))
            else:
                overlay_vmin = float(overlay_limits[0])
                overlay_vmax = float(overlay_limits[1])
            if np.isclose(overlay_vmin, overlay_vmax):
                delta = (
                    1.0
                    if np.isclose(overlay_vmax, 0.0)
                    else max(abs(overlay_vmax) * 0.01, 1e-6)
                )
                overlay_vmin -= delta
                overlay_vmax += delta

            crop_h, crop_w = crop_mask.shape
            yy, xx = np.indices((crop_h, crop_w), dtype=np.float32)
            norm_x = (float(x0) + xx - rect_x) / rect_w
            norm_y = (float(y0) + yy - rect_y) / rect_h
            valid = (
                (norm_x >= 0.0)
                & (norm_x <= 1.0)
                & (norm_y >= 0.0)
                & (norm_y <= 1.0)
            )
            epic_x = np.clip(
                np.rint(norm_x * (epic.shape[1] - 1)).astype(np.int32),
                0,
                epic.shape[1] - 1,
            )
            epic_y = np.clip(
                np.rint(norm_y * (epic.shape[0] - 1)).astype(np.int32),
                0,
                epic.shape[0] - 1,
            )
            overlay_crop = np.full(crop_mask.shape, np.nan, dtype=np.float32)
            overlay_crop[valid] = epic[epic_y[valid], epic_x[valid]]

            coords = (
                np.empty((0, 2), dtype=np.int32)
                if strategy_pixel_set is None
                else np.asarray(strategy_pixel_set, dtype=np.int32)
            )
            if coords.ndim == 2 and coords.shape[1] == 2 and coords.shape[0] > 0:
                coords = coords[
                    (coords[:, 0] >= 0)
                    & (coords[:, 0] < epic.shape[1])
                    & (coords[:, 1] >= 0)
                    & (coords[:, 1] < epic.shape[0])
                ]
                if coords.shape[0] > 0:
                    selected_ids = coords[:, 1] * epic.shape[1] + coords[:, 0]
                    flat_ids = epic_y * epic.shape[1] + epic_x
                    strategy_region = valid & np.isin(flat_ids, selected_ids)
                    strategy_contour = segment_contour_from_region(strategy_region)

    return MicroscopeCellImageData(
        label=int(label),
        image=crop_image,
        overlay_crop=overlay_crop,
        overlay_vmin=float(overlay_vmin),
        overlay_vmax=float(overlay_vmax),
        focused_contour=focused_contour,
        strategy_contour=strategy_contour,
        crop_bounds=(int(x0), int(y0), int(x1), int(y1)),
    )


def segment_contour_from_region(region: np.ndarray) -> np.ndarray:
    region_mask = np.asarray(region, dtype=bool)
    if region_mask.size == 0 or not np.any(region_mask):
        return np.zeros_like(region_mask, dtype=bool)

    padded = np.pad(region_mask, 1, mode="constant", constant_values=False)
    eroded = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return region_mask & (~eroded)


def compute_breakdowns(raw_wells: Dict[str, np.ndarray], full_phases: List[int]) -> Dict[str, int]:
    breakdowns: Dict[str, int] = {}
    for name in WELL_NAMES:
        line = np.mean(raw_wells[name], axis=(1, 2))
        peak_until = full_phases[-1] + np.argmax(line[full_phases[-1]:])
        peak_until = peak_until if line[peak_until] > line[full_phases[-1] - 1] else full_phases[-1] - 1
        breakdowns[name] = int(peak_until)
    return breakdowns


def build_signal_parts(raw_lines_selected: np.ndarray, phases: List[int]) -> List[Tuple[int, np.ndarray]]:
    parts: List[Tuple[int, np.ndarray]] = []
    for idx, (start, end) in enumerate(zip([0] + phases, phases + [None])):
        selection = raw_lines_selected[:, start:end].copy()
        selection = (selection.T - selection.T[0]).T
        selection *= 1000
        parts.append((idx, selection))
    return parts


def build_breakdown_lines(
    raw_well: np.ndarray,
    filter_points,
    ptss_selected: np.ndarray,
    selected_range,
    breakdown_index: int,
) -> np.ndarray:
    slicer = slice(selected_range[0], breakdown_index)
    well_tmp = raw_well[slicer]
    well_corr, _, _ = correct_well(well_tmp, coords=filter_points)
    breakdown_lines = []
    for idx in range(ptss_selected.shape[0]):
        breakdown_lines.append(well_corr[:, ptss_selected[idx, 1], ptss_selected[idx, 0]])
    return np.asarray(breakdown_lines)
