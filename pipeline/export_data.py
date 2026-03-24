from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..epic_cardio.data_correction import correct_well
from ..epic_cardio.defs import WELL_NAMES


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
