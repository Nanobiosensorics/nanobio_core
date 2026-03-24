from __future__ import annotations

import numpy as np


def mask_phase_samples(lines, phases):
    arr = np.asarray(lines, dtype=np.float32).copy()
    if arr.ndim == 1:
        for phase in phases:
            if 0 <= phase < arr.shape[0]:
                arr[phase] = np.nan
        return arr

    for phase in phases:
        if 0 <= phase < arr.shape[1]:
            arr[:, phase] = np.nan
    return arr


def mean_baseline_series(well: np.ndarray) -> np.ndarray:
    series = np.mean(well, axis=(1, 2))
    series = series - series[0]
    return series
