"""Load a source-bound recovery bundle before destructive legacy preprocessing."""

import hashlib
import json

import numpy as np

from .data_correction import correct_phase_offsets


def cube_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def prepare_recovery_source(path, name, original_selected, drift, background_coords, phases,
                            current_allowed=None, current_time=None):
    with np.load(path, allow_pickle=False) as bundle:
        metadata = json.loads(str(bundle['metadata'].item()))
        record = metadata['wells'][name]
        if current_time is not None and not np.array_equal(bundle['time_seconds'], current_time):
            raise ValueError('Pixel recovery acquisition timestamps changed; regenerate the bundle')
        if record['input_sha256'] != cube_digest(original_selected):
            raise ValueError(f'{name}: pixel recovery source does not match raw data/orientation/selected range')
        if metadata['phases'] != list(phases):
            raise ValueError(f'{name}: pixel recovery source phase boundaries changed; regenerate the bundle')
        if metadata['registration_threshold'] != drift['threshold']:
            raise ValueError('Pixel recovery registration threshold changed; regenerate the bundle')
        if metadata['registration_method'] != drift['filter_method']:
            raise ValueError('Pixel recovery registration method changed; regenerate the bundle')
        values = bundle[name].copy()
        allowed = bundle[name + '_allowed']
        if current_allowed is not None and not np.array_equal(allowed, current_allowed):
            raise ValueError(f'{name}: border/active-area settings changed; regenerate pixel recovery')
    if values.shape != original_selected.shape:
        raise ValueError('Pixel recovery cube shape does not match selected raw data')
    corrected = (values - values[0]) * 1000
    coords = []
    for x, y in background_coords:
        x, y = int(x), int(y)
        if 0 <= x < allowed.shape[1] and 0 <= y < allowed.shape[0] and allowed[y, x]:
            if np.isfinite(corrected[:, y, x]).all():
                coords.append((x, y))
    if not coords:
        candidates = allowed & np.isfinite(corrected).all(axis=0)
        candidates &= np.nan_to_num(corrected[-1], nan=np.inf) < drift['threshold']
        coords = [tuple(p[::-1]) for p in np.argwhere(candidates)[:7]]
    if not coords:
        raise ValueError(f'{name}: no valid background pixels in recovered source')
    xs, ys = np.array(coords).T
    reducer = np.median if drift['filter_method'] == 'median' else np.mean
    background = reducer(corrected[:, ys, xs], axis=1)
    corrected -= background[:, None, None]
    corrected = np.maximum(corrected, 0)
    boundary_missing = np.zeros(allowed.shape, dtype=bool)
    if drift.get('phase_offset_correction', True):
        for start in phases:
            if 0 < int(start) < len(corrected):
                boundary_missing |= ~np.isfinite(corrected[int(start) - 1]) | ~np.isfinite(corrected[int(start)])
        corrected, _ = correct_phase_offsets(corrected, phases)
    # Preserve missingness through all corrections, then render as zero for
    # legacy finite-cube consumers. The bundle retains the full method mask.
    error_mask = ~allowed | ~np.isfinite(corrected).any(axis=0) | boundary_missing
    # A skipped nonfinite boundary must not become an artificial zero-to-signal
    # step when legacy consumers render missing values as zero. Such pixels have
    # no observable endpoint pair for estimating the requested phase offset.
    corrected[:, error_mask] = 0
    return np.nan_to_num(corrected, nan=0), coords, error_mask
