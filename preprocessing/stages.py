"""Pure stages in the prescribed quantitative order; nm through stage five."""

from dataclasses import asdict
import warnings
import numpy as np

from ..processing.active_area import active_area as detect_active_area
from ..epic_cardio.data_correction import _phase_translation, _shift_phase_frames
from ..epic_cardio.math_ops import find_local_maxima
from .model import StageState, PreprocessResult, OBSERVED, MISSING, EXCLUDED
from .recovery import (
    recover_pixels, detect_returning_pulses, detect_short_gaps, detect_persistent_steps,
    detect_terminal_dropouts, temporal_interpolate, spatial_fill, stitch_step,
)
from .curation import apply_curation


def _nanmedian(values, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def validity(raw, config):
    values = np.asarray(raw.values_nm, dtype=np.float64).copy()
    valid = (raw.codes != 0) & np.isfinite(values)
    boundaries = [0, *raw.phases, len(values)]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        jumps = np.abs(np.diff(values[left:right], axis=0)) * 1000 > config.validity.max_jump_pm
        valid[left + 1:right] &= ~jumps
    if config.validity.flat_trace_is_missing:
        valid[:, np.all(np.diff(values, axis=0) == 0, axis=0)] = False
    method = np.where(valid, OBSERVED, MISSING).astype(np.uint8)
    method[:, ~raw.allowed] = EXCLUDED
    values[method == MISSING] = np.nan
    receipt = dict(parameters=asdict(config.validity), missing_samples=int((method == MISSING).sum()),
                   phase_source=raw.phase_source)
    return StageState(raw, values, method, raw.allowed.copy(), {'validity': receipt})


def active_area(state, config):
    result = state.copy()
    # Failure is explicit: a damaged frame zero must not silently widen the domain.
    _, detected, threshold = detect_active_area(state.raw.values_nm[0], config.active_area.margin_px)
    result.allowed &= detected
    result.method[:, ~result.allowed] = EXCLUDED
    result.values[:, ~result.allowed] = state.raw.values_nm[:, ~result.allowed]
    result.receipts['active_area'] = dict(parameters=asdict(config.active_area), threshold=threshold,
                                        allowed_pixels=int(result.allowed.sum()), source_frame=0)
    return result


def interphase_registration(state, config):
    result = state.copy()
    bounds = [0, *state.raw.phases, len(state.values)]
    points = []
    if not config.registration.enabled or len(bounds) == 2:
        result.receipts['interphase_registration'] = dict(
            parameters=asdict(config.registration),
            shifts=[dict(start=left, end=right, dx=0, dy=0)
                    for left, right in zip(bounds[:-1], bounds[1:])])
        return result
    observed = state.method == OBSERVED
    baseline = _nanmedian(np.where(observed[:config.reference.frames],
                                   state.values[:config.reference.frames], np.nan))
    for left, right in zip(bounds[:-1], bounds[1:]):
        signal = np.where(observed[left:right], (state.values[left:right] - baseline) * 1000, np.nan)
        signal = np.where(np.isfinite(signal), signal, -np.inf)
        projection = np.max(signal, axis=0)
        # Unsupported pixels are masked and kept below the peak detection threshold.
        projection = np.where(np.isfinite(projection), projection, config.registration.threshold_pm - 1)
        xs, ys = find_local_maxima(projection, config.registration.threshold_pm,
                                  np.max(projection), config.registration.neighborhood_size, ~state.allowed)
        points.append(np.column_stack([xs, ys]))
    reference = int(np.argmax([len(p) for p in points]))
    shifts = []
    for i, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
        shift = np.zeros(2, dtype=int)
        if config.registration.enabled and i != reference:
            shift = _phase_translation(points[i], points[reference], config.registration.max_shift_px)
        if np.any(shift):
            moved_values = _shift_phase_frames(state.values[left:right], shift)
            moved_method = _shift_phase_frames(state.method[left:right], shift)
            # A donor outside the allowed source domain can never become an observation.
            unavailable = (moved_method == EXCLUDED) & state.allowed[None]
            moved_values[unavailable] = np.nan
            moved_method[unavailable] = MISSING
            result.values[left:right, state.allowed] = moved_values[:, state.allowed]
            result.method[left:right, state.allowed] = moved_method[:, state.allowed]
        shifts.append(dict(start=left, end=right, dx=int(shift[0]), dy=int(shift[1])))
    result.receipts['interphase_registration'] = dict(parameters=asdict(config.registration), shifts=shifts)
    return result


def within_phase_recovery(state, config):
    result = state.copy()
    result.acquired_values = state.values.copy()
    result.acquired_valid = state.method == OBSERVED
    if config.recovery.enabled:
        recovered = recover_pixels(state.values, state.method == OBSERVED, state.raw.time_s,
                                   state.raw.phases, state.allowed, config.recovery)
        result.values = recovered.values
        result.method = recovered.methods
        result.receipts['within_phase_recovery'] = dict(
            parameters=asdict(config.recovery), counts=recovered.summary, steps=recovered.steps,
            spatial_radius=recovered.spatial_radius.tolist(), donor_count=recovered.donor_count.tolist())
    else:
        result.receipts['within_phase_recovery'] = dict(parameters=asdict(config.recovery), counts={})
    return result


def curated_repairs(state, config, curation=None):
    return apply_curation(state, config, curation)


def reference(state, config):
    result = state.copy()
    end = state.raw.phases[0] if state.raw.phases else len(state.values)
    anchors = np.where(state.method[:end] == OBSERVED, state.values[:end], np.nan)
    ranks = np.cumsum(np.isfinite(anchors), axis=0)
    baseline = _nanmedian(np.where(ranks <= config.reference.frames, anchors, np.nan))
    result.values[:, state.allowed] = (state.values[:, state.allowed] - baseline[state.allowed]) * 1000
    unavailable = ~np.isfinite(result.values) & state.allowed[None]
    result.method[unavailable] = MISSING
    result.receipts['reference'] = dict(parameters=asdict(config.reference), baseline_nm=baseline.tolist())
    return result


def background(state, config):
    result = state.copy()
    settings = config.background
    coordinates = []
    eligible = state.allowed.copy()
    if state.raw.cell_mask is not None:
        eligible &= ~state.raw.cell_mask.astype(bool)
    if settings.coordinates:
        for x, y in settings.coordinates:
            x, y = int(x), int(y)
            if 0 <= y < eligible.shape[0] and 0 <= x < eligible.shape[1] and eligible[y, x]:
                coordinates.append((x, y))
    elif settings.method != 'none':
        score = _nanmedian(np.where(state.method == OBSERVED, np.abs(state.values), np.nan))
        candidates = np.flatnonzero(eligible & np.isfinite(score) & (score < settings.threshold_pm))
        for index in candidates[np.argsort(score.ravel()[candidates], kind='stable')]:
            y, x = np.unravel_index(index, eligible.shape)
            if all(np.hypot(x - px, y - py) >= settings.spacing for px, py in coordinates):
                coordinates.append((int(x), int(y)))
            if len(coordinates) >= settings.n_pixels:
                break
    trace = np.zeros(len(state.values))
    if coordinates and settings.method != 'none':
        xs, ys = np.asarray(coordinates).T
        samples = np.where(state.method[:, ys, xs] == OBSERVED, state.values[:, ys, xs], np.nan)
        if settings.method == 'median':
            trace = _nanmedian(samples, axis=1)
        else:
            count = np.isfinite(samples).sum(axis=1)
            trace = np.divide(np.nansum(samples, axis=1), count,
                              out=np.full(len(samples), np.nan), where=count > 0)
        result.values[:, state.allowed] -= trace[:, None]
        result.method[~np.isfinite(result.values) & state.allowed[None]] = MISSING
    result.receipts['background'] = dict(parameters=asdict(settings), coordinates=coordinates,
                                        trace_pm=trace.tolist(), available=bool(coordinates))
    return result


def phase_offset(state, config):
    result = state.copy()
    settings = config.phase_offset
    maps = []
    bounds = [0, *state.raw.phases, len(state.values)]
    if settings.mode != 'none':
        for i in range(1, len(bounds) - 1):
            left, start, end = bounds[i - 1:i + 2]
            offset = np.full(state.allowed.shape, np.nan)
            midpoint = (state.raw.time_s[start - 1] + state.raw.time_s[start]) / 2
            for y, x in zip(*np.nonzero(state.allowed)):
                if settings.mode == 'single':
                    ids_before, ids_after = [start - 1], [start]
                else:
                    ids_before = np.flatnonzero(result.method[left:start, y, x] == OBSERVED) + left
                    ids_after = np.flatnonzero(result.method[start:end, y, x] == OBSERVED) + start
                    if min(len(ids_before), len(ids_after)) < settings.anchor_frames:
                        continue
                    ids_before = ids_before[-settings.anchor_frames:]
                    ids_after = ids_after[:settings.anchor_frames]
                before, after = result.values[ids_before, y, x], result.values[ids_after, y, x]
                if not np.isfinite(before).all() or not np.isfinite(after).all():
                    continue
                if settings.mode == 'single':
                    delta = before[0] - after[0]
                else:
                    pre = np.polyfit(state.raw.time_s[ids_before] - midpoint, before, 1)
                    post = np.polyfit(state.raw.time_s[ids_after] - midpoint, after, 1)
                    delta = pre[1] - post[1]
                offset[y, x] = delta
                result.values[start:end, y, x] += delta
            maps.append(dict(frame=start, offset_pm=offset.tolist(),
                             unsupported_pixels=int((state.allowed & ~np.isfinite(offset)).sum())))
    result.receipts['phase_offset'] = dict(parameters=asdict(settings), maps=maps)
    return result


def finalize(state, config, provenance=None):
    coverage = np.mean(state.method == OBSERVED, axis=0)
    receipts = dict(state.receipts)
    receipts['finalize'] = dict(observed_samples=int((state.method == OBSERVED).sum()))
    return PreprocessResult(state.values.copy(), state.method.copy(), coverage, receipts, provenance or {})
