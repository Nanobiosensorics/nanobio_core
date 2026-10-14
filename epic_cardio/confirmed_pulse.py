"""Opt-in repair of a reviewed returning acquisition pulse with clean anchors."""

import numpy as np
from scipy import ndimage


def repair_confirmed_pulse(values, valid, time, phases, allowed, start, end,
                           clean_frames=6, threshold_pm=75.0, minimum_neighbors=2):
    """Interpolate [start, end) only where original observations confirm a return.

    Values are nm and time is seconds. The interval must be reviewed externally;
    this does not extend automatic short-gap recovery or infer a physical cause.
    Both clean windows must remain in one phase and agree with a local trend.
    Repaired samples never supply anchors or neighbor evidence.
    """
    raw = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(raw)
    time = np.asarray(time, dtype=float)
    allowed = np.asarray(allowed, dtype=bool)
    if raw.ndim != 3 or valid.shape != raw.shape or allowed.shape != raw.shape[1:]:
        raise ValueError('Expected matching time/spatial arrays')
    if len(time) != len(raw) or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError('Acquisition times must increase strictly')
    left, right = start - clean_frames, end + clean_frames
    if clean_frames < 3 or not 0 <= left < start < end < right <= len(raw):
        raise ValueError('Complete clean windows are required')
    if any(left < int(p) < right for p in phases):
        raise ValueError('Pulse and clean anchors cannot cross a phase boundary')
    anchors = np.r_[np.arange(left, start), np.arange(end, right)]
    offsets = time - time[start]
    design = np.column_stack([offsets[anchors], np.ones(len(anchors))])
    fit = np.linalg.lstsq(design, raw[anchors].reshape(len(anchors), -1), rcond=None)[0]
    predicted = (offsets[:, None] * fit[0] + fit[1]).reshape(raw.shape)
    residual = (raw - predicted) * 1000
    anchor_residual = residual[anchors]
    noise = 1.4826 * np.median(np.abs(anchor_residual - np.median(anchor_residual, axis=0)), axis=0)
    tolerance = np.maximum(threshold_pm, 6 * noise)
    clean = allowed & valid[anchors].all(axis=0)
    clean &= np.max(np.abs(anchor_residual), axis=0) <= tolerance
    # Independent pre/post fits distinguish returning pulses from real steps.
    pre = np.polyfit(offsets[left:start], raw[left:start].reshape(clean_frames, -1), 1)
    post = np.polyfit(offsets[end:right], raw[end:right].reshape(clean_frames, -1), 1)
    midpoint = (offsets[start] + offsets[end - 1]) / 2
    disagreement = np.abs((midpoint * (pre[0] - post[0]) + pre[1] - post[1]) * 1000)
    clean &= disagreement.reshape(allowed.shape) <= tolerance
    excursion = np.any(valid[start:end] & (np.abs(residual[start:end]) > tolerance), axis=0)
    candidates = clean & excursion
    neighbor_count = ndimage.convolve(candidates.astype(int), np.ones((3, 3), dtype=int),
                                     mode='constant', cval=0)
    neighbor_count -= candidates.astype(int)
    accepted = candidates & (neighbor_count >= minimum_neighbors)
    result = raw.copy()
    fractions = (time[start:end] - time[start - 1]) / (time[end] - time[start - 1])
    interpolation = raw[start - 1][None] + fractions[:, None, None] * (raw[end] - raw[start - 1])[None]
    result[start:end, accepted] = interpolation[:, accepted]
    receipt = dict(allowed_pixels=int(allowed.sum()), clean_anchor_pixels=int(clean.sum()),
                   excursion_pixels=int(excursion.sum()), candidate_pixels=int(candidates.sum()),
                   repaired_pixels=int(accepted.sum()),
                   rejected_neighbor_pixels=int((candidates & ~accepted).sum()))
    return result, accepted, receipt
