"""Explicit reviewed intervals, never inferred automatically."""

from dataclasses import asdict
from pathlib import Path
import datetime
import hashlib
import json
import warnings

import numpy as np
import yaml

from ..epic_cardio.confirmed_pulse import repair_confirmed_pulse
from .config import CurationConfig
from .model import CURATED, OBSERVED, MISSING


def load_curation(source):
    if source is None:
        return []
    if isinstance(source, (str, Path)):
        with Path(source).open() as stream:
            source = yaml.safe_load(stream)
    if isinstance(source, dict):
        source = source.get('entries', [source])
    entries = []
    for entry in source or []:
        entry = dict(entry)
        required = {'well', 'start_frame', 'end_frame', 'reviewer', 'date', 'reason'}
        if required - entry.keys():
            raise ValueError(f'Curation missing fields: {sorted(required - entry.keys())}')
        if any(not entry[key] for key in ('well', 'reviewer', 'date', 'reason')):
            raise ValueError('Curation requires well, reviewer, date and reason')
        if isinstance(entry['date'], (datetime.date, datetime.datetime)):
            entry['date'] = entry['date'].isoformat()
        if not isinstance(entry['start_frame'], int) or not isinstance(entry['end_frame'], int):
            raise ValueError('Curation frame bounds must be integers')
        if entry.get('method', 'low_rank') not in ('low_rank', 'common_mode', 'per_pixel'):
            raise ValueError('Unknown curation method')
        if entry.get('action', 'confirm') not in ('confirm', 'add', 'exclude'):
            raise ValueError('Unknown curation action')
        if not 0 <= entry.get('min_correlation', 0.8) <= 1:
            raise ValueError('Curation min_correlation must be between zero and one')
        CurationConfig(**{key: entry[key] for key in ('max_rank', 'basis_variance', 'min_explained')
                          if key in entry})
        entries.append(entry)
    return entries


def curation_hash(curation):
    payload = json.dumps(load_curation(curation), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def pulse_shape_correlations(shapes):
    """Pairwise Pearson correlations of aligned per-well shapes in pm.

    Return a nested well-keyed dict. Constant shapes or fewer than two shared
    finite frames give None; callers must supply the same interval/frame order.
    """
    result = {}
    for well, shape in shapes.items():
        shape = np.asarray(shape, dtype=float)
        if shape.ndim != 1:
            raise ValueError('Pulse shapes must be one-dimensional')
        result[well] = {}
        for other, other_shape in shapes.items():
            other_shape = np.asarray(other_shape, dtype=float)
            if shape.shape != other_shape.shape:
                raise ValueError('Pulse shapes must have matching frame counts')
            finite = np.isfinite(shape) & np.isfinite(other_shape)
            a, b = shape[finite], other_shape[finite]
            supported = len(a) >= 2 and np.ptp(a) > 0 and np.ptp(b) > 0
            result[well][other] = float(np.corrcoef(a, b)[0, 1]) if supported else None
    return result


def repair_common_mode(state, result, evidence, valid, start, end, settings):
    """Remove a reviewed well pulse using original evidence and phase-local anchors."""
    clean_frames = settings['clean_frames']
    left, right = start - clean_frames, end + clean_frames
    if (clean_frames < settings['minimum_clean_frames']
            or not 0 <= left < start < end < right <= len(evidence)):
        raise ValueError('Complete clean windows are required')
    if any(left < int(p) < right for p in state.raw.phases):
        raise ValueError('Pulse and clean anchors cannot cross a phase boundary')
    anchors = np.r_[np.arange(left, start), np.arange(end, right)]
    time = state.raw.time_s - state.raw.time_s[start]
    design = np.column_stack([time[anchors], np.ones(len(anchors))])
    pulse_design = np.column_stack([time[start:end], np.ones(end - start)])
    complete = state.allowed & valid[anchors].all(axis=0)
    complete &= np.isfinite(evidence[anchors]).all(axis=0)
    complete &= np.isfinite(state.values[anchors]).all(axis=0)
    shape_pm = np.full(end - start, np.nan)
    amplitude = np.full(state.allowed.shape, np.nan)
    correlation = np.full(state.allowed.shape, np.nan)
    noise_limit_map = np.full(state.allowed.shape, np.nan)
    outcomes = np.full(state.allowed.shape, 'excluded', dtype='<U20')
    outcomes[state.allowed & ~complete] = 'no_anchor'
    repaired = np.zeros(state.allowed.shape, bool)
    max_residual = None
    if complete.any():
        # Fit only complete original anchors; no threshold or neighbor gate.
        fit = np.linalg.lstsq(design, evidence[anchors][:, complete], rcond=None)[0]
        line = pulse_design @ fit
        residual = (evidence[start:end, complete] - line) * 1000
        residual = np.where(valid[start:end, complete], residual, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            shape_pm = np.nanmedian(residual, axis=1)
        finite = np.isfinite(residual).all(axis=0)
        shape_energy = np.dot(shape_pm, shape_pm)
        a = np.full(complete.sum(), np.nan)
        c = np.full(complete.sum(), np.nan)
        if np.isfinite(shape_pm).all() and shape_energy > 0:
            a[finite] = shape_pm @ residual[:, finite] / shape_energy
            centered_shape = shape_pm - shape_pm.mean()
            centered = residual[:, finite] - residual[:, finite].mean(axis=0)
            denominator = np.linalg.norm(centered_shape) * np.linalg.norm(centered, axis=0)
            c[finite] = np.divide(centered_shape @ centered, denominator,
                                  out=np.zeros(finite.sum()), where=denominator > 0)
            c[finite] = np.clip(c[finite], -1, 1)
        amplitude[complete], correlation[complete] = a, c
        correlated = finite & (c >= settings['min_correlation'])
        post_fit = residual - shape_pm[:, None] * a[None]
        anchor_residual = (evidence[anchors][:, complete] - design @ fit) * 1000
        noise = settings['noise_scale'] * np.median(
            np.abs(anchor_residual - np.median(anchor_residual, axis=0)), axis=0)
        noise_limit = np.maximum(settings['noise_multiplier'] * noise, 1e-6)
        noise_limit_map[complete] = noise_limit
        within_noise = np.max(np.abs(post_fit), axis=0) <= noise_limit
        # Preserve the baseline of earlier step stitching while using original
        # acquisition residuals to identify the common component.
        corrected_fit = np.linalg.lstsq(design, state.values[anchors][:, complete], rcond=None)[0]
        corrected_line = pulse_design @ corrected_fit
        replacement = corrected_line.copy()
        keep_noise = correlated & within_noise
        replacement[:, keep_noise] += post_fit[:, keep_noise] / 1000
        unchanged = ~correlated & finite & (np.max(np.abs(residual), axis=0) <= settings['threshold_pm'])
        labels = np.full(complete.sum(), 'uncorrelated_pulse', dtype='<U20')
        labels[unchanged] = 'unchanged'
        labels[correlated] = 'anchor_line'
        labels[keep_noise] = 'common_component'
        outcomes[complete] = labels
        repaired[complete] = correlated
        result.values[start:end, repaired] = replacement[:, correlated]
        result.method[start:end, repaired] = CURATED
        # Automatic recovery must not overwrite a reviewed genuine transient.
        unchanged_map = outcomes == 'unchanged'
        result.values[start:end, unchanged_map] = evidence[start:end, unchanged_map]
        result.method[start:end, unchanged_map] = OBSERVED
        if correlated.any():
            max_residual = float(np.max(np.abs((replacement - corrected_line)[:, correlated])) * 1000)
    missing = np.isin(outcomes, ['no_anchor', 'uncorrelated_pulse'])
    result.values[start:end, missing] = np.nan
    result.method[start:end, missing] = MISSING
    counts = {name: int(np.sum(outcomes == name)) for name in
              ('common_component', 'anchor_line', 'unchanged', 'no_anchor', 'uncorrelated_pulse')}
    return dict(allowed_pixels=int(state.allowed.sum()), clean_anchor_pixels=int(complete.sum()),
                repaired_pixels=int(repaired.sum()), shape_pm=shape_pm,
                shape_amplitude_pm=float(np.max(np.abs(shape_pm))) if np.isfinite(shape_pm).all() else None,
                amplitude_map=amplitude, correlation_map=correlation, noise_limit_pm=noise_limit_map,
                outcomes=outcomes,
                counts=counts, max_residual_after_pm=max_residual)


def pulse_basis_similarities(bases):
    """Compare aligned row bases using principal angles, independent of sign/rotation.

    Similarity is sum(cos(angle)**2) / max(rank_a, rank_b), penalizing unmatched
    dimensions. Empty or nonfinite bases give None. Angles are in radians.
    """
    spaces = {}
    frame_count = None
    for well, basis in bases.items():
        basis = np.asarray(basis, dtype=float)
        if basis.ndim != 2:
            raise ValueError('Pulse bases must be two-dimensional (rank, frames)')
        if frame_count is not None and basis.shape[1] != frame_count:
            raise ValueError('Pulse bases must have matching frame counts')
        frame_count = basis.shape[1]
        spaces[well] = None
        if basis.size and np.isfinite(basis).all():
            _, singular, vectors = np.linalg.svd(basis, full_matrices=False)
            rank = np.sum(singular > singular[0] * max(basis.shape) * np.finfo(float).eps)
            if rank:
                spaces[well] = vectors[:rank]
    result = {}
    for well, space in spaces.items():
        result[well] = {}
        for other, other_space in spaces.items():
            comparison = None
            if space is not None and other_space is not None:
                cosines = np.clip(np.linalg.svd(space @ other_space.T, compute_uv=False), 0, 1)
                comparison = dict(principal_angles_rad=np.arccos(cosines), cosines=cosines,
                                  similarity=float(np.sum(cosines ** 2)
                                                   / max(len(space), len(other_space))))
            result[well][other] = comparison
    return result


def repair_low_rank(state, result, evidence, valid, start, end, settings, replace_interval=False):
    """Subtract a trimmed SVD pulse model, retaining all unmodelled residual signal."""
    clean_frames = settings['clean_frames']
    left, right = start - clean_frames, end + clean_frames
    if (clean_frames < settings['minimum_clean_frames']
            or not 0 <= left < start < end < right <= len(evidence)):
        raise ValueError('Complete clean windows are required')
    if any(left < int(p) < right for p in state.raw.phases):
        raise ValueError('Pulse and clean anchors cannot cross a phase boundary')
    anchors = np.r_[np.arange(left, start), np.arange(end, right)]
    time = state.raw.time_s - state.raw.time_s[start]
    design = np.column_stack([time[anchors], np.ones(len(anchors))])
    pulse_design = np.column_stack([time[start:end], np.ones(end - start)])
    complete = state.allowed & valid[anchors].all(axis=0)
    complete &= np.isfinite(evidence[anchors]).all(axis=0)
    complete &= np.isfinite(state.values[anchors]).all(axis=0)
    basis = np.empty((0, end - start))
    singular = np.empty(0)
    variance = np.empty(0)
    fraction_map = np.full(state.allowed.shape, np.nan)
    norm_map = np.full(state.allowed.shape, np.nan)
    post_norm_map = np.full(state.allowed.shape, np.nan)
    basis_fit_mask = np.zeros(state.allowed.shape, bool)
    outcomes = np.full(state.allowed.shape, 'excluded', dtype='<U20')
    outcomes[state.allowed & ~complete] = 'no_anchor'
    coefficient_maps = np.empty((0, *state.allowed.shape))
    repaired = np.zeros(state.allowed.shape, bool)
    partial_rejections = (np.full(state.allowed.shape, 'not_partial', dtype='<U24')
                          if replace_interval else None)
    trimmed_count = 0
    max_residual = None
    if complete.any():
        fit = np.linalg.lstsq(design, evidence[anchors][:, complete], rcond=None)[0]
        residual = ((evidence[start:end, complete] - pulse_design @ fit) * 1000).T
        finite = valid[start:end, complete].all(axis=0) & np.isfinite(residual).all(axis=1)
        if replace_interval:
            partial_rejections[complete] = np.where(finite, 'not_partial', 'no_basis')
        norms = np.linalg.norm(residual[finite], axis=1)
        fit_indices = np.flatnonzero(finite)
        if len(fit_indices):
            # Exclude the largest 1% (rounded up), retaining at least one donor.
            trimmed_count = min(int(np.ceil(0.01 * len(fit_indices))), len(fit_indices) - 1)
            order = np.argsort(norms, kind='stable')
            fit_indices = fit_indices[order[:len(order) - trimmed_count]]
            local_fit_mask = np.zeros(complete.sum(), bool)
            local_fit_mask[fit_indices] = True
            basis_fit_mask[complete] = local_fit_mask
            _, singular, vectors = np.linalg.svd(residual[fit_indices], full_matrices=False)
            energy = singular ** 2
            if energy.sum() > 0:
                variance = energy / energy.sum()
                numerical_rank = int(np.sum(singular > singular[0]
                                            * max(len(fit_indices), end - start) * np.finfo(float).eps))
                rank = min(settings['max_rank'], numerical_rank,
                           int(np.searchsorted(np.cumsum(variance), settings['basis_variance'])) + 1)
                basis = vectors[:rank].copy()
                # Fix arbitrary SVD signs for interpretable, repeatable receipts.
                pivots = np.argmax(np.abs(basis), axis=1)
                basis *= np.where(basis[np.arange(rank), pivots] < 0, -1, 1)[:, None]
            else:
                variance = np.zeros_like(energy)
        coefficients = np.full((complete.sum(), len(basis)), np.nan)
        coefficients[finite] = residual[finite] @ basis.T
        post_fit = np.full_like(residual, np.nan)
        post_fit[finite] = residual[finite] - coefficients[finite] @ basis
        fractions = np.full(complete.sum(), np.nan)
        energy = np.sum(residual[finite] ** 2, axis=1)
        fractions[finite] = np.clip(1 - np.divide(np.sum(post_fit[finite] ** 2, axis=1), energy,
                                                 out=np.zeros_like(energy), where=energy > 0), 0, 1)
        fraction_map[complete] = fractions
        norm_map[complete] = np.where(finite, np.linalg.norm(residual, axis=1), np.nan)
        post_norm_map[complete] = np.linalg.norm(post_fit, axis=1)
        coefficient_maps = np.full((len(basis), *state.allowed.shape), np.nan)
        coefficient_maps[:, complete] = coefficients.T
        near_noise = finite & (np.max(np.abs(residual), axis=1) <= settings['threshold_pm'])
        accepted = finite & (fractions >= settings['min_explained'])
        partial_accepted = np.zeros(len(accepted), bool)
        if replace_interval and len(basis):
            # An event replaces the interval with its anchor line. A dropout inside
            # it need not invalidate all frames when the remaining measurements
            # independently identify the learned instrument component.
            measured = valid[start:end, complete].T & np.isfinite(residual)
            anchor_residual = (evidence[anchors][:, complete] - design @ fit) * 1000
            quiet = np.max(np.abs(anchor_residual), axis=0) <= settings['threshold_pm']
            partial_reason = np.full(len(accepted), 'unexplained_event', dtype='<U24')
            partial_reason[~quiet] = 'anchor_variation'
            duration = state.raw.time_s[end] - state.raw.time_s[start - 1]
            no_ramp = np.abs(fit[0]) * duration * 1000 <= settings['threshold_pm']
            partial_reason[~no_ramp] = 'anchor_ramp'
            quiet &= no_ramp
            patterns, groups = np.unique(measured, axis=0, return_inverse=True)
            for group, known in enumerate(patterns):
                pixels = (groups == group) & ~finite & quiet
                if not pixels.any():
                    continue
                if known.sum() < max(len(basis) + 1, settings['minimum_clean_frames']):
                    partial_reason[pixels] = 'insufficient_samples'
                    continue
                design_known = basis[:, known].T
                if np.linalg.matrix_rank(design_known) < len(basis):
                    partial_reason[pixels] = 'nonidentifiable_basis'
                    continue
                coefficients_partial = np.linalg.lstsq(design_known, residual[pixels][:, known].T,
                                                       rcond=None)[0].T
                remaining = residual[pixels][:, known] - coefficients_partial @ basis[:, known]
                energy_partial = np.sum(residual[pixels][:, known] ** 2, axis=1)
                explained = 1 - np.divide(np.sum(remaining ** 2, axis=1), energy_partial,
                                           out=np.ones_like(energy_partial), where=energy_partial > 0)
                partial_accepted[pixels] = explained >= settings['min_explained']
                fractions[pixels] = np.clip(explained, 0, 1)
                coefficients[pixels] = coefficients_partial
                local_norms = norm_map[complete]
                local_norms[pixels] = np.sqrt(energy_partial)
                norm_map[complete] = local_norms
                local_post_norms = post_norm_map[complete]
                local_post_norms[pixels] = np.linalg.norm(remaining, axis=1)
                post_norm_map[complete] = local_post_norms
            accepted |= partial_accepted
            partial_reason[partial_accepted] = 'accepted'
            partial_reason[finite] = 'not_partial'
            partial_rejections[complete] = partial_reason
            fraction_map[complete] = fractions
            coefficient_maps[:, complete] = coefficients.T
        if not replace_interval:
            accepted |= near_noise
        unexplained = 'unexplained_event' if replace_interval else 'unexplained_pulse'
        labels = np.full(complete.sum(), unexplained, dtype='<U20')
        labels[accepted] = 'low_rank_component'
        labels[partial_accepted] = 'partial_event'
        labels[accepted & (fractions < settings['min_explained'])] = 'near_noise'
        outcomes[complete] = labels
        repaired[complete] = accepted
        corrected_fit = np.linalg.lstsq(design, state.values[anchors][:, complete], rcond=None)[0]
        corrected_line = pulse_design @ corrected_fit
        replacement = corrected_line if replace_interval else corrected_line + post_fit.T / 1000
        result.values[start:end, repaired] = replacement[:, accepted]
        result.method[start:end, repaired] = CURATED
        if accepted.any():
            max_residual = float(np.max(np.abs((replacement - corrected_line)[:, accepted])) * 1000)
    unexplained = 'unexplained_event' if replace_interval else 'unexplained_pulse'
    missing = np.isin(outcomes, ['no_anchor', unexplained])
    result.values[start:end, missing] = np.nan
    result.method[start:end, missing] = MISSING
    counts = {name: int(np.sum(outcomes == name)) for name in
              ('low_rank_component', 'near_noise', 'no_anchor', unexplained,
               *(['partial_event'] if replace_interval else []))}
    return dict(allowed_pixels=int(state.allowed.sum()), clean_anchor_pixels=int(complete.sum()),
                repaired_pixels=int(repaired.sum()), chosen_rank=len(basis), singular_values_pm=singular,
                explained_variance=variance, basis_vectors=basis, coefficient_maps=coefficient_maps,
                explained_fraction_map=fraction_map, residual_norm_pm=norm_map,
                post_fit_residual_norm_pm=post_norm_map, basis_fit_mask=basis_fit_mask,
                basis_fit_pixels=int(basis_fit_mask.sum()), basis_trimmed_pixels=trimmed_count,
                outcomes=outcomes, counts=counts, max_residual_after_pm=max_residual,
                **(dict(partial_event_rejections=partial_rejections) if replace_interval else {}))


def apply_curation(state, config, curation):
    result = state.copy()
    receipts = []
    # Every interval uses original observed evidence, never another repair's anchors.
    evidence = state.values if state.acquired_values is None else state.acquired_values
    valid = state.method == OBSERVED if state.acquired_valid is None else state.acquired_valid
    for entry in load_curation(curation):
        if entry['well'] != state.raw.well:
            continue
        settings = asdict(config.curation)
        settings['clean_frames'] = entry.get('clean_frames', settings['clean_frames'])
        settings['threshold_pm'] = entry.get('threshold_pm', entry.get('threshold', settings['threshold_pm']))
        settings['method'] = entry.get('method', settings['method'])
        settings['min_correlation'] = entry.get('min_correlation', settings['min_correlation'])
        for key in ('max_rank', 'basis_variance', 'min_explained'):
            settings[key] = entry.get(key, settings[key])
        start, end = entry['start_frame'], entry['end_frame']
        if settings['method'] == 'low_rank':
            receipt = repair_low_rank(state, result, evidence, valid, start, end, settings)
            receipts.append(dict(entry=entry, parameters=settings, **receipt))
            continue
        if settings['method'] == 'common_mode':
            receipt = repair_common_mode(state, result, evidence, valid, start, end, settings)
            receipts.append(dict(entry=entry, parameters=settings, **receipt))
            continue
        per_pixel_settings = {key: value for key, value in settings.items()
                              if key not in ('method', 'min_correlation', 'max_rank',
                                             'basis_variance', 'min_explained')}
        values, accepted, receipt = repair_confirmed_pulse(
            evidence, valid, state.raw.time_s, state.raw.phases, state.allowed, start, end,
            **per_pixel_settings)
        # Detection uses original observations. Correction honors any earlier stitched baseline.
        anchors = np.isfinite(state.values[start - 1]) & np.isfinite(state.values[end])
        accepted &= anchors
        fraction = ((state.raw.time_s[start:end] - state.raw.time_s[start - 1])
                    / (state.raw.time_s[end] - state.raw.time_s[start - 1]))
        interpolation = (state.values[start - 1][None]
                         + fraction[:, None, None] * (state.values[end] - state.values[start - 1])[None])
        result.values[start:end, accepted] = interpolation[:, accepted]
        result.method[start:end, accepted] = CURATED
        receipt['repaired_pixels'] = int(accepted.sum())
        receipts.append(dict(entry=entry, parameters=settings, **receipt))
    result.receipts['curated_repairs'] = receipts
    return result
