"""Reordered recovery: immutable evidence, joint labels, then disjoint fixes.

All recovery arithmetic is phase-local and in nm; thresholds/receipts are pm.
Time loops operate on whole pixel batches, never on individual pixel traces.
"""

from dataclasses import asdict

import numpy as np
import bottleneck as bn
from scipy import ndimage

from ..epic_cardio.data_correction import _phase_translation, _shift_phase_frames
from ..epic_cardio.math_ops import find_local_maxima
from ..processing.active_area import active_area
from . import stages
from .curation import repair_low_rank
from .model import StageState, OBSERVED, TEMPORAL, SPATIAL, STITCHED, CURATED, MISSING, EXCLUDED
from .recovery import _runs, fill_spatial_frame

STABLE, GAP, PULSE, STEP, DROPOUT, UNRESOLVED = range(6)
LABEL_NAMES = ('stable', 'gap', 'pulse', 'step', 'dropout', 'unresolved')


def _bounds(state):
    return [0, *state.raw.phases, len(state.values)]


def sample_mask(raw, config):
    values = np.asarray(raw.values_nm, dtype=np.float64).copy()
    _, detected, threshold = active_area(values[0], config.active_area.margin_px)
    allowed = raw.allowed & detected
    valid = (raw.codes != 0) & np.isfinite(values) & allowed[None]
    methods = np.where(valid, OBSERVED, MISSING).astype(np.uint8)
    methods[:, ~allowed] = EXCLUDED
    values[methods == MISSING] = np.nan
    suspect = np.zeros(values.shape, bool)
    bounds = [0, *raw.phases, len(values)]
    for left, right in zip(bounds[:-1], bounds[1:]):
        suspect[left + 1:right] = (np.abs(np.diff(values[left:right], axis=0)) * 1000
                                   > config.validity.max_jump_pm)
    receipts = dict(sample_mask=dict(missing_samples=int((methods == MISSING).sum()),
                                    suspect=suspect, active_threshold=threshold,
                                    allowed_pixels=int(allowed.sum()), phase_source=raw.phase_source))
    return StageState(raw, values, methods, allowed, receipts, acquired_codes=raw.codes.copy())


def register(state, config):
    result = state.copy()
    result.acquired_codes = state.acquired_codes.copy()
    suspect = state.receipts['sample_mask']['suspect'].copy()
    bounds = _bounds(state)
    points = []
    if config.registration.enabled and len(bounds) > 2:
        baseline = bn.nanmedian(state.values[:config.reference.frames], axis=0)
        for left, right in zip(bounds[:-1], bounds[1:]):
            signal = (state.values[left:right] - baseline) * 1000
            projection = bn.nanmedian(signal, axis=0)
            projection = np.where(state.allowed & np.isfinite(projection), projection,
                                  config.registration.threshold_pm - 1)
            xs, ys = find_local_maxima(projection, config.registration.threshold_pm,
                                      np.max(projection), config.registration.neighborhood_size,
                                      ~state.allowed)
            points.append(np.column_stack([xs, ys]))
    reference = int(np.argmax([len(p) for p in points])) if points else 0
    shifts = []
    for i, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
        shift = (_phase_translation(points[i], points[reference], config.registration.max_shift_px)
                 if points and i != reference else np.zeros(2, dtype=int))
        if np.any(shift):
            values = _shift_phase_frames(state.values[left:right], shift).copy()
            methods = _shift_phase_frames(state.method[left:right], shift).copy()
            codes = _shift_phase_frames(state.acquired_codes[left:right], shift).copy()
            moved_suspect = _shift_phase_frames(suspect[left:right], shift)
            unsupported = (methods == EXCLUDED) & state.allowed[None]
            values[unsupported], methods[unsupported], codes[unsupported] = np.nan, MISSING, 0
            result.values[left:right, state.allowed] = values[:, state.allowed]
            result.method[left:right, state.allowed] = methods[:, state.allowed]
            result.acquired_codes[left:right, state.allowed] = codes[:, state.allowed]
            suspect[left:right, state.allowed] = moved_suspect[:, state.allowed]
        shifts.append(dict(start=left, end=right, dx=int(shift[0]), dy=int(shift[1])))
    result.receipts['sample_mask'] = dict(state.receipts['sample_mask'], suspect=suspect)
    result.receipts['interphase_registration'] = dict(parameters=asdict(config.registration),
                                                     projection='median', shifts=shifts)
    result.acquired_values = result.values.copy()
    result.acquired_valid = result.method == OBSERVED
    return result


def _median(values, radius, minimum):
    padded = np.pad(values, ((radius, radius), (0, 0), (0, 0)), constant_values=np.nan)
    width = 2 * radius + 1
    return bn.move_median(padded, window=width, min_count=minimum, axis=0)[width - 1:]


def _baseline(values, valid, blocked, config):
    """Two passes; estimate noise only after removing first-pass contamination."""
    radius = max(config.recovery.median_radius, config.v2.max_pulse_frames)
    measured = np.where(valid & ~blocked[:, None, None], values, np.nan)
    first = _median(measured, radius, config.recovery.minimum_observations)
    residual = (measured - first) * 1000
    differences = np.diff(measured, axis=0) * 1000
    center = bn.nanmedian(differences, axis=0)
    noise = config.recovery.noise_scale * bn.nanmedian(np.abs(differences - center), axis=0) / np.sqrt(2)
    limit = np.maximum(config.recovery.threshold_pm, config.recovery.noise_multiplier * noise)
    outliers = np.abs(residual) > limit
    # Remove both sides of sharp changes so that neither edge inflates MAD.
    edges = np.abs(differences - center) > limit
    outliers[1:] |= edges
    outliers[:-1] |= edges
    clean = np.where(outliers, np.nan, measured)
    baseline = _median(clean, radius, config.recovery.minimum_observations)
    residual = (clean - baseline) * 1000
    center = bn.nanmedian(residual, axis=0)
    noise = config.recovery.noise_scale * bn.nanmedian(np.abs(residual - center), axis=0)
    noise = np.where(np.isfinite(noise), noise, 0)
    return baseline, np.maximum(config.recovery.threshold_pm, config.recovery.noise_multiplier * noise)


def _well_returning_intervals(values, valid, time, allowed, limits, config):
    """Pair well-wide changes so long intervals cannot self-mask in a median."""
    trace = bn.nanmedian(values[:, allowed], axis=1)
    changes = np.diff(values, axis=0) * 1000
    fraction = ((np.abs(changes) > limits) & valid[1:] & valid[:-1]).sum(axis=(1, 2))
    fraction = fraction / max(allowed.sum(), 1)
    candidates = np.flatnonzero(fraction >= config.v2.event_fraction) + 1
    covered = np.zeros(len(values), bool)
    intervals = []
    count = config.curation.clean_frames
    if count < config.curation.minimum_clean_frames:
        return intervals
    for a in candidates:
        if covered[a] or a < count:
            continue
        # Aggregate evidence proposes intervals; per-pixel anchor lines qualify them below.
        offsets = time[a - count:a] - time[a]
        if not np.isfinite(trace[a - count:a]).all():
            continue
        slope, level = np.polyfit(offsets, trace[a - count:a], 1)
        for b in candidates[candidates > a]:
            if b + count > len(values):
                break
            predicted = level + slope * (time[b:b + count] - time[a])
            if np.max(np.abs(trace[b:b + count] - predicted)) * 1000 > config.recovery.threshold_pm:
                continue
            intervals.append((int(a), int(b)))
            covered[a:b + 1] = True
            break
    return intervals


def _phase_events(state, config, left, right):
    values, valid = state.values[left:right], state.acquired_valid[left:right]
    baseline, limits = _baseline(values, valid, np.zeros(right - left, bool), config)
    residual = (values - baseline) * 1000
    median = bn.nanmedian(residual[:, state.allowed], axis=1)
    center = bn.nanmedian(median)
    mad = config.recovery.noise_scale * bn.nanmedian(np.abs(median - center))
    z = np.abs(median - center) / max(float(mad), 1e-6)
    fraction = ((np.abs(residual) > limits) & valid).sum(axis=(1, 2)) / max(state.allowed.sum(), 1)
    detected = (fraction >= config.v2.event_fraction) & (z >= config.v2.event_z)
    for a, b in _well_returning_intervals(values, valid, state.raw.time_s[left:right],
                                        state.allowed, limits, config):
        detected[a:b] = True
    return baseline, limits, median - center, center, mad, fraction, detected


def _extend_edges(proposed, excluded, median, threshold, maximum, fixed_intervals=()):
    extended = proposed.copy()
    for a, b in _runs(proposed):
        # Shared bounds already union the individually extended intervals. Extending
        # them again would exceed the per-side cap and break plate alignment.
        if any(a < end and b > start for start, end in fixed_intervals):
            continue
        for side in (-1, 1):
            frame = int(a - 1 if side == -1 else b)
            for _ in range(maximum):
                if (not 0 <= frame < len(proposed) or excluded[frame]
                        or not np.isfinite(median[frame]) or abs(median[frame]) <= threshold):
                    break
                extended[frame] = True
                frame += side
    return extended


def detect_event_bounds(raw, config, curation):
    """Bounded-memory first pass; never repair samples while gathering plate evidence."""
    state = register(sample_mask(raw, config), config)
    events = []
    bounds = _bounds(state)
    for left, right in zip(bounds[:-1], bounds[1:]):
        _, limits, median, _, _, _, detected = _phase_events(state, config, left, right)
        proposed = detected.copy()
        excluded = np.zeros(right - left, bool)
        for entry in curation:
            if entry['well'] != raw.well:
                continue
            a, b = entry['start_frame'], entry['end_frame']
            if not 0 <= a < b <= len(raw.values_nm) or any(a < p < b for p in raw.phases):
                raise ValueError('Curation interval outside acquisition or crossing a phase boundary')
            target = excluded if entry.get('action') == 'exclude' else proposed
            target[max(a - left, 0):max(min(b, right) - left, 0)] = True
        proposed &= ~excluded
        proposed = _extend_edges(proposed, excluded, median, bn.nanmedian(limits[state.allowed]),
                                 config.v2.max_edge_extension)
        for a, b in _runs(proposed):
            detected_bounds = [[int(left + x), int(left + y)] for x, y in _runs(detected)
                               if x < b and y > a]
            events.append(dict(phase=[left, right], bounds=[int(left + a), int(left + b)],
                               detected_bounds=detected_bounds))
    return events


def plate_consensus(proposals, config, phase_bounds=None):
    """Align overlapping same-phase events; reject robust endpoint outliers first."""
    plans = {well: [] for well in proposals}
    remaining = [(well, event) for well, events in proposals.items() for event in events
                 if event['detected_bounds']]
    while remaining:
        group = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            for item in remaining[:]:
                event = item[1]
                if any(event['phase'] == other['phase'] and event['bounds'][0] < other['bounds'][1]
                       and event['bounds'][1] > other['bounds'][0] for _, other in group):
                    group.append(item)
                    remaining.remove(item)
                    changed = True
        endpoints = np.asarray([event['bounds'] for _, event in group])
        center = np.median(endpoints, axis=0)
        mad = np.median(np.abs(endpoints - center), axis=0)
        # A one-frame quantization floor admits normal edge uncertainty, not long tails.
        keep = (np.abs(endpoints - center) <= np.maximum(2, 3 * 1.4826 * mad)).all(axis=1)
        retained = [item for item, good in zip(group, keep) if good]
        voters = {well for well, _ in retained}
        if len(voters) / max(len(proposals), 1) < config.v2.consensus_fraction:
            continue
        shared = [min(e['bounds'][0] for _, e in retained), max(e['bounds'][1] for _, e in retained)]
        receipt = dict(bounds=shared, phase=group[0][1]['phase'],
                       per_well_bounds={well: event for well, event in group},
                       outlier_wells=sorted({well for (well, _), good in zip(group, keep) if not good}),
                       supporting_wells=sorted(voters))
        # Undetected wells receive consensus too; only matching acquisition phases qualify.
        for well in proposals:
            if phase_bounds is not None and receipt['phase'] not in phase_bounds[well]:
                continue
            receipt['per_well_bounds'].setdefault(
                well, dict(phase=receipt['phase'], bounds=None, detected_bounds=[]))
            plans[well].append(receipt)
    return plans


def well_events(state, config, curation, event_plan=None):
    result = state.copy()
    result.recovery_baselines = {}
    excluded = np.zeros(len(state.values), bool)
    curated = np.zeros(len(state.values), bool)
    entries = [entry for entry in curation if entry['well'] == state.raw.well]
    bounds = _bounds(state)
    for entry in entries:
        a, b = entry['start_frame'], entry['end_frame']
        if not 0 <= a < b <= len(state.values):
            raise ValueError('Curation interval outside acquisition')
        if any(a < p < b for p in state.raw.phases):
            raise ValueError('V2 events cannot cross a phase boundary')
        (excluded if entry.get('action') == 'exclude' else curated)[a:b] = True
    blocked = np.zeros(len(state.values), bool)
    receipts = []
    for left, right in zip(bounds[:-1], bounds[1:]):
        values, valid = state.values[left:right], state.acquired_valid[left:right]
        baseline, limits, median, center, mad, fraction, detected = _phase_events(state, config, left, right)
        proposed = (detected | curated[left:right]) & ~excluded[left:right]
        consensus = [e for e in event_plan or [] if e['phase'] == [left, right]]
        for event in consensus:
            # Replace the local automatic bounds, including a rejected outlier's tail.
            for a, b in event['per_well_bounds'][state.raw.well]['detected_bounds']:
                proposed[a - left:b - left] = False
            a, b = event['bounds']
            proposed[a - left:b - left] = True
        proposed |= curated[left:right]
        proposed &= ~excluded[left:right]
        proposed = _extend_edges(proposed, excluded[left:right], median, bn.nanmedian(limits[state.allowed]),
                                 config.v2.max_edge_extension,
                                 [(e['bounds'][0] - left, e['bounds'][1] - left) for e in consensus])
        if not proposed.any() and not excluded[left:right].any():
            result.recovery_baselines[left, right] = (baseline, limits)
        for a, b in _runs(proposed):
            start, end = int(left + a), int(left + b)
            matching_consensus = [e for e in consensus if e['bounds'][0] < end and e['bounds'][1] > start]
            settings = asdict(config.curation)
            overlapping = [e for e in entries if e.get('action') != 'exclude'
                           and e['start_frame'] < end and e['end_frame'] > start]
            for entry in overlapping:
                for key in settings:
                    if key in entry:
                        settings[key] = entry[key]
            clean = settings['clean_frames']
            anchor_frames = np.r_[np.arange(start - clean, start), np.arange(end, end + clean)]
            supported = (start - clean >= left and end + clean <= right
                         and clean >= settings['minimum_clean_frames'])
            if supported:
                # Other proposed/excluded intervals cannot supply event anchors.
                supported = not (proposed[anchor_frames - left] | excluded[anchor_frames]).any()
            if supported:
                complete = valid[anchor_frames - left].all(axis=0) & state.allowed
                offsets = state.raw.time_s[anchor_frames] - state.raw.time_s[start]
                design = np.column_stack([offsets, np.ones(len(offsets))])
                pulse_time = state.raw.time_s[start:end] - state.raw.time_s[start]
                fit = np.linalg.lstsq(design, state.values[anchor_frames][:, complete], rcond=None)[0]
                line = np.column_stack([pulse_time, np.ones(end - start)]) @ fit
                residual_pm = (state.values[start:end, complete] - line) * 1000
                fraction[a:b] = ((np.abs(residual_pm) > limits[complete])
                                 & valid[a:b, complete]).sum(axis=1) / max(state.allowed.sum(), 1)
                anchor_median = bn.nanmedian(residual_pm, axis=1) if complete.any() else np.zeros(end - start)
                confirmed = ((fraction[a:b] >= config.v2.event_fraction)
                             & (np.abs(anchor_median - center) >= config.v2.event_z * max(float(mad), 1e-6)))
                if not overlapping and not matching_consensus and not confirmed.any():
                    continue
                receipt = repair_low_rank(state, result, state.acquired_values, state.acquired_valid,
                                          start, end, settings, replace_interval=True)
            else:
                result.values[start:end, state.allowed] = np.nan
                result.method[start:end, state.allowed] = MISSING
                outcomes = np.full(state.allowed.shape, 'excluded', dtype='<U20')
                outcomes[state.allowed] = 'no_anchor'
                receipt = dict(outcomes=outcomes, counts=dict(no_anchor=int(state.allowed.sum())),
                               repaired_pixels=0, chosen_rank=0, max_residual_after_pm=None)
            blocked[start:end] = True
            receipts.append(dict(start_frame=start, end_frame=end,
                                 detected_bounds=[[int(left + x), int(left + y)] for x, y in _runs(detected)
                                                  if x < b and y > a],
                                 final_bounds=[start, end],
                                 consensus=matching_consensus,
                                 fraction_pixels=float(np.max(fraction[a:b])),
                                 fraction_by_frame=fraction[a:b].copy(),
                                 source=('curated' if overlapping else
                                         ('consensus' if matching_consensus else 'detected')),
                                 detected=bool(detected[a:b].any()), parameters=settings, **receipt))
    result.receipts['well_events'] = dict(events=receipts, intervals=blocked, excluded_intervals=excluded,
                                         parameters=asdict(config.v2))
    return result


def _fit(values, time, target):
    """OLS against actual timestamps for all pixels; complete anchors checked by caller."""
    offsets = time - target
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    safe = np.where(finite, values, 0)
    mean_time = np.divide(np.sum(offsets[:, None, None] * finite, axis=0), count,
                          out=np.zeros(values.shape[1:]), where=count > 0)
    centered = offsets[:, None, None] - mean_time
    denominator = np.sum(centered ** 2 * finite, axis=0)
    slope = np.divide(np.sum(centered * safe, axis=0), denominator,
                      out=np.zeros(values.shape[1:]), where=denominator > 0)
    level = np.divide(safe.sum(axis=0), count, out=np.full(values.shape[1:], np.nan), where=count > 0)
    intercept = level - slope * mean_time
    return slope, intercept


def label_phase(values, valid, time, blocked, allowed, config, estimates=None):
    """Joint segment decisions on original evidence, before any value correction."""
    n = len(values)
    labels = np.where(valid, STABLE, UNRESOLVED).astype(np.uint8)
    baseline, limit = estimates if estimates is not None else _baseline(values, valid, blocked, config)
    difference = np.diff(values, axis=0) * 1000
    typical = bn.nanmedian(difference, axis=0)
    edges = np.zeros(values.shape, bool)
    edges[1:] = (np.abs(difference - typical) > limit) & valid[1:] & valid[:-1]
    edges[blocked] = False
    edges[1:] &= ~blocked[:-1, None, None]
    taken = np.broadcast_to(blocked[:, None, None], values.shape).copy()
    missing = np.concatenate([np.zeros((1, *allowed.shape), int), np.cumsum(~valid, axis=0)])
    window = config.recovery.anchor_frames
    # Pair changes with a stable return, including returns beyond the repair cap.
    # Long returning segments become unresolved once, never two independent steps.
    for a in np.flatnonzero(edges.any(axis=(1, 2))):
        if a < window or blocked[a - window:a].any():
            continue
        pending = edges[a] & ~taken[a] & valid[a - window:a].all(axis=0) & allowed
        if not pending.any():
            continue
        slope, intercept = _fit(values[a - window:a], time[a - window:a], time[a])
        for b in np.flatnonzero(edges[a + 1:].any(axis=(1, 2))) + a + 1:
            if b + window > n or blocked[a:b + window].any():
                continue
            supported = (missing[b + window] == missing[a - window]) & pending & edges[b]
            post = values[b:b + window]
            expected = intercept + (time[b:b + window] - time[a])[:, None, None] * slope
            returns = np.max(np.abs(post - expected), axis=0) * 1000 <= limit
            opposite = difference[a - 1] * difference[b - 1] < 0
            accepted = supported & returns & opposite
            if accepted.any():
                label = PULSE if b - a <= config.v2.max_pulse_frames else UNRESOLVED
                labels[a:b, accepted] = label
                taken[a:b + 1, accepted] = True
                pending &= ~accepted
            if not pending.any():
                break
    # Invalid runs are segmented across pixels with a time-state machine.
    starts = np.full(allowed.shape, -1, int)
    for b in range(n + 1):
        bad = ~valid[b] if b < n else np.zeros(allowed.shape, bool)
        opening = bad & (starts < 0)
        starts[opening] = b
        closing = ~bad & (starts >= 0) & allowed
        for a in np.unique(starts[closing]):
            pixels = closing & (starts == a)
            if (b == n and a > 0 and b - a >= config.recovery.tail_frames):
                pixels &= valid[:a].sum(axis=0) >= config.recovery.minimum_observations
                labels[a:b, pixels] = DROPOUT
            elif (a > 0 and b < n and b - a <= config.recovery.temporal_gap
                  and not blocked[a - 1:b + 1].any()):
                pixels &= valid[a - 1] & valid[b]
                labels[a:b, pixels] = GAP
        starts[closing] = -1
    steps = []
    width, history = config.recovery.step_window, config.recovery.step_history
    step_valid = valid & (labels == STABLE) & ~blocked[:, None, None]
    step_values = np.where(step_valid, values, np.nan)
    for k in np.flatnonzero(edges.any(axis=(1, 2))):
        if k < history + width or k + history > n or blocked[k - history - width:k + history].any():
            continue
        candidate = edges[k] & ~taken[k] & allowed
        pre = step_values[k - width:k]
        post = step_values[k:k + width]
        pre_slope, pre_level = _fit(pre, time[k - width:k], time[k])
        _, post_level = _fit(post, time[k:k + width], time[k])
        offset = (post_level - pre_level) * 1000
        minimum = config.recovery.minimum_observations
        support = step_valid[k - width:k].sum(axis=0) >= minimum
        support &= step_valid[k:k + width].sum(axis=0) >= minimum
        support &= step_valid[k - history - width:k - history].sum(axis=0) >= minimum
        support &= step_valid[k + width:k + history].sum(axis=0) >= minimum
        earlier_slope, earlier_level = _fit(step_values[k - history - width:k - history],
                                          time[k - history - width:k - history], time[k])
        later = step_values[k + width:k + history]
        expected = earlier_level + (time[k + width:k + history] - time[k])[:, None, None] * earlier_slope
        persistent = np.abs(bn.nanmedian(later - expected, axis=0)) * 1000 > limit
        displacement = (post - pre_level - (time[k:k + width] - time[k])[:, None, None] * pre_slope) * 1000
        sign_count = ((np.sign(displacement) == np.sign(offset)) & step_valid[k:k + width]).sum(axis=0)
        persistent &= sign_count >= config.recovery.sign_agreement * step_valid[k:k + width].sum(axis=0)
        minimum_displacement = np.min(np.where(step_valid[k:k + width], np.abs(displacement), np.inf), axis=0)
        persistent &= minimum_displacement >= config.recovery.minimum_displacement * np.abs(offset)
        candidate &= support & persistent & (np.abs(offset) > limit)
        neighbors = np.zeros(allowed.shape, int)
        radius = config.recovery.neighbor_radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == dy == 0:
                    continue
                neighbor = ndimage.shift(difference[k - 1], (dy, dx), order=0, mode='constant', cval=np.nan)
                known = ndimage.shift(support & allowed, (dy, dx), order=0, mode='constant', cval=0)
                tolerance = np.maximum(config.recovery.neighbor_tolerance_pm,
                                       np.abs(offset) * config.recovery.neighbor_tolerance_fraction)
                neighbors += (known & (np.sign(neighbor) == np.sign(offset))
                              & (np.abs(neighbor - offset) < tolerance))
        # Broad regional responses are protected; consensus is only a sensor-step
        # assumption for small clusters, never evidence to erase a biological region.
        regional = candidate.sum() / max(allowed.sum(), 1) >= config.v2.regional_step_fraction
        confirmed = candidate & ((neighbors >= config.recovery.minimum_step_neighbors) | regional)
        approved = confirmed & (config.recovery.stitch and not regional)
        labels[k, candidate] = np.where(confirmed[candidate], STEP, UNRESOLVED)
        taken[k, confirmed] = True
        for y, x in zip(*np.nonzero(candidate)):
            steps.append(dict(frame=int(k), x=int(x), y=int(y), offset_pm=float(offset[y, x]),
                              neighbors=int(neighbors[y, x]), applied=bool(approved[y, x]),
                              evidence=('regional response preserved' if regional
                                        else 'neighbor-consensus assumption')))
    # Unpaired changes without history/consensus are a single unresolved segment.
    # A small terminal pulse must not silently become a new observed plateau.
    for k in np.flatnonzero(edges.any(axis=(1, 2))):
        pending = edges[k] & ~taken[k] & allowed
        regional = pending.sum() / max(allowed.sum(), 1) >= config.v2.regional_step_fraction
        if regional:
            end = min(n, k + max(config.v2.max_pulse_frames, config.recovery.median_radius) + 1)
            taken[k:end, pending] = True
            continue
        if k <= 2 * config.v2.max_pulse_frames and pending.any():
            prefix = labels[:k, pending]
            prefix[prefix == STABLE] = UNRESOLVED
            labels[:k, pending] = prefix
        for b in range(k + 1, n + 1):
            closing = pending & (edges[b] if b < n and not blocked[b] else True)
            if blocked[b - 1]:
                closing = pending
            if closing.any():
                segment = labels[k:b, closing]
                segment[segment == STABLE] = UNRESOLVED
                labels[k:b, closing] = segment
                pending &= ~closing
            if not pending.any():
                break
    # Valid disturbed segments without a qualified label are unavailable.
    unresolved = valid & (np.abs(values - baseline) * 1000 > limit) & (labels == STABLE) & ~taken
    labels[unresolved] = UNRESOLVED
    labels[blocked] = STABLE
    labels[:, ~allowed] = STABLE
    return labels, steps


def recover(state, config):
    result = state.copy()
    labels = np.zeros(state.values.shape, np.uint8)
    events = state.receipts['well_events']
    blocked = events['intervals'] | events['excluded_intervals']
    steps = []
    bounds = _bounds(state)
    if config.recovery.enabled:
        for left, right in zip(bounds[:-1], bounds[1:]):
            phase_labels, detected = label_phase(state.acquired_values[left:right],
                                                state.acquired_valid[left:right],
                                                state.raw.time_s[left:right],
                                                blocked[left:right], state.allowed, config,
                                                state.recovery_baselines.get((left, right)))
            labels[left:right] = phase_labels
            for step in detected:
                steps.append(dict(step, frame=step['frame'] + left, phase_start=left, phase_end=right))
        unresolved = (labels == UNRESOLVED) & ~blocked[:, None, None] & state.allowed[None]
        result.values[unresolved] = np.nan
        result.method[unresolved] = MISSING
        # Merge adjacent gap/pulse labels; interpolation uses the immutable evidence.
        temporal = (labels == GAP) | (labels == PULSE)
        for left, right in zip(bounds[:-1], bounds[1:]):
            starts = np.full(state.allowed.shape, -1, int)
            for b in range(left, right + 1):
                active = temporal[b] if b < right else np.zeros(state.allowed.shape, bool)
                starts[active & (starts < 0)] = b
                closing = ~active & (starts >= 0)
                for a in np.unique(starts[closing]):
                    pixels = closing & (starts == a)
                    fraction = ((state.raw.time_s[a:b] - state.raw.time_s[a - 1])
                                / (state.raw.time_s[b] - state.raw.time_s[a - 1]))
                    anchors = state.acquired_values[[a - 1, b]][:, pixels]
                    result.values[a:b, pixels] = anchors[0] + fraction[:, None] * (anchors[1] - anchors[0])
                    result.method[a:b, pixels] = TEMPORAL
                starts[closing] = -1
        for step in steps:
            if step['applied']:
                k, y, x, right = step['frame'], step['y'], step['x'], step['phase_end']
                # Cumulative offsets also translate earlier repairs. STITCHED
                # dominates their donor provenance once any stitch touches them.
                eligible = np.isin(result.method[k:right, y, x], [OBSERVED, TEMPORAL, CURATED, STITCHED])
                result.values[k:right, y, x][eligible] -= step['offset_pm'] / 1000
                result.method[k:right, y, x][eligible] = STITCHED
        cache = {}
        for frame in np.flatnonzero((labels == DROPOUT).any(axis=(1, 2))):
            donors = np.isin(result.method[frame], [OBSERVED, TEMPORAL])
            filled, radii, _ = fill_spatial_frame(result.values[frame], donors, labels[frame] == DROPOUT,
                                                 state.allowed, config.recovery, cache)
            repaired = radii > 0
            result.values[frame, repaired] = filled[repaired]
            result.method[frame, repaired] = SPATIAL
    result.receipts['pixel_events'] = dict(labels=labels, label_names=LABEL_NAMES, steps=steps,
                                         counts={name: int((labels[:, state.allowed] == i).sum())
                                                 for i, name in enumerate(LABEL_NAMES)})
    return result


def phase_offset(state, config):
    if config.phase_offset.mode != 'level':
        return stages.phase_offset(state, config)
    result = state.copy()
    maps = []
    bounds = _bounds(state)
    k = config.phase_offset.anchor_frames
    for left, start, end in zip(bounds, bounds[1:], bounds[2:]):
        supported = np.zeros(state.allowed.shape, bool)
        offset = np.full(state.allowed.shape, np.nan)
        if start - left >= k and end - start >= k:
            anchors = np.r_[np.arange(start - k, start), np.arange(start, start + k)]
            supported = np.isin(result.method[anchors], [OBSERVED, TEMPORAL, CURATED]).all(axis=0)
            supported &= np.isfinite(result.values[anchors]).all(axis=0) & state.allowed
            delta = (np.median(result.values[start - k:start], axis=0)
                     - np.median(result.values[start:start + k], axis=0))
            offset[supported] = delta[supported]
            result.values[start:end, supported] += delta[supported]
        unanchored = state.allowed & ~supported
        maps.append(dict(frame=start, offset_pm=offset, offset_unanchored=unanchored,
                         anchored_pixels=int(supported.sum()), unanchored_pixels=int(unanchored.sum())))
    result.receipts['phase_offset'] = dict(parameters=asdict(config.phase_offset), maps=maps,
                                         background_checks=stages._background_continuity(
                                             result, config, bounds))
    return result


def preprocess_v2(raw, config, curation, provenance, event_plan=None):
    state = sample_mask(raw, config)
    state = register(state, config)
    state = well_events(state, config, curation, event_plan)
    state = recover(state, config)
    state = stages.reference(state, config)
    state = stages.background(state, config)
    state = phase_offset(state, config)
    return stages.finalize(state, config, provenance)
