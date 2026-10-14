"""Phase-local pixel recovery with explicit validity and reconstruction provenance."""

from dataclasses import dataclass

import numpy as np
import bottleneck as bn
from scipy import ndimage
from scipy.spatial import Delaunay

OBSERVED, TEMPORAL, SPATIAL, STITCHED, MISSING, EXCLUDED = range(6)
METHOD_NAMES = ['observed', 'temporal', 'spatial', 'stitched', 'missing', 'excluded']
FOUR = ndimage.generate_binary_structure(2, 1)


@dataclass(frozen=True)
class RecoveryConfig:
    temporal_gap: int = 5
    threshold_pm: float = 75.0
    noise_multiplier: float = 6.0
    tail_frames: int = 10
    radii: tuple = (1, 2, 4, 8)
    max_cluster_area: int = 81
    stitch: bool = True
    minimum_step_neighbors: int = 2


@dataclass
class RecoveryResult:
    values: np.ndarray
    methods: np.ndarray
    spatial_radius: np.ndarray
    donor_count: np.ndarray
    steps: list
    summary: dict


def _runs(mask):
    delta = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(delta == 1), np.flatnonzero(delta == -1))


def _valid_centered_median(values, radius=5):
    """Ignore unavailable samples; insufficient support gives no outlier verdict."""
    padded = np.pad(values, ((radius, radius), (0, 0), (0, 0)), mode='edge')
    width = 2 * radius + 1
    median = bn.move_median(padded, window=width, min_count=3, axis=0)
    return median[width - 1:width - 1 + len(values)]


def _spatial_operator(hole, observed, allowed, config):
    """Compile a linear map from ORIGINAL donors, including a joint harmonic solve."""
    yy, xx = np.nonzero(hole)
    if len(xx) > config.max_cluster_area:
        return None
    points = np.column_stack([xx, yy])
    search = hole.copy()
    distance = ndimage.distance_transform_edt(~observed)
    anchors = None
    hull = None
    radius = None
    for r in range(1, max(config.radii) + 1):
        search = ndimage.binary_dilation(search, FOUR, mask=allowed)
        if r not in config.radii or np.max(distance[hole]) > r:
            continue
        anchors = np.column_stack(np.nonzero(search & observed)[::-1])
        if len(anchors) < 3:
            continue
        try:
            hull = Delaunay(anchors)
            if np.all(hull.find_simplex(points) >= 0):
                radius = r
                break
        except Exception:
            continue
    if radius is None:
        return None
    ring = ndimage.binary_dilation(hole, FOUR) & observed
    direct = np.column_stack(np.nonzero(ring)[::-1])
    supported = False
    if len(direct) >= 3:
        try:
            supported = np.all(Delaunay(direct).find_simplex(points) >= 0)
        except Exception:
            pass
    if supported:
        # Solve all mutually dependent missing pixels together. This is the
        # converged linear system for iterative four-neighbor harmonic filling.
        unknown = {tuple(p): i for i, p in enumerate(points)}
        donors = {tuple(p): i for i, p in enumerate(direct)}
        matrix = np.zeros((len(points), len(points)))
        rhs = np.zeros((len(points), len(direct)))
        for i, (x, y) in enumerate(points):
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (x + dx, y + dy)
                if neighbor in unknown:
                    matrix[i, i] += 1
                    matrix[i, unknown[neighbor]] -= 1
                elif neighbor in donors:
                    matrix[i, i] += 1
                    rhs[i, donors[neighbor]] += 1
        try:
            weights = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            return None
        return points, direct, weights, radius
    simplex = hull.find_simplex(points)
    transform = hull.transform[simplex]
    barycentric = np.einsum('nij,nj->ni', transform[:, :2], points - transform[:, 2])
    barycentric = np.column_stack([barycentric, 1 - barycentric.sum(axis=1)])
    weights = np.zeros((len(points), len(anchors)))
    for i, vertices in enumerate(hull.simplices[simplex]):
        weights[i, vertices] = barycentric[i]
    return points, anchors, weights, radius


def fill_spatial_frame(values, observed, targets, allowed, config=None, cache=None):
    """Fill supported components; excluded/unsupported samples remain unchanged."""
    config = config or RecoveryConfig()
    cache = {} if cache is None else cache
    result = np.asarray(values, dtype=float).copy()
    observed = observed & allowed & np.isfinite(values)
    targets = targets & allowed & ~observed
    radius_map = np.zeros(allowed.shape, dtype=np.uint8)
    donor_map = np.zeros(allowed.shape, dtype=np.uint16)
    labels, count = ndimage.label(targets, FOUR)
    for label, region in enumerate(ndimage.find_objects(labels, count), start=1):
        if region is None:
            continue
        hole = labels == label
        if hole.sum() > config.max_cluster_area:
            continue
        # The full donor validity pattern is part of the cache key: no weights
        # from a different frame's missing boundary may silently be reused.
        reach = hole.copy()
        for _ in range(max(config.radii)):
            reach = ndimage.binary_dilation(reach, FOUR, mask=allowed)
        key = (np.flatnonzero(hole).tobytes(), np.packbits(observed & reach).tobytes(),
               config.radii, config.max_cluster_area, allowed.tobytes())
        if key not in cache:
            if len(cache) >= 8192:
                cache.clear()
            cache[key] = _spatial_operator(hole, observed, allowed, config)
        operator = cache[key]
        if operator is None:
            continue
        points, donors, weights, radius = operator
        estimate = weights @ values[donors[:, 1], donors[:, 0]]
        result[points[:, 1], points[:, 0]] = estimate
        radius_map[points[:, 1], points[:, 0]] = radius
        donor_map[points[:, 1], points[:, 0]] = np.count_nonzero(weights > 1e-12, axis=1)
    return result, radius_map, donor_map


def recover_pixels(values, original_valid, time, phases, allowed, config=None):
    """Recover wavelength cubes in nm; thresholds/offset receipts are in pm.

    Never interpolate across acquisition phases. Spatial values are estimates
    and never become independent donors. Unresolved allowed samples are NaN.
    """
    config = config or RecoveryConfig()
    raw = np.asarray(values, dtype=float)
    time = np.asarray(time, dtype=float)
    allowed = np.asarray(allowed, dtype=bool)
    valid = np.asarray(original_valid, dtype=bool) & np.isfinite(raw)
    if raw.ndim != 3 or valid.shape != raw.shape or allowed.shape != raw.shape[1:]:
        raise ValueError('Expected matching time × height × width values/validity and spatial allowed mask')
    if len(time) != len(raw) or np.any(~np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError('Time must match the cube and increase strictly')
    if config.temporal_gap < 1 or not config.radii or config.max_cluster_area < 1:
        raise ValueError('Recovery limits must be positive')
    result = raw.copy()
    methods = np.full(raw.shape, OBSERVED, dtype=np.uint8)
    methods[:, ~allowed] = EXCLUDED
    methods[~valid & allowed[None]] = MISSING
    result[methods == MISSING] = np.nan
    targets = np.zeros(raw.shape, dtype=bool)
    radii = np.zeros(raw.shape, dtype=np.uint8)
    donors = np.zeros(raw.shape, dtype=np.uint16)
    steps = []
    bounds = [0, *sorted({int(p) for p in phases if 0 < p < len(raw)}), len(raw)]
    responsive = allowed & (valid.sum(axis=0) >= 3)
    for left, right in zip(bounds[:-1], bounds[1:]):
        if right - left < 3:
            continue
        segment = raw[left:right]
        measured = np.where(valid[left:right], segment, np.nan)
        baseline = _valid_centered_median(measured)
        residual = (segment - baseline) * 1000
        measured_residual = np.where(valid[left:right], np.abs(residual), np.nan)
        noise = 1.4826 * _valid_centered_median(measured_residual)
        suspect = np.abs(residual) > np.maximum(config.threshold_pm, config.noise_multiplier * noise)
        changes = np.diff(segment, axis=0) * 1000
        typical = ndimage.median_filter(changes, size=(11, 1, 1), mode='nearest')
        deviations = changes - typical
        step_noise = 1.4826 * ndimage.median_filter(np.abs(deviations), size=(11, 1, 1), mode='nearest')
        candidate_map = np.abs(deviations) > np.maximum(config.threshold_pm,
                                                       config.noise_multiplier * step_noise)
        for y, x in zip(*np.nonzero(responsive)):
            known = valid[left:right, y, x]
            bad = ~known | suspect[:, y, x]
            # Median noise can mask part of a pulse and leave its falling edge
            # incorrectly classified as a permanent offset. Resolve the whole
            # short returning pulse first, using only same-phase observations.
            trace = segment[:, y, x]
            for a in np.flatnonzero(np.abs(changes[:, y, x]) > config.threshold_pm) + 1:
                if a < 3 or methods[left + a, y, x] == TEMPORAL:
                    continue
                for b in range(a + 1, min(a + config.temporal_gap + 1, len(known) - 2)):
                    if not known[a - 3:b + 3].all():
                        continue
                    if abs(trace[b] - np.median(trace[b:b + 3])) * 1000 > config.threshold_pm:
                        continue
                    anchors = np.r_[np.arange(a - 3, a), np.arange(b, b + 3)]
                    offsets = time[left + anchors] - time[left + a]
                    fit = np.polyfit(offsets, trace[anchors], 1)
                    residuals = (trace[anchors] - np.polyval(fit, offsets)) * 1000
                    outer_noise = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
                    tolerance = max(config.threshold_pm, config.noise_multiplier * outer_noise)
                    if abs(trace[b] - trace[a - 1]) * 1000 > tolerance:
                        continue
                    samples = np.arange(left + a, left + b)
                    interpolated = np.interp(time[samples], time[[left + a - 1, left + b]],
                                             trace[[a - 1, b]])
                    amplitude = np.max(np.abs(trace[a:b] - interpolated)) * 1000
                    if amplitude <= tolerance:
                        continue
                    result[samples, y, x] = interpolated
                    methods[samples, y, x] = TEMPORAL
                    break
            for a, b in _runs(bad):
                if (b - a > config.temporal_gap or a == 0 or b == len(known)
                        or not known[a - 1] or not known[b]):
                    continue
                if abs(segment[b, y, x] - segment[a - 1, y, x]) * 1000 > config.threshold_pm:
                    continue
                samples = np.arange(left + a, left + b)
                result[samples, y, x] = np.interp(time[samples], time[[left + a - 1, left + b]],
                                                raw[[left + a - 1, left + b], y, x])
                methods[samples, y, x] = TEMPORAL
            observed_ids = np.flatnonzero(known)
            if len(observed_ids) >= 3 and len(known) - observed_ids[-1] - 1 >= config.tail_frames:
                targets[left + observed_ids[-1] + 1:right, y, x] = True
            elif len(observed_ids) == 0 and left > 0 and valid[:left, y, x].sum() >= 3:
                targets[left:right, y, x] = True
            # Other unbracketed/long returning gaps remain unresolved.
            rejected = bad & (methods[left:right, y, x] == OBSERVED)
            result[left:right, y, x][rejected] = np.nan
            methods[left:right, y, x][rejected] = MISSING
            last = -100
            for k in np.flatnonzero(candidate_map[:, y, x]) + 1:
                if k < 6 or k + 6 >= len(known) or k - last < 6 or not known[k - 6:k + 6].all():
                    continue
                if np.any(methods[left + k - 6:left + k + 6, y, x] == TEMPORAL):
                    continue
                offsets = time[left + k - 6:left + k + 6] - time[left + k]
                before = np.polyfit(offsets[:6], segment[k - 6:k, y, x], 1)
                after = np.polyfit(offsets[6:], segment[k:k + 6, y, x], 1)
                offset = float((after[1] - before[1]) * 1000)
                displacement = (segment[k:k + 6, y, x] - np.polyval(before, offsets[6:])) * 1000
                if abs(offset) < config.threshold_pm or np.mean(np.sign(displacement) == np.sign(offset)) < 0.8:
                    continue
                if np.min(np.abs(displacement)) < 0.3 * abs(offset):
                    continue
                neighbor_count = 0
                for py in range(max(0, y - 1), min(raw.shape[1], y + 2)):
                    for px in range(max(0, x - 1), min(raw.shape[2], x + 2)):
                        if (py, px) == (y, x) or not allowed[py, px]:
                            continue
                        if not valid[left + k - 6:left + k + 6, py, px].all():
                            continue
                        jump = deviations[k - 1, py, px]
                        if np.sign(jump) == np.sign(offset) and abs(jump - offset) < max(25, abs(offset) * 0.25):
                            neighbor_count += 1
                returns_to_earlier_level = False
                history_supported = (k >= 18 and k + 12 <= len(known)
                                     and known[k - 18:k - 12].all() and known[k + 6:k + 12].all())
                if history_supported:
                    earlier_offsets = time[left + k - 18:left + k - 12] - time[left + k]
                    earlier = np.polyfit(earlier_offsets, trace[k - 18:k - 12], 1)
                    later_offsets = time[left + k + 6:left + k + 12] - time[left + k]
                    return_error = (trace[k + 6:k + 12] - np.polyval(earlier, later_offsets)) * 1000
                    earlier_residual = (trace[k - 18:k - 12] - np.polyval(earlier, earlier_offsets)) * 1000
                    earlier_noise = 1.4826 * np.median(np.abs(earlier_residual - np.median(earlier_residual)))
                    leverage = np.sqrt(1 / 6 + (np.mean(later_offsets) - np.mean(earlier_offsets)) ** 2
                                       / np.sum((earlier_offsets - np.mean(earlier_offsets)) ** 2))
                    tolerance = max(config.threshold_pm, config.noise_multiplier * earlier_noise * leverage)
                    returns_to_earlier_level = abs(np.median(return_error)) <= tolerance
                approved = bool(config.stitch and neighbor_count >= config.minimum_step_neighbors
                                and history_supported and not returns_to_earlier_level)
                evidence = 'neighbor-consensus assumption' if approved else 'unconfirmed step'
                if returns_to_earlier_level:
                    evidence = 'return to earlier baseline; not a persistent offset'
                steps.append(dict(frame=int(left + k), phase_start=int(left), x=int(x), y=int(y),
                                  offset_pm=offset, neighbors=neighbor_count, applied=approved,
                                  evidence=evidence))
                if approved:
                    result[left + k:right, y, x] -= offset / 1000
                    original = methods[left + k:right, y, x] == OBSERVED
                    methods[left + k:right, y, x][original] = STITCHED
                last = k
    cache = {}
    for frame in np.flatnonzero(targets.any(axis=(1, 2))):
        observed = (methods[frame] == OBSERVED) | (methods[frame] == STITCHED)
        filled, radius, count = fill_spatial_frame(result[frame], observed, targets[frame],
                                                  allowed, config, cache)
        repaired = radius > 0
        result[frame][repaired] = filled[repaired]
        methods[frame][repaired] = SPATIAL
        radii[frame] = radius
        donors[frame] = count
    assert np.array_equal(result[:, ~allowed], raw[:, ~allowed], equal_nan=True)
    summary = {name: int(np.sum(methods == i)) for i, name in enumerate(METHOD_NAMES)}
    summary.update(step_candidates=len(steps), applied_steps=sum(e['applied'] for e in steps))
    return RecoveryResult(result, methods, radii, donors, steps, summary)
