import bottleneck as bn
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib
import matplotlib.pyplot as plt
from .math_ops import find_local_maxima

def select_indices(array, threshold, num_indices, spacing = 2, random_distance=5, layer=0):
    # Get the indices of the true elements in the boolean array
    
    bool_array = array < (threshold + layer * 25)
    
    true_indices = np.argwhere(bool_array)
    dst = distance_transform_edt(bool_array)

    rows, cols = bool_array.shape
    center_row = rows // 2
    center_col = cols // 2
    
    distances_to_center = np.array([np.sqrt((y - center_row)**2 + (x - center_col)**2) for y, x in true_indices])
    preference_weights = 1 / (distances_to_center + 1)  # Adding 1 to avoid division by zero
    
    chosen_indices = []
    
    n_attempts = 0
    
    while len(chosen_indices) < num_indices and n_attempts < 1000:
        # Choose a random true index
        idx = np.random.choice(len(true_indices), p=preference_weights.ravel() / np.sum(preference_weights))
        random_true_index = np.unravel_index(idx, (rows, cols))
        
        # Check if the chosen index is at least 'random_distance' units away from other chosen indices
        if all(np.linalg.norm(np.array(chosen_index) - random_true_index) >= random_distance for chosen_index in chosen_indices) and dst[random_true_index[0],random_true_index[1]] >= spacing:
            chosen_indices.append(random_true_index)
            
        n_attempts += 1
    if len(chosen_indices) < num_indices:
        if layer < 10:
            return select_indices(array, threshold, num_indices, spacing, random_distance, layer + 1)
        else:
            if len(chosen_indices) > 0:
                x_indices, y_indices = zip(*chosen_indices)
                return x_indices, y_indices
            return []
        
    # Split the chosen indices into x and y groups
    x_indices, y_indices = zip(*chosen_indices)
    
    return x_indices, y_indices

def dilate_mask(mask, times=1):
    for i in range(times):
        mask = ndimage.binary_dilation(mask, [
        [False, True, False],
        [ True, True,  True],
        [False, True, False],
    ])
    return mask


def _centered_moving_median(values, radius):
    window_size = 2 * radius + 1
    padded = np.pad(values, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    trailing = bn.move_median(
        padded, window=window_size, min_count=window_size, axis=0,
    )
    return trailing[window_size - 1:window_size - 1 + len(values)]


def _interpolate_phase_jumps(segment, threshold, max_gap):
    corrected = segment.copy()
    corrected_mask = np.zeros_like(corrected, dtype=bool)
    if corrected.shape[0] < 3 or max_gap < 1:
        return corrected, corrected_mask

    baseline = _centered_moving_median(corrected, max_gap)
    candidates = np.isfinite(corrected) & (
        np.abs(corrected - baseline) > threshold
    )

    temporal_structure = np.zeros((3, 3, 3), dtype=bool)
    temporal_structure[:, 1, 1] = True
    labels, _ = ndimage.label(candidates, structure=temporal_structure)
    run_lengths = np.bincount(labels.ravel())
    corrected_mask = (labels > 0) & (run_lengths[labels] <= max_gap)

    boundary_labels = np.unique(
        np.concatenate((labels[0].ravel(), labels[-1].ravel()))
    )
    boundary_labels = boundary_labels[boundary_labels > 0]
    if boundary_labels.size:
        corrected_mask &= ~np.isin(labels, boundary_labels)

    samples = np.arange(corrected.shape[0])
    flat = corrected.reshape(corrected.shape[0], -1)
    mask_flat = corrected_mask.reshape(corrected.shape[0], -1)
    for pixel in np.flatnonzero(np.any(mask_flat, axis=0)):
        missing = mask_flat[:, pixel]
        known = ~missing & np.isfinite(flat[:, pixel])
        if np.count_nonzero(known) < 2:
            missing[:] = False
            continue
        flat[missing, pixel] = np.interp(
            samples[missing], samples[known], flat[known, pixel],
        )

    return corrected, corrected_mask


def interpolate_frame_jumps(well, threshold=75.0, phases=(), max_gap=5):
    """Interpolate trace-local artifacts without crossing phase boundaries.

    Each acquisition phase is corrected independently. Samples above the
    absolute ``threshold`` from their local median are grouped into connected
    runs. Runs no longer than ``max_gap`` are interpolated from valid neighbors.
    """
    corrected = np.asarray(well).copy()
    if corrected.ndim != 3 or corrected.shape[0] < 3:
        return corrected, {"samples": 0, "frames": 0}

    threshold = float(threshold)
    if threshold <= 0:
        return corrected, {"samples": 0, "frames": 0}

    corrected_mask = np.zeros_like(corrected, dtype=bool)
    max_gap = max(0, int(max_gap))
    boundaries = sorted({
        int(phase) for phase in phases or ()
        if 0 < int(phase) < corrected.shape[0]
    })
    starts = [0, *boundaries]
    ends = [*boundaries, corrected.shape[0]]
    for start, end in zip(starts, ends):
        phase_corrected, phase_mask = _interpolate_phase_jumps(
            corrected[start:end], threshold, max_gap,
        )
        corrected[start:end] = phase_corrected
        corrected_mask[start:end] = phase_mask

    affected_frames = np.any(corrected_mask, axis=(1, 2))
    return corrected, {
        "samples": int(np.count_nonzero(corrected_mask)),
        "frames": int(np.count_nonzero(affected_frames)),
    }

def correct_well(well, threshold=75, coords=[], mode='mean', pixel_filtering=True):
    corr_data = well.copy()
    
    df = np.abs(np.diff(corr_data, axis=0))
    df = np.sum(df, axis=0)
    mask = np.logical_or(df == 0, df > 1000)
    
    corr_data -= corr_data[0, :, :]
    corr_data *= 1000
    mask = dilate_mask(mask, 1) if pixel_filtering else np.zeros_like(mask, dtype=bool)

    if len(coords) > 0:
        valid_coords = []
        for point in coords:
            x, y = int(point[0]), int(point[1])
            if not (0 <= y < corr_data.shape[1] and 0 <= x < corr_data.shape[2]):
                continue
            signal = corr_data[:, y, x]
            if mask[y, x] or not np.all(np.isfinite(signal)) or np.max(np.abs(signal)) > 5000:
                continue
            valid_coords.append((x, y))
        coords = (
            [[point[0] for point in valid_coords], [point[1] for point in valid_coords]]
            if valid_coords else []
        )
    else:
        coords = select_indices(corr_data[-1], threshold, 7, 2, 2)

    if len(coords) > 0:
        filter_method = np.median if mode == 'median' else np.mean
        fltr = filter_method(corr_data[:, coords[1], coords[0]], axis=1)
        corr_data -= fltr[:, None, None]
        # corr_data[:, mask] = 0
    else:
        print('Could not perform random background correction!')
    corr_data[np.abs(corr_data) > 5000] = 0
    corr_data[:, mask] = 0
    corr_data = np.clip(corr_data, 0, np.max(corr_data))
    
    return corr_data, {} if len(coords) == 0 else list(zip(coords[0], coords[1])), mask

def _phase_translation(source_points, target_points, radius=3):
    """Return a confidence-gated integer shift from source to target points."""
    source = np.asarray(source_points, dtype=np.int32).reshape(-1, 2)
    target = np.asarray(target_points, dtype=np.int32).reshape(-1, 2)
    if len(source) < 5 or len(target) < 5:
        return np.zeros(2, dtype=np.int32)

    target_set = {tuple(point) for point in target.tolist()}
    tree = cKDTree(target)
    candidates = []
    for dy in range(-int(radius), int(radius) + 1):
        for dx in range(-int(radius), int(radius) + 1):
            shift = np.asarray([dx, dy], dtype=np.int32)
            shifted = source + shift
            exact = sum(tuple(point) in target_set for point in shifted.tolist())
            distances = tree.query(shifted, k=1)[0]
            nearby = int(np.count_nonzero(distances <= 1.1))
            mean_distance = float(np.mean(np.minimum(distances, 4.0)))
            score = (exact, nearby, -mean_distance, -abs(dx) - abs(dy))
            candidates.append((score, shift, exact, nearby))

    best = max(candidates, key=lambda item: item[0])
    zero = next(item for item in candidates if np.all(item[1] == 0))
    required_gain = max(3, int(np.ceil(zero[2] * 0.25)))
    if best[2] - zero[2] < required_gain or best[3] < zero[3]:
        return np.zeros(2, dtype=np.int32)
    return best[1]


def _shift_phase_frames(frames, translation):
    """Apply an integer translation with nearest-edge fill and no wraparound."""
    dx, dy = (int(value) for value in translation)
    if dx == 0 and dy == 0:
        return frames
    height, width = frames.shape[1:]
    source_x = np.clip(np.arange(width) - dx, 0, width - 1)
    source_y = np.clip(np.arange(height) - dy, 0, height - 1)
    return frames[:, source_y[:, None], source_x[None, :]]


def correct_interphase_well_shifts(raw_well, phases, threshold, coords, mode,
                                   pixel_filtering=True):
    """Register each phase independently to the strongest phase projection."""
    shifted_well = raw_well.copy()
    # The activity mask is used only to keep broken pixels out of registration;
    # it does not mask the returned raw cube and is independent of the user's
    # preprocessing pixel-filter toggle.
    corrected, _filter_points, mask = correct_well(
        raw_well, coords=coords, threshold=threshold, mode=mode,
    )
    boundaries = sorted({
        int(phase) for phase in phases
        if 0 < int(phase) < corrected.shape[0]
    })
    starts = [0, *boundaries]
    ends = [*boundaries, corrected.shape[0]]
    phase_points = []
    for start, end in zip(starts, ends):
        projection = np.max(corrected[start:end], axis=0)
        if np.max(projection) < threshold:
            phase_points.append(np.empty((0, 2), dtype=np.int32))
            continue
        xs, ys = find_local_maxima(
            projection, threshold, np.max(projection), 3, mask,
        )
        phase_points.append(np.column_stack([xs, ys]).astype(np.int32))

    if not phase_points:
        return shifted_well
    reference_index = int(np.argmax([len(points) for points in phase_points]))
    reference_points = phase_points[reference_index]
    for phase_index, (start, end) in enumerate(zip(starts, ends)):
        if phase_index == reference_index:
            continue
        translation = _phase_translation(phase_points[phase_index], reference_points)
        shifted_well[start:end] = _shift_phase_frames(
            shifted_well[start:end], translation,
        )
    return shifted_well
