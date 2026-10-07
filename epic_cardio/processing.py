from operator import itemgetter
from dataclasses import dataclass
import numpy as np
import os
from typing import Any, Callable, Dict, Optional
from .data_correction import correct_well, correct_interphase_well_shifts, interpolate_frame_jumps
from .filter import border_filter_for_well
from .math_ops import calculate_cell_maximas
from .measurement_load import load_measurement, wl_map_to_wells, load_high_freq_measurement
from .defs import *
import json
from scipy.ndimage import distance_transform_edt, label as label_connected
from skimage.segmentation import watershed
from tqdm import tqdm

class RangeType():
    MEASUREMENT_PHASE=0
    INDIVIDUAL_POINT=1


@dataclass(frozen=True)
class WatershedResult:
    labels: np.ndarray
    foreground_mask: np.ndarray
    active_labels: np.ndarray
    unresolved_labels: np.ndarray
    pixel_counts: Dict[int, int]

def load_data(path, measurement_type=MeasurementType.TYPE_NORMAL, flip=[False, False]):
    try:
        # Betölti a 3x4-es well képet a projekt mappából.
        wl_map, time = load_measurement(path)
            
        if measurement_type == MeasurementType.TYPE_HIGH_FREQ:
            h_paths = [ (p, os.path.join(path, p), int(p.split('_')[0][6:])) for p in os.listdir(path) if "Cardio" in p]
            h_paths = sorted(h_paths, key=lambda x: x[2])
            
            print(h_paths)
            for name, path, idx in h_paths:
                wl_map_h, time_h = load_high_freq_measurement(path)
                
                wl_map_h -= wl_map_h[0]
                    
                if wl_map is None:
                    wl_map = wl_map_h
                    time = time_h
                else:
                    wl_map_h += np.mean(wl_map[-50:], axis=0)
                    wl_map = np.concatenate([wl_map, wl_map_h])
                    time = np.concatenate([time, time_h[:-1] + 100 + time[-1]])
                
        # Itt szétválasztásra kerülnek a wellek. Betöltéskor egy 240x320-as képen található a 3x4 elrendezésű 12 well.
        wells = wl_map_to_wells(wl_map, flip=flip)
        phases = list(np.where((np.diff(time)) > 60)[0] + 1)
        print([(n+1, p) for n, p in enumerate(phases)])
        return wells, time, phases
    except Exception as e:
        print(f'Error occured during data load: {e}')
        return None

def load_params(path):
    filter_params = {}
    preprocessing, localization = {}, {}
    if os.path.exists(os.path.join(path, '.metadata/parameters.json')):
        with open(os.path.join(path, '.metadata/parameters.json'), 'r') as f:
            obj = json.load(f)
            filter_params = obj['filter_ptss']
            preprocessing = obj['preprocessing']
            localization = obj['localization']
    return filter_params, preprocessing, localization

def save_params(path, well_data, preprocessing, localization):
    if not os.path.exists(os.path.join(path, '.metadata')):
        os.mkdir(os.path.join(path, '.metadata'))
    parameters = {
        # This code iterates through each key-value pair in well_data.
        # And because the json module does not support NumPy integers it converts them to Python integers.
        # For each value[-1], which is a list of tuples, it iterates through each tuple x in the list, 
        # converting each element y within the tuple from a NumPy integer to a Python integer, and reconstructs the tuple with these converted values. 
        'filter_ptss' : {key: [tuple(int(y) for y in x) for x in value[-1]] for key, value in well_data.items()},
        'preprocessing': preprocessing,
        'localization': localization
    }
    with open(os.path.join(path, '.metadata/parameters.json'), 'w+') as f:
        json.dump(parameters, f)


def preprocessing(
    preprocessing_params, wells, time, phases, background_coords={},
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
):
    well_data = {}
    filter_ptss = {}
    
    if len(preprocessing_params) == 0:
        preprocessing_params = {
            'signal_range' : {
                'range_type': RangeType.MEASUREMENT_PHASE,
                'ranges': [0, None],
            },
            'drift_correction': {
                'threshold': 75,
                'filter_method': 'mean',
                'background_selector': True,
                'frame_jump_correction': True,
                'frame_jump_threshold': 75.0,
            }
        }
    
    rngs = preprocessing_params['signal_range']['ranges']
    if rngs[1] != None:
        if rngs[0] >= rngs[1]:
            raise ValueError('End point of the range has to be greater than starting point!')

    if preprocessing_params['signal_range']['range_type'] == RangeType.MEASUREMENT_PHASE:
        selected_range = [0 if rngs[0] == 0 or rngs[0] > len(phases) else phases[rngs[0] - 1],
                        len(time) + 1 if rngs[1] == None else phases[rngs[1] - 1]]
    else:   
        selected_range = rngs
        
    slicer = slice(selected_range[0], selected_range[1])
    time = time[slicer]
    phases = [p for p in phases if p >= selected_range[0] and p < selected_range[1]]

    if selected_range[0] > 0:
        tmp = []
        for p in phases:
            if p - selected_range[0] > 0:
                tmp.append(p - selected_range[0])
        phases = tmp

    for done, name in enumerate(WELL_NAMES, start=1):
        if progress_callback is None:
            print("Parsing", name, end='\r')
        well_tmp = wells[name]
        
        # if export_params['breakdown_signal']:
        #     line = np.mean(wells[name], axis=(1,2))
        #     peak_until = phases[-1] + np.argmax(line[phases[-1]:])
        #     peak_until = peak_until if line[peak_until] > line[phases[-1] - 1] else phases[-1] - 1
        #     breakdowns[name] = peak_until
        
        drift_params = preprocessing_params['drift_correction']
        well_tmp = well_tmp[slicer]
        well_tmp = correct_interphase_well_shifts(
            well_tmp, phases,
            coords=(
                [] if not drift_params['background_selector'] or len(background_coords) == 0
                else background_coords[name]
            ),
            threshold=drift_params['threshold'], mode=drift_params['filter_method'],
        )
        well_corr, coords, _ = correct_well(well_tmp,
                                        coords=[] if len(background_coords) == 0 else background_coords[name],
                                        threshold=drift_params['threshold'],
                                        mode=drift_params['filter_method'])
        jump_summary = {"samples": 0, "frames": 0}
        if drift_params.get('frame_jump_correction', True):
            well_corr, jump_summary = interpolate_frame_jumps(
                well_corr, threshold=drift_params.get('frame_jump_threshold', 75.0), phases=phases,
            )
        well_data[name] = well_corr
        filter_ptss[name] = coords
        if progress_callback is not None:
            correction_text = ""
            if jump_summary["samples"]:
                correction_text = (
                    f"; corrected {jump_summary['samples']} jump samples "
                    f"in {jump_summary['frames']} frames"
                )
            progress_callback(done, len(WELL_NAMES), f"Preprocessed well {name}{correction_text}")
    if progress_callback is None:
        print("Parsing finished!")
    return well_data, time, phases, filter_ptss, selected_range

def localization(
    preprocessing_params, localization_params, wells, phases, selected_range, background_coords={},
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
):
    # Sejt szűrés a wellekből.
    well_data = {}
    slicer = slice(selected_range[0], selected_range[1])
    well_names = (
        WELL_NAMES
        if progress_callback is not None
        else tqdm(WELL_NAMES, desc="Parsing", unit="well")
    )
    for done, name in enumerate(well_names, start=1):
        border_filter = border_filter_for_well(localization_params, name)
        well_tmp = wells[name][slicer]
        
        drift_params = preprocessing_params['drift_correction']
        well_tmp = correct_interphase_well_shifts(well_tmp, phases,
                                        coords=[] if not drift_params['background_selector'] else background_coords[name],
                                        threshold=drift_params['threshold'],
                                        mode=drift_params['filter_method'])
        
        well_corr, filter_ptss, mask = correct_well(well_tmp, 
                                            coords=[] if not drift_params['background_selector'] else background_coords[name],
                                            threshold=drift_params['threshold'],
                                            mode=drift_params['filter_method'])
        if drift_params.get('frame_jump_correction', True):
            well_corr, _ = interpolate_frame_jumps(
                well_corr, threshold=drift_params.get('frame_jump_threshold', 75.0), phases=phases,
            )
        
        ptss = calculate_cell_maximas(well_corr, 
                    min_threshold=localization_params['threshold_range'][0], 
                    max_threshold=localization_params['threshold_range'][1], 
                    neighborhood_size=localization_params['neighbourhood_size'],
                    error_mask=None if not localization_params['error_mask_filtering'] else mask)

        if len(ptss) > 0 and any(v > 0 for v in border_filter.values()):
            ptss = np.asarray(ptss)
            y_max, x_max = well_corr.shape[1], well_corr.shape[2]
            top = min(border_filter['top'], y_max)
            bottom = min(border_filter['bottom'], y_max)
            left = min(border_filter['left'], x_max)
            right = min(border_filter['right'], x_max)
            is_inside = (
                (ptss[:, 0] >= left) &
                (ptss[:, 0] < x_max - right) &
                (ptss[:, 1] >= top) &
                (ptss[:, 1] < y_max - bottom)
            )
            ptss = ptss[is_inside]

        well_data[name] = (well_corr, ptss, filter_ptss)
        if progress_callback is not None:
            progress_callback(done, len(WELL_NAMES), f"Localized well {name}")
    return well_data

def parse_selection(well_data:dict, selector: Any, evaluation_params:dict) -> (dict, dict):
    selected_ptss = {}
    for name in WELL_NAMES:
        if not evaluation_params['cell_selector']:
            selected_ptss[name] = well_data[name][1]
        else:
            if len(selector.saved_ids[name]) > 0:
                ptss_selected = np.array(itemgetter(*selector.saved_ids[name])(well_data[name][1]), np.uint8)
                if(ptss_selected.ndim == 1):
                    ptss_selected = np.expand_dims(ptss_selected, axis=0)
                selected_ptss[name] = ptss_selected
    return selected_ptss

def threshold_bounded_watershed(signal, projected_labels, threshold=160,
                                max_growth_distance=None, seed_points=None, seed_regions=None):
    """Run marker-controlled watershed inside a strict threshold foreground.

    ``projected_labels`` supplies stable cell identities and their spatial
    reference footprints. Pixels at or below ``threshold`` are authoritative
    background and are never promoted to foreground by the marker image.
    Each overlapping label gets one seed near its optional ``seed_points``
    coordinate. Each connected foreground component is partitioned by distance
    to the full projected ``seed_regions`` footprint. Centroid distance resolves
    equal footprint distances. Disconnected foreground without an eligible seed
    stays unlabeled. Growth is unlimited by default.
    """
    signal_img = np.asarray(signal)
    if signal_img.ndim == 3:
        signal_img = np.max(signal_img, axis=0)
    elif signal_img.ndim != 2:
        raise ValueError("Input signal data must be a 2D or 3D numpy array.")

    reference_labels = np.asarray(projected_labels)
    if reference_labels.ndim != 2 or reference_labels.shape != signal_img.shape:
        raise ValueError("Projected labels must be a 2D array matching the signal shape.")
    if not np.issubdtype(reference_labels.dtype, np.integer):
        if not np.all(np.isfinite(reference_labels)) or not np.all(reference_labels == np.rint(reference_labels)):
            raise ValueError("Projected labels must contain finite integer values.")
    reference_labels = reference_labels.astype(np.int32, copy=False)
    if np.any(reference_labels < 0):
        raise ValueError("Projected labels must be non-negative.")

    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("Watershed threshold must be finite.")
    if max_growth_distance is not None:
        max_growth_distance = float(max_growth_distance)
        if not np.isfinite(max_growth_distance) or max_growth_distance < 0:
            raise ValueError("Maximum watershed growth distance must be finite and non-negative.")

    signal_img = signal_img.astype(np.float32, copy=False)
    foreground = np.isfinite(signal_img) & (signal_img > threshold)
    components, component_count = label_connected(foreground)
    markers = np.zeros(reference_labels.shape, dtype=np.int32)
    requested_labels = np.unique(reference_labels)
    requested_labels = requested_labels[requested_labels > 0].astype(np.int32, copy=False)
    regions = {
        int(label_id): np.column_stack(np.nonzero(reference_labels == int(label_id))[::-1]).astype(np.int32)
        for label_id in requested_labels.tolist()
    }
    if seed_regions is not None:
        for label_id, coordinates in seed_regions.items():
            coords = np.asarray(coordinates)
            if coords.size == 0:
                regions[int(label_id)] = np.empty((0, 2), dtype=np.int32)
                continue
            if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
                raise ValueError("Watershed seed regions must contain finite (x, y) coordinates.")
            coords = np.unique(np.rint(coords).astype(np.int32), axis=0)
            valid = ((coords[:, 0] >= 0) & (coords[:, 0] < signal_img.shape[1])
                     & (coords[:, 1] >= 0) & (coords[:, 1] < signal_img.shape[0]))
            regions[int(label_id)] = coords[valid]
        requested_labels = np.asarray(sorted(set(requested_labels.tolist()) | set(regions)), dtype=np.int32)

    for label_id in requested_labels.tolist():
        region = regions.get(label_id, np.empty((0, 2), dtype=np.int32))
        if region.size == 0:
            continue
        eligible = region[foreground[region[:, 1], region[:, 0]]]
        if eligible.size == 0:
            continue
        point = np.asarray((seed_points or {}).get(label_id, region.mean(axis=0)), dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("Watershed seed points must be finite (x, y) coordinates.")
        distances = ((eligible[:, 0] - point[0]) ** 2 + (eligible[:, 1] - point[1]) ** 2)
        nearest_eligible = eligible[int(np.argmin(distances))]
        component_id = int(components[nearest_eligible[1], nearest_eligible[0]])
        available = eligible[markers[eligible[:, 1], eligible[:, 0]] == 0]
        if available.size == 0:
            ys, xs = np.nonzero((components == component_id) & (markers == 0))
            available = np.column_stack([xs, ys])
        if available.size:
            nearest = np.argmin((available[:, 0] - point[0]) ** 2 + (available[:, 1] - point[1]) ** 2)
            x, y = available[int(nearest)]
            markers[y, x] = label_id
    active_labels = np.unique(markers)
    active_labels = active_labels[active_labels > 0].astype(np.int32, copy=False)
    unresolved_labels = np.setdiff1d(requested_labels, active_labels, assume_unique=True)

    if active_labels.size == 0:
        labels = np.zeros(signal_img.shape, dtype=np.int32)
    else:
        labels = np.zeros(signal_img.shape, dtype=np.int32)
        for component_id in range(1, component_count + 1):
            ys, xs = np.nonzero(components == component_id)
            seed_y, seed_x = np.nonzero((components == component_id) & (markers > 0))
            if seed_x.size == 0:
                continue
            seed_labels = markers[seed_y, seed_x]
            order = np.argsort(seed_labels, kind="stable")
            seed_x, seed_y, seed_labels = seed_x[order], seed_y[order], seed_labels[order]
            seed_coordinates = np.asarray([
                (seed_points or {}).get(int(label_id), [x, y])
                for label_id, x, y in zip(seed_labels, seed_x, seed_y)
            ], dtype=float)
            best_region = np.full(xs.shape, np.inf)
            best_centroid = np.full(xs.shape, np.inf)
            owners = np.zeros(xs.shape, dtype=np.int32)
            for label_id, centroid in zip(seed_labels, seed_coordinates):
                region = regions[int(label_id)]
                region_distance = np.min(
                    (xs[:, None] - region[None, :, 0]) ** 2
                    + (ys[:, None] - region[None, :, 1]) ** 2,
                    axis=1,
                )
                centroid_distance = ((xs - centroid[0]) ** 2 + (ys - centroid[1]) ** 2)
                replace = ((region_distance < best_region)
                           | ((region_distance == best_region) & (centroid_distance < best_centroid)))
                owners[replace] = int(label_id)
                best_region[replace] = region_distance[replace]
                best_centroid[replace] = centroid_distance[replace]
            labels[ys, xs] = owners

    if max_growth_distance is not None and active_labels.size > 0:
        for label_id in active_labels.tolist():
            footprint = np.zeros(reference_labels.shape, dtype=bool)
            region = regions[int(label_id)]
            footprint[region[:, 1], region[:, 0]] = True
            distance = distance_transform_edt(~footprint)
            labels[(labels == int(label_id)) & (distance > max_growth_distance)] = 0

    labels[~foreground] = 0
    pixel_counts = {
        int(label_id): int(np.count_nonzero(labels == int(label_id)))
        for label_id in requested_labels.tolist()
    }
    return WatershedResult(
        labels=labels,
        foreground_mask=foreground,
        active_labels=active_labels,
        unresolved_labels=unresolved_labels,
        pixel_counts=pixel_counts,
    )


def subpixel_watershed(signal, microscope_mask, translation, scale, threshold=160,
                       max_growth_distance=None, subdivisions=8):
    """Partition sensor foreground by complete microscope footprints on a finer grid.

    Sensor values are repeated without interpolation. Native labels become full
    region seeds; flat watershed grows them through four-neighbour foreground
    paths, never across background gaps. Fractions use the entire sensor area
    as denominator.
    """
    from ..image_fitting.microscope import project_mask_to_epic

    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2 or subdivisions < 1:
        raise ValueError("Signal must be 2D and subdivisions must be positive.")
    factor = int(subdivisions)
    foreground = np.isfinite(signal) & (signal > float(threshold))
    fine_foreground = np.repeat(np.repeat(foreground, factor, axis=0), factor, axis=1)
    projected = project_mask_to_epic(microscope_mask, translation, scale, fine_foreground.shape)
    components, count = label_connected(fine_foreground)
    owners = np.zeros(projected.shape, dtype=np.int32)
    for component_id in range(1, count + 1):
        ys, xs = np.nonzero(components == component_id)
        if not len(xs):
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        component = components[y0:y1, x0:x1] == component_id
        seeds = np.where(component, projected[y0:y1, x0:x1], 0)
        if not np.any(seeds):
            continue
        assigned = watershed(np.zeros(component.shape, dtype=np.uint8), seeds,
                             mask=component, connectivity=1)
        keep = component
        if max_growth_distance is not None:
            distances = distance_transform_edt(seeds == 0)
            keep = keep & (distances <= float(max_growth_distance) * factor)
        owners[y0:y1, x0:x1][keep] = assigned[keep]
    participation = {}
    for y in range(signal.shape[0]):
        for x in range(signal.shape[1]):
            block = owners[y * factor:(y + 1) * factor, x * factor:(x + 1) * factor]
            label_ids, counts = np.unique(block[block > 0], return_counts=True)
            for label_id, area in zip(label_ids.tolist(), counts.tolist()):
                participation.setdefault(int(label_id), []).append((x, y, area / float(factor * factor)))
    return owners, {
        label_id: np.asarray(rows, dtype=np.float32) for label_id, rows in participation.items()
    }


def watershed_segmentation(well, coords, ws_threshold=160, distance_threshold=np.inf, mask=None):
    if well.ndim == 3:
        well_img = np.max(well, axis=0)
    elif well.ndim != 2:
        raise ValueError("Input well data must be 2D or 3D numpy array.")
    else:
        well_img = well.copy()
    
    marker_labels = np.zeros(well_img.shape, dtype=np.int32)
    coords_idx = np.rint(coords).astype(np.int32)
    coords_idx[:, 0] = np.clip(coords_idx[:, 0], 0, well_img.shape[1] - 1)
    coords_idx[:, 1] = np.clip(coords_idx[:, 1], 0, well_img.shape[0] - 1)
    if mask is not None:
        marker_labels = np.asarray(mask, dtype=np.int32)
    else:
        for i, (x_coord, y_coord) in enumerate(coords_idx):
            marker_labels[y_coord, x_coord] = i + 1

    max_growth_distance = None if np.isinf(distance_threshold) else distance_threshold
    return threshold_bounded_watershed(
        well_img,
        marker_labels,
        threshold=ws_threshold,
        max_growth_distance=max_growth_distance,
    ).labels
