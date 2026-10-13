"""Automatic sensor-boundary detection from unprocessed wavelength maps."""

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu

from .border_filter import border_filter_for_well


def active_area(image, margin=2):
    """Return the raw active footprint and its inward sensor-pixel margin."""
    if margin not in (1, 2):
        raise ValueError('Use a one- or two-pixel inward margin')
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError('Border detection requires a 2D raw wavelength map')
    finite = np.isfinite(image)
    values = image[finite]
    if not len(values):
        raise ValueError('No finite raw wavelength pixels')
    separated = float(np.median(values) - values.min()) > 10.0
    threshold = float(threshold_otsu(values)) if separated else None
    active = finite & (image > threshold) if separated else finite
    distance = distance_transform_edt(np.pad(active, 1))[1:-1, 1:-1]
    allowed = active & (distance > margin)
    if allowed.sum() < 0.5 * image.size:
        raise ValueError(f'Unexpectedly small active sensor area: {allowed.sum()} pixels')
    return active, allowed, threshold


def encode_mask(mask):
    return [''.join('1' if value else '0' for value in row) for row in mask]


def rectangle_trims(mask):
    yy, xx = np.nonzero(mask)
    if not len(yy):
        raise ValueError('Detected active sensor area is empty')
    return dict(top=int(yy.min()), bottom=int(mask.shape[0] - 1 - yy.max()),
                left=int(xx.min()), right=int(mask.shape[1] - 1 - xx.max()))


def detect_active_areas(raw_wells, margin=2):
    """Detect from full-acquisition frame zero, before signal-range slicing."""
    records = {}
    for well, cube in raw_wells.items():
        cube = np.asarray(cube)
        if cube.ndim != 3 or not len(cube):
            raise ValueError(f'{well}: border detection requires a nonempty raw sensor cube')
        _, allowed, threshold = active_area(cube[0], margin)
        records[well] = dict(rows=encode_mask(allowed), inward_margin_pixels=margin,
                             raw_wavelength_threshold=threshold, source_frame=0,
                             method='raw_wavelength_otsu', border=rectangle_trims(allowed))
    return records


def apply_detected_border_filters(localization_params, records):
    """Save detected masks while keeping any stricter existing manual trims."""
    if not records:
        return
    borders = {}
    for well, record in records.items():
        existing = border_filter_for_well(localization_params, well)
        borders[well] = {side: max(existing[side], value) for side, value in record['border'].items()}
    localization_params.setdefault('filter_border_per_well', {}).update(borders)
    localization_params.setdefault('filter_active_area_per_well', {}).update(records)
    localization_params['border_filter_mode'] = 'per-well'
