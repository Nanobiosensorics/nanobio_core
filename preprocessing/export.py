"""Quantitative sums with explicit support fractions."""

import numpy as np
from .model import OBSERVED, TEMPORAL, SPATIAL, STITCHED, CURATED


def weighted_region_sum(result, region, min_observed_weight=0.5):
    """Return (sum, observed_fraction, reconstructed_fraction) per frame.

    The observed threshold is relative to total requested weight. Missing weights
    are never rescaled; an unsupported sum is NaN rather than silently reduced.
    """
    region = np.asarray(region, dtype=float)
    if region.ndim != 2 or region.shape[1] != 3 or not np.isfinite(region).all():
        raise ValueError('Region must contain finite (x, y, fraction) rows')
    x, y = region[:, 0].astype(int), region[:, 1].astype(int)
    weights = region[:, 2]
    if (np.any(weights < 0) or weights.sum() <= 0 or np.any(region[:, :2] != region[:, :2].astype(int))
            or np.any(x < 0) or np.any(y < 0) or np.any(x >= result.values_pm.shape[2])
            or np.any(y >= result.values_pm.shape[1]) or not 0 <= min_observed_weight <= 1):
        raise ValueError('Invalid region coordinates, weights or observed threshold')
    values = result.values_pm[:, y, x]
    method = result.method[:, y, x]
    observed = (method == OBSERVED) & np.isfinite(values)
    reconstructed = np.isin(method, [TEMPORAL, SPATIAL, STITCHED, CURATED]) & np.isfinite(values)
    available = observed | reconstructed
    observed_fraction = (observed * weights).sum(axis=1) / weights.sum()
    reconstructed_fraction = (reconstructed * weights).sum(axis=1) / weights.sum()
    value = np.sum(np.where(available, values, 0) * weights, axis=1)
    value[(observed_fraction < min_observed_weight) | ~available.any(axis=1)] = np.nan
    return value, observed_fraction, reconstructed_fraction
