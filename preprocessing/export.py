"""Quantitative sums with explicit support fractions."""

import warnings

import numpy as np
from .model import OBSERVED, TEMPORAL, SPATIAL, STITCHED, CURATED


def weighted_region_sum(result, region, min_available_weight=0.5, *, min_observed_weight=None,
                        return_method_fractions=False):
    """Return (sum, observed_fraction, reconstructed_fraction) per frame.

    The available threshold is relative to total requested weight. Missing weights
    are never rescaled; an unsupported sum is NaN rather than silently reduced.
    Set return_method_fractions=True to append a dictionary of per-method fractions.
    """
    if min_observed_weight is not None:
        warnings.warn('min_observed_weight is deprecated; use min_available_weight',
                      DeprecationWarning, stacklevel=2)
        min_available_weight = min_observed_weight
    region = np.asarray(region, dtype=float)
    if region.ndim != 2 or region.shape[1] != 3 or not np.isfinite(region).all():
        raise ValueError('Region must contain finite (x, y, fraction) rows')
    x, y = region[:, 0].astype(int), region[:, 1].astype(int)
    weights = region[:, 2]
    if (np.any(weights < 0) or weights.sum() <= 0 or np.any(region[:, :2] != region[:, :2].astype(int))
            or np.any(x < 0) or np.any(y < 0) or np.any(x >= result.values_pm.shape[2])
            or np.any(y >= result.values_pm.shape[1]) or not 0 <= min_available_weight <= 1):
        raise ValueError('Invalid region coordinates, weights or available threshold')
    values = result.values_pm[:, y, x]
    method = result.method[:, y, x]
    observed = (method == OBSERVED) & np.isfinite(values)
    reconstructed = np.isin(method, [TEMPORAL, SPATIAL, STITCHED, CURATED]) & np.isfinite(values)
    available = observed | reconstructed
    observed_fraction = (observed * weights).sum(axis=1) / weights.sum()
    reconstructed_fraction = (reconstructed * weights).sum(axis=1) / weights.sum()
    available_fraction = observed_fraction + reconstructed_fraction
    value = np.sum(np.where(available, values, 0) * weights, axis=1)
    value[(available_fraction < min_available_weight) | (available_fraction <= 0)] = np.nan
    if return_method_fractions:
        fractions = {name: (((method == code) & np.isfinite(values)) * weights).sum(axis=1) / weights.sum()
                     for name, code in [('observed', OBSERVED), ('temporal', TEMPORAL), ('spatial', SPATIAL),
                                        ('stitched', STITCHED), ('curated', CURATED)]}
        return value, observed_fraction, reconstructed_fraction, fractions
    return value, observed_fraction, reconstructed_fraction
