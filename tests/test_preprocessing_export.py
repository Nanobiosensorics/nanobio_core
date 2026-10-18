"""Available region support includes repaired measurements without rescaling."""

import numpy as np
import pytest

from nanobio_core.preprocessing import (
    weighted_region_sum, PreprocessConfig, OBSERVED, TEMPORAL, SPATIAL, STITCHED,
    CURATED, MISSING, EXCLUDED,
)
from nanobio_core.preprocessing.model import PreprocessResult


@pytest.mark.parametrize('repair', [TEMPORAL, STITCHED, SPATIAL, CURATED])
def test_repaired_cell_pulse_remains_available(repair):
    values = np.array([[[100., 200.]], [[100., 200.]]])
    methods = np.array([[[OBSERVED, OBSERVED]], [[OBSERVED, repair]]], dtype=np.uint8)
    result = PreprocessResult(values, methods, None, {}, {})
    sums, observed, reconstructed, fractions = weighted_region_sum(
        result, [(0, 0, 0.25), (1, 0, 0.75)], return_method_fractions=True)
    np.testing.assert_array_equal(sums, [175, 175])
    np.testing.assert_array_equal(observed, [1, 0.25])
    np.testing.assert_array_equal(reconstructed, [0, 0.75])
    np.testing.assert_array_equal(sum(fractions.values()), observed + reconstructed)


@pytest.mark.parametrize('unavailable', [MISSING, EXCLUDED])
def test_cell_with_more_than_half_unavailable_weight_is_missing(unavailable):
    result = PreprocessResult(np.array([[[100., 200.]]]),
                              np.array([[[CURATED, unavailable]]], dtype=np.uint8), None, {}, {})
    sums, observed, reconstructed = weighted_region_sum(result, [(0, 0, 0.49), (1, 0, 0.51)])
    assert np.isnan(sums[0])
    assert observed[0] == 0 and reconstructed[0] == 0.49
    assert weighted_region_sum(result, [(0, 0, 0.5), (1, 0, 0.5)])[0][0] == 50


def test_export_deprecated_alias_and_saved_config():
    result = PreprocessResult(np.array([[[100.]]]), np.array([[[CURATED]]], dtype=np.uint8), None, {}, {})
    with pytest.warns(DeprecationWarning):
        sums = weighted_region_sum(result, [(0, 0, 1)], min_observed_weight=0.5)[0]
    assert sums[0] == 100
    with pytest.warns(DeprecationWarning):
        settings = PreprocessConfig.from_dict({'export': {'min_observed_weight': 0.75}})
    assert settings.export.min_available_weight == 0.75
    assert settings.to_dict()['export'] == {'min_available_weight': 0.75}
    assert PreprocessConfig.from_dict(settings.to_dict()) == settings


def test_zero_weight_available_pixel_cannot_create_a_sum():
    result = PreprocessResult(np.array([[[100., 200.]]]),
                              np.array([[[CURATED, MISSING]]], dtype=np.uint8), None, {}, {})
    assert np.isnan(weighted_region_sum(result, [(0, 0, 0), (1, 0, 1)], min_available_weight=0)[0][0])
