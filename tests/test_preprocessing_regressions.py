"""Regressions from the independent July/March preprocessing verification."""

from dataclasses import replace
import json

import numpy as np
import pytest

from nanobio_core.preprocessing import PreprocessConfig, RawWell
from nanobio_core.preprocessing.model import (
    StageState, OBSERVED, TEMPORAL, SPATIAL, STITCHED, CURATED, MISSING, EXCLUDED,
)
from nanobio_core.preprocessing import stages, pipeline
from nanobio_core.preprocessing.recovery import recover_pixels


@pytest.mark.parametrize('method', [TEMPORAL, SPATIAL, STITCHED, CURATED])
@pytest.mark.parametrize('aggregation', ['mean', 'median'])
def test_background_keeps_finite_repaired_picks(method, aggregation):
    config = PreprocessConfig()
    config = replace(config, background=replace(config.background, method=aggregation,
                                                coordinates=((1, 1), (2, 2))))
    values = np.full((12, 4, 4), 40.0)
    values[:, 1, 1] = 10
    values[:, 2, 2] = 20
    methods = np.full(values.shape, OBSERVED, np.uint8)
    methods[3:9, 1, 1] = method
    methods[3:9, 2, 2] = method
    raw = RawWell(values.copy(), np.ones(values.shape, np.uint16), np.arange(12, dtype=float),
                  (), np.ones((4, 4), bool), 'C1')
    state = StageState(raw, values, methods, raw.allowed, {})
    result = stages.background(state, config)
    np.testing.assert_array_equal(result.receipts['background']['trace_pm'], np.full(12, 15.0))
    assert np.isfinite(result.values).all()
    np.testing.assert_array_equal(result.method, state.method)
    np.testing.assert_array_equal(result.values[3:9, 0, 0], np.full(6, 25.0))


@pytest.mark.parametrize('later_observations', [(), (1, 3)])
def test_recovery_qualification_never_depends_on_previous_phase(later_observations):
    time = np.arange(24, dtype=float)
    time[12:] += 100
    values = np.broadcast_to(800 + np.arange(24)[:, None, None] * 0.001, (24, 5, 5)).copy()
    valid = np.ones(values.shape, bool)
    valid[12:, 2, 2] = False
    for frame in later_observations:
        valid[12 + frame, 2, 2] = True
    allowed = np.ones((5, 5), bool)
    full = recover_pixels(values, valid, time, (12,), allowed)
    changed = valid.copy()
    changed[:12, 2, 2] = False
    altered = recover_pixels(values, changed, time, (12,), allowed)
    independent = recover_pixels(values[12:], valid[12:], time[12:], (), allowed)
    np.testing.assert_array_equal(full.values[12:], altered.values[12:])
    np.testing.assert_array_equal(full.methods[12:], altered.methods[12:])
    np.testing.assert_array_equal(full.values[12:], independent.values)
    np.testing.assert_array_equal(full.methods[12:], independent.methods)


@pytest.mark.parametrize('range_type, ranges, expected', [
    (0, [2, None], (4, 8)), (0, [1, 3], (2, 6)), (1, [3, 7], (3, 7)),
])
def test_loader_matches_saved_project_orientation_and_selected_range(
        tmp_path, monkeypatch, range_type, ranges, expected):
    time = np.array([0, 9, 100, 109, 200, 209, 300, 309], float)
    yy, xx = np.indices((240, 320))
    values = 800 + yy[None] * 0.01 + xx[None] * 0.001 + np.arange(8)[:, None, None]
    codes = np.broadcast_to(yy[None] * 320 + xx[None], values.shape).astype(np.uint16).copy()
    (tmp_path / 'DMR').mkdir()
    for frame, code in enumerate(codes, 1):
        code.tofile(tmp_path / 'DMR' / str(frame))
    (tmp_path / 'test_WL_Power').write_bytes(bytes(614400))
    metadata = tmp_path / 'result/.metadata'
    metadata.mkdir(parents=True)
    parameters = dict(flip=[False, True], preprocessing=dict(
        flip=[True, False], signal_range=dict(range_type=range_type, ranges=ranges)))
    (metadata / 'parameters.json').write_text(json.dumps(parameters))
    monkeypatch.setattr(pipeline.measurement_load, 'load_measurement', lambda path: (values, time))
    loaded, saved = pipeline.load_raw_measurement(tmp_path)
    left, right = expected
    well = loaded['C1']
    np.testing.assert_array_equal(well.values_nm, np.flip(values[left:right, :80, :80], axis=1))
    np.testing.assert_array_equal(well.codes, np.flip(codes[left:right, :80, :80], axis=1))
    np.testing.assert_array_equal(well.time_s, time[left:right])
    assert well.phases == tuple(p - left for p in (2, 4, 6) if left < p < right)
    assert saved == parameters
