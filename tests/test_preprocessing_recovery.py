"""Synthetic detector benchmarks and compatibility of the split implementation."""

from dataclasses import replace
import numpy as np
import pytest

from nanobio_core.epic_cardio.pixel_recovery import recover_pixels as original_recovery
from nanobio_core.preprocessing.config import RecoveryConfig
from nanobio_core.preprocessing.model import OBSERVED, TEMPORAL, SPATIAL, STITCHED, EXCLUDED
from nanobio_core.preprocessing.recovery import (
    recover_pixels, detect_short_gaps, detect_terminal_dropouts, detect_returning_pulses,
    detect_persistent_steps, spatial_fill,
)


def recording():
    time = np.arange(70, dtype=float)
    values = np.broadcast_to(800 + time[:, None, None] * 0.001, (70, 9, 9)).copy()
    valid = np.ones(values.shape, dtype=bool)
    allowed = np.ones((9, 9), dtype=bool)
    allowed[0] = False
    return values, valid, time, allowed


@pytest.mark.parametrize('artifact', ['pulse', 'gap', 'step', 'terminal', 'biological', 'mixed'])
def test_split_recovery_matches_original_values_and_codes(artifact):
    values, valid, time, allowed = recording()
    if artifact in ('pulse', 'mixed'):
        values[12:15, 4, 4] += 0.4
    if artifact in ('gap', 'mixed'):
        valid[8:10, 3, 3] = False
        values[8:10, 3, 3] = 0
    if artifact in ('step', 'mixed'):
        values[25:, 5:7, 5:7] += 0.3
    if artifact in ('terminal', 'mixed'):
        # Keep anchors in the later phase for historical equivalence. Entirely
        # missing phases intentionally no longer qualify using previous phases.
        valid[60:, 2, 2] = False
        values[60:, 2, 2] = 0
    if artifact == 'biological':
        values[25:, 4, 4] += 0.3
    old = original_recovery(values, valid, time, [50], allowed)
    new = recover_pixels(values, valid, time, [50], allowed)
    np.testing.assert_array_equal(new.values, old.values)
    np.testing.assert_array_equal(new.methods, old.methods)
    assert new.steps == old.steps


def test_each_detector_and_fixers_on_known_artifacts(monkeypatch):
    values, valid, time, allowed = recording()
    config = RecoveryConfig()
    values[12:15, 4, 4] += 0.4
    methods = np.zeros(values.shape, dtype=np.uint8)
    pulses = detect_returning_pulses(values[:, 4, 4], valid[:, 4, 4],
                                     np.diff(values[:, 4, 4]) * 1000, time, 0, methods, 4, 4, config)
    assert [(a, b) for a, b, _ in pulses] == [(12, 15)]
    assert not methods.any()
    known = np.ones(70, dtype=bool)
    known[12:15] = False
    assert detect_short_gaps(values[:, 3, 3], known, np.zeros(70, bool), config) == [(12, 15)]
    known[40:] = False
    assert detect_terminal_dropouts(known, config) == 40
    assert detect_terminal_dropouts(np.zeros(70, bool), config) is None
    assert detect_terminal_dropouts(np.ones(70, bool), config) is None
    values[25:, 5:7, 5:7] += 0.3
    calls = []
    from nanobio_core.preprocessing import recovery
    original = detect_persistent_steps

    def capture(*args):
        result = original(*args)
        calls.extend(result)
        return result

    monkeypatch.setattr(recovery, 'detect_persistent_steps', capture)
    result = recover_pixels(values, valid, time, (), allowed)
    assert any(step['applied'] and step['frame'] == 25 for step in calls)
    assert np.all(result.methods[12:15, 4, 4] == TEMPORAL)
    assert np.all(result.methods[25:, 5, 5] == STITCHED)
    np.testing.assert_allclose(result.values[:, 5, 5], 800 + time * 0.001, atol=1e-10)


def test_real_isolated_biological_step_is_not_altered():
    values, valid, time, allowed = recording()
    values[25:, 4, 4] += 0.3
    result = recover_pixels(values, valid, time, (), allowed)
    np.testing.assert_array_equal(result.values[:, 4, 4], values[:, 4, 4])
    assert not any(step['applied'] for step in result.steps)


def test_phase_isolation_excluded_bytes_and_original_spatial_donors():
    values, valid, time, allowed = recording()
    valid[35:, 4, 4] = False
    modified = values.copy()
    modified[:30] += 15
    a = recover_pixels(values, valid, time, (30,), allowed)
    b = recover_pixels(modified, valid, time, (30,), allowed)
    np.testing.assert_array_equal(a.values[30:], b.values[30:])
    assert a.values[:, ~allowed].tobytes() == values[:, ~allowed].tobytes()
    assert np.all(a.methods[40:, 4, 4] == SPATIAL)
    # A previous spatial estimate offered as a target cannot enclose a new hole.
    frame = np.full((9, 9), np.nan)
    frame[4, 4] = 100
    filled, radius, donors = spatial_fill(frame, np.zeros((9, 9), bool),
                                          np.ones((9, 9), bool), allowed, RecoveryConfig())
    assert not radius.any() and not donors.any()
    np.testing.assert_array_equal(filled, frame)


def test_synthetic_benchmark_sensitivity_false_positives_and_error():
    values, valid, time, allowed = recording()
    clean = values.copy()
    injected = np.zeros(values.shape, bool)
    for y, x in [(2, 2), (4, 4), (6, 6)]:
        values[12:15, y, x] += 0.4
        injected[12:15, y, x] = True
    result = recover_pixels(values, valid, time, (), allowed)
    classified = result.methods == TEMPORAL
    sensitivity = np.count_nonzero(classified & injected) / injected.sum()
    false_positive_rate = np.count_nonzero(classified & ~injected) / (~injected).sum()
    error_pm = np.abs(result.values[injected] - clean[injected]) * 1000
    assert sensitivity == 1
    assert false_positive_rate == 0
    assert error_pm.max() < 1e-6


@pytest.mark.parametrize('artifact, method', [('gap', TEMPORAL), ('step', STITCHED), ('tail', SPATIAL)])
def test_per_detector_benchmark_sensitivity_false_positives_and_error(artifact, method):
    values, valid, time, allowed = recording()
    clean = values.copy()
    injected = np.zeros(values.shape, bool)
    if artifact == 'gap':
        valid[12:15, 4, 4] = False
        injected[12:15, 4, 4] = True
    elif artifact == 'step':
        values[25:, 4:6, 4:6] += 0.3
        injected[25:, 4:6, 4:6] = True
    else:
        valid[40:, 4, 4] = False
        injected[40:, 4, 4] = True
    result = recover_pixels(values, valid, time, (), allowed)
    classified = result.methods == method
    assert np.count_nonzero(classified & injected) / injected.sum() == 1
    assert np.count_nonzero(classified & ~injected) == 0
    error_pm = np.abs(result.values[injected] - clean[injected]) * 1000
    assert error_pm.max() < 1e-6
