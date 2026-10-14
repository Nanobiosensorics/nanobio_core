"""Observed levels, bounded gap extrapolation, and background continuity."""

from dataclasses import replace
import json

import numpy as np
import pytest

from nanobio_core.preprocessing import PreprocessConfig, RawWell
from nanobio_core.preprocessing.config import PhaseOffsetConfig
from nanobio_core.preprocessing.model import (
    StageState, OBSERVED, TEMPORAL, SPATIAL, STITCHED, CURATED, MISSING, EXCLUDED,
)
from nanobio_core.preprocessing import stages


def long_gap_state():
    time = np.arange(60, dtype=float) * 9
    time[30:] += 3600
    rng = np.random.default_rng(7)
    trace = np.concatenate([0.08 * (time[:30] - np.median(time[25:30])),
                            400 + 0.12 * (time[30:] - np.median(time[30:35]))])
    values = np.broadcast_to(trace[:, None, None], (60, 3, 3)).copy()
    values += rng.normal(0, 0.5, values.shape)
    allowed = np.ones((3, 3), bool)
    allowed[0, 0] = False
    raw = RawWell(values.copy(), np.ones(values.shape, np.uint16), time, (30,), allowed)
    method = np.full(values.shape, OBSERVED, np.uint8)
    method[:, ~allowed] = EXCLUDED
    return StageState(raw, values, method, allowed, {})


def test_level_recovers_additive_offset_with_noise_and_long_gap_slopes():
    state = long_gap_state()
    config = PreprocessConfig()
    result = stages.phase_offset(state, config)
    offsets = np.asarray(result.receipts['phase_offset']['maps'][0]['offset_pm'])
    assert np.max(np.abs(offsets[state.allowed] + 400)) < 2
    np.testing.assert_allclose(np.median(result.values[25:30, state.allowed], axis=0),
                               np.median(result.values[30:35, state.allowed], axis=0), atol=1e-12)
    np.testing.assert_array_equal(result.values[:30], state.values[:30])
    assert result.values[:, ~state.allowed].tobytes() == state.values[:, ~state.allowed].tobytes()
    np.testing.assert_array_equal(result.method, state.method)


def test_extrapolation_cap_bounds_slope_amplification():
    state = long_gap_state()
    config = PreprocessConfig(phase_offset=PhaseOffsetConfig('extrapolate'))
    result = stages.phase_offset(state, config)
    offsets = np.asarray(result.receipts['phase_offset']['maps'][0]['offset_pm'])
    assert np.max(np.abs(offsets[state.allowed] + 400)) < 100
    unlimited = replace(config, phase_offset=replace(config.phase_offset, max_extrapolation_s=10000))
    full = stages.phase_offset(state, unlimited)
    full_offsets = np.asarray(full.receipts['phase_offset']['maps'][0]['offset_pm'])
    assert np.min(np.abs(full_offsets[state.allowed] + 400)) > 300
    # The cap starts at the nearest valid sample, even when endpoints are unavailable.
    state.method[29:31, 1, 1] = MISSING
    bounded = stages.phase_offset(state, config)
    pre_ids, post_ids = np.arange(24, 29), np.arange(31, 36)
    time = state.raw.time_s
    pre = np.polyfit(time[pre_ids] - (time[28] + 300), state.values[pre_ids, 1, 1], 1)
    post = np.polyfit(time[post_ids] - (time[31] - 300), state.values[post_ids, 1, 1], 1)
    assert bounded.receipts['phase_offset']['maps'][0]['offset_pm'][1][1] == pytest.approx(pre[1] - post[1])


@pytest.mark.parametrize('method', [OBSERVED, TEMPORAL, SPATIAL, STITCHED, CURATED])
def test_level_uses_finite_observed_or_repaired_anchors_and_flags_insufficient_support(method):
    state = long_gap_state()
    state.method[25:35, 1, 1] = method
    state.values[29:31, 1, 1] = np.nan
    state.method[30:, 2, 2] = MISSING
    state.method[30:34, 2, 2] = OBSERVED
    result = stages.phase_offset(state, PreprocessConfig())
    receipt = result.receipts['phase_offset']['maps'][0]
    expected = np.median(state.values[24:29, 1, 1]) - np.median(state.values[31:36, 1, 1])
    assert receipt['offset_pm'][1][1] == pytest.approx(expected)
    assert receipt['unsupported_pixels'] == 1
    np.testing.assert_array_equal(result.values[:, 2, 2], state.values[:, 2, 2])


@pytest.mark.parametrize('aggregation', ['mean', 'median'])
def test_background_continuity_with_different_pick_slopes(aggregation):
    state = long_gap_state()
    # Independent pick slopes make the old extrapolated offsets move a zero background.
    time = state.raw.time_s
    state.values[:30, 1, 1] = 0.04 * (time[:30] - time[29])
    state.values[30:, 1, 1] = 250 - 0.16 * (time[30:] - time[30])
    state.values[:, 2, 1] += 25
    config = PreprocessConfig()
    config = replace(config, background=replace(config.background, method=aggregation,
                                                coordinates=((1, 1), (1, 2), (2, 1))))
    background = stages.background(state, config)
    result = stages.phase_offset(background, config)
    check = result.receipts['phase_offset']['background_checks'][0]
    assert check['frame'] == 30
    assert check['passed']
    assert abs(check['background_step_pm']) <= check['tolerance_pm']
    unchanged = stages.phase_offset(background, replace(config, phase_offset=PhaseOffsetConfig('none')))
    np.testing.assert_array_equal(unchanged.values, background.values)
    # A failed audit remains visible rather than reporting fitted midpoint agreement.
    extrapolate = replace(config, phase_offset=PhaseOffsetConfig('extrapolate', max_extrapolation_s=10000))
    old = stages.phase_offset(background, extrapolate)
    if aggregation == 'median':
        assert not old.receipts['phase_offset']['background_checks'][0]['passed']


def test_background_continuity_reports_unavailable_support():
    state = long_gap_state()
    config = PreprocessConfig(background=replace(PreprocessConfig().background, coordinates=((1, 1),)))
    state.method[30:, 1, 1] = MISSING
    state.values[30:, 1, 1] = np.nan
    result = stages.phase_offset(stages.background(state, config), config)
    check = result.receipts['phase_offset']['background_checks'][0]
    assert check['background_step_pm'] is None
    assert check['passed'] is None


@pytest.mark.parametrize('mode', ['none', 'single', 'level', 'extrapolate', 'robust'])
def test_config_roundtrip_canonicalizes_robust_alias(mode):
    saved = PreprocessConfig().to_dict()
    saved['phase_offset'] = dict(mode=mode, anchor_frames=5)
    restored = PreprocessConfig.from_dict(json.loads(json.dumps(saved)))
    assert restored.phase_offset.mode == ('extrapolate' if mode == 'robust' else mode)
    assert restored.phase_offset.max_extrapolation_s == 300
    assert PreprocessConfig.from_dict(restored.to_dict()) == restored
    custom = replace(restored, phase_offset=replace(restored.phase_offset, max_extrapolation_s=123.4))
    assert PreprocessConfig.from_dict(custom.to_dict()) == custom
    assert PreprocessConfig.profile('default').phase_offset.mode == 'level'
    assert PreprocessConfig.profile('observed_only').phase_offset.mode == 'level'
    assert PreprocessConfig.profile('legacy').phase_offset.mode == 'single'
    if mode == 'robust':
        canonical = PreprocessConfig(phase_offset=PhaseOffsetConfig('extrapolate'))
        assert restored.config_hash() == canonical.config_hash()


@pytest.mark.parametrize('limit', [-1, np.inf, np.nan])
def test_invalid_extrapolation_cap_rejected(limit):
    with pytest.raises(ValueError, match='Maximum extrapolation'):
        PreprocessConfig(phase_offset=PhaseOffsetConfig(max_extrapolation_s=limit))
