"""Reviewed well pulses and observed-only background availability."""

from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from nanobio_core.epic_cardio.confirmed_pulse import repair_confirmed_pulse
from nanobio_core.preprocessing import Cache, PreprocessConfig, RawWell, preprocess_well
from nanobio_core.preprocessing import cli, stages
from nanobio_core.preprocessing.curation import load_curation, pulse_shape_correlations
from nanobio_core.preprocessing.model import StageState, OBSERVED, CURATED, MISSING, EXCLUDED


def interval(**overrides):
    return dict(well='C1', start_frame=12, end_frame=18, reviewer='Reviewer', date='2026-10-14',
                reason='Reviewed acquisition pulse', method=overrides.pop('method', 'common_mode'),
                **overrides)


def common_pulse():
    rng = np.random.default_rng(42)
    time = np.cumsum(rng.uniform(0.8, 1.2, 36))
    slopes = np.linspace(0.0001, 0.002, 64).reshape(8, 8)
    trend = 800 + time[:, None, None] * slopes
    values = trend + rng.normal(0, 0.0005, trend.shape)
    shape = np.array([0.2, 0.8, 1.0, 0.7, 0.4, 0.1])
    scales = np.linspace(0.2, 1.6, 64).reshape(8, 8)
    scales[3, 3] = 0.8
    values[12:18] += shape[:, None, None] * scales
    transient = np.array([30, -20, -10, 20, -30, 10]) / 1000
    values[12:18, 1, 1] = trend[12:18, 1, 1] + transient
    values[12:18, 1, 2] = trend[12:18, 1, 2] + transient * 10
    # A strongly correlated pixel with excess non-common residual uses its line.
    values[12:18, 2, 2] += 0.04
    codes = np.ones(values.shape, np.uint16)
    codes[6, 1, 3] = 0
    allowed = np.ones((8, 8), bool)
    allowed[0, 0] = False
    return RawWell(values, codes, time, (28,), allowed, 'C1'), trend, scales, shape


@pytest.mark.parametrize('profile', ['default', 'observed_only'])
def test_common_mode_scaled_pulse_noise_transient_and_missing_anchors(profile):
    raw, trend, scales, shape = common_pulse()
    config = PreprocessConfig.profile(profile)
    state = stages.validity(raw, config, [interval()])
    state = stages.within_phase_recovery(state, config)
    repaired = stages.curated_repairs(state, config, [interval()])
    receipt = repaired.receipts['curated_repairs'][0]
    mask = np.all(repaired.method[12:18] == CURATED, axis=0)
    assert mask.sum() == 60
    assert mask[3, 3]  # 800 pm pulse
    assert mask[7, 7]  # 1600 pm pulse, beyond the validity cutoff
    assert np.max(np.abs(repaired.values[12:18, mask] - trend[12:18, mask])) * 1000 < 6
    anchors = np.r_[np.arange(6, 12), np.arange(18, 24)]
    design = np.column_stack([raw.time_s[anchors] - raw.time_s[12], np.ones(len(anchors))])
    fit = np.linalg.lstsq(design, state.values[anchors][:, mask], rcond=None)[0]
    line = np.column_stack([raw.time_s[12:18] - raw.time_s[12], np.ones(6)]) @ fit
    residual = np.max(np.abs(repaired.values[12:18, mask] - line), axis=0) * 1000
    assert np.all(residual <= receipt['noise_limit_pm'][mask] + 1e-7)
    np.testing.assert_array_equal(repaired.values[12:18, 1, 1], raw.values_nm[12:18, 1, 1])
    assert np.all(repaired.method[12:18, 1, 1] == OBSERVED)
    assert np.all(repaired.method[12:18, 1, 2:4] == MISSING)
    assert np.isnan(repaired.values[12:18, 1, 2:4]).all()
    assert receipt['outcomes'][1, 2] == 'uncorrelated_pulse'
    assert receipt['outcomes'][1, 3] == 'no_anchor'
    assert receipt['outcomes'][2, 2] == 'anchor_line'
    assert receipt['counts']['unchanged'] == 1
    assert receipt['counts']['no_anchor'] == 1
    assert receipt['counts']['uncorrelated_pulse'] == 1
    assert sum(receipt['counts'].values()) == raw.allowed.sum()
    assert receipt['shape_pm'].shape == (6,)
    assert receipt['amplitude_map'].shape == raw.allowed.shape
    assert receipt['correlation_map'][3, 3] > 0.99
    assert receipt['max_residual_after_pm'] == pytest.approx(residual.max())
    np.testing.assert_array_equal(repaired.values[:12], state.values[:12])
    np.testing.assert_array_equal(repaired.values[18:], state.values[18:])
    assert repaired.values[:, ~raw.allowed].tobytes() == raw.values_nm[:, ~raw.allowed].tobytes()
    assert np.all(repaired.method[:, ~raw.allowed] == EXCLUDED)
    assert state.acquired_valid[18, 7, 7]  # falling edge is not an invalid clean anchor


def test_common_mode_pipeline_receipt_cache_and_cli_roundtrip(tmp_path, monkeypatch):
    raw, _, _, _ = common_pulse()
    config = PreprocessConfig.profile('observed_only')
    config = replace(config, background=replace(config.background, method='none'),
                     registration=replace(config.registration, enabled=False))
    monkeypatch.setattr(stages, 'detect_active_area', lambda frame, margin: (None, raw.allowed, 0))
    result = preprocess_well(raw, config, [interval()])
    assert np.all(result.method[12:18, 7, 7] == CURATED)
    cache = Cache(tmp_path / 'cache')
    cache.put(result)
    saved = cache.get(result.provenance)
    np.testing.assert_array_equal(saved.values_pm, result.values_pm)
    for key in ('shape_pm', 'amplitude_map', 'correlation_map', 'outcomes'):
        assert isinstance(saved.receipts['curated_repairs'][0][key], np.ndarray)
        np.testing.assert_array_equal(saved.receipts['curated_repairs'][0][key],
                                      result.receipts['curated_repairs'][0][key])
    monkeypatch.setattr(cli, 'preprocess_measurement', lambda *args, **kwargs: {'C1': result})
    cli.main(['review', str(tmp_path / 'measurement'), '--out', str(tmp_path / 'review')])
    report = json.loads((tmp_path / 'review' / 'review-receipts.json').read_text())
    assert len(report['C1']['receipts']['curated_repairs'][0]['shape_pm']) == 6


def test_per_pixel_matches_previous_curation_bit_for_bit():
    raw, _, _, _ = common_pulse()
    config = PreprocessConfig()
    state = stages.within_phase_recovery(stages.validity(raw, config), config)
    settings = asdict(config.curation)
    settings.pop('method')
    settings.pop('min_correlation')
    for key in ('max_rank', 'basis_variance', 'min_explained'):
        settings.pop(key)
    _, accepted, receipt = repair_confirmed_pulse(
        state.acquired_values, state.acquired_valid, raw.time_s, raw.phases, state.allowed,
        12, 18, **settings)
    accepted &= np.isfinite(state.values[11]) & np.isfinite(state.values[18])
    fraction = (raw.time_s[12:18] - raw.time_s[11]) / (raw.time_s[18] - raw.time_s[11])
    interpolation = state.values[11][None] + fraction[:, None, None] * (state.values[18] - state.values[11])
    expected = state.copy()
    expected.values[12:18, accepted] = interpolation[:, accepted]
    expected.method[12:18, accepted] = CURATED
    actual = stages.curated_repairs(state, config, [interval(method='per_pixel')])
    assert accepted.any()
    assert actual.values.tobytes() == expected.values.tobytes()
    np.testing.assert_array_equal(actual.method, expected.method)
    receipt['repaired_pixels'] = int(accepted.sum())
    for key, value in receipt.items():
        assert actual.receipts['curated_repairs'][0][key] == value
    # Explicit per_pixel entries retain the original validity gate, too.
    np.testing.assert_array_equal(stages.validity(raw, config, [interval(method='per_pixel')]).values,
                                  stages.validity(raw, config).values)


@pytest.mark.parametrize('method', ['low_rank', 'common_mode', 'per_pixel'])
def test_curation_yaml_method_correlation_and_config(tmp_path, method):
    path = tmp_path / 'curation.yml'
    path.write_text('entries:\n  - ' + '\n    '.join(
        f'{key}: {value}' for key, value in interval(method=method, min_correlation=0.95).items()) + '\n')
    loaded = load_curation(path)
    assert loaded[0]['method'] == method
    assert loaded[0]['min_correlation'] == 0.95
    config = PreprocessConfig()
    assert config.curation.method == 'low_rank'
    assert config.curation.min_correlation == 0.8
    changed = replace(config, curation=replace(config.curation, method=method, min_correlation=0.95))
    assert PreprocessConfig.from_dict(changed.to_dict()) == changed
    assert config.config_hash() != changed.config_hash()


@pytest.mark.parametrize('entry_settings', [{'method': 'invalid'}, {'min_correlation': -0.1},
                                           {'min_correlation': 1.1}, {'min_correlation': float('nan')}])
def test_invalid_curation_settings_rejected(entry_settings):
    with pytest.raises(ValueError):
        load_curation([interval(**entry_settings)])
    with pytest.raises(ValueError):
        replace(PreprocessConfig(), curation=replace(PreprocessConfig().curation, **entry_settings))


def test_common_mode_phase_guard_and_no_shape():
    raw, _, _, _ = common_pulse()
    config = PreprocessConfig()
    state = stages.validity(raw, config, [interval()])
    with pytest.raises(ValueError, match='phase'):
        stages.curated_repairs(replace(state, raw=replace(raw, phases=(20,))), config, [interval()])
    state.method[6:12] = MISSING
    state.values[6:12] = np.nan
    result = stages.curated_repairs(state, config, [interval()])
    receipt = result.receipts['curated_repairs'][0]
    assert receipt['counts']['no_anchor'] == raw.allowed.sum()
    assert receipt['shape_amplitude_pm'] is None
    assert receipt['max_residual_after_pm'] is None
    assert np.isnan(receipt['shape_pm']).all()


def test_cross_well_shape_correlations():
    shape = np.array([1, 3, 2, 0], dtype=float)
    correlations = pulse_shape_correlations({'A1': shape, 'C1': shape * 2, 'B1': -shape,
                                             'A2': np.zeros(4)})
    assert correlations['A1']['C1'] == pytest.approx(1)
    assert correlations['A1']['B1'] == pytest.approx(-1)
    assert correlations['A1']['A2'] is None
    with pytest.raises(ValueError, match='matching'):
        pulse_shape_correlations({'A1': shape, 'C1': shape[:2]})


@pytest.mark.parametrize('aggregation', ['median', 'mean'])
def test_observed_only_background_fallback_and_phase_local_interpolation(aggregation):
    time = np.array([0, 1, 2, 4, 5, 6, 100, 101, 102, 103, 104, 105], dtype=float)
    values = np.broadcast_to((10 + time)[:, None, None], (12, 3, 3)).copy()
    methods = np.full(values.shape, OBSERVED, np.uint8)
    cell_mask = np.zeros((3, 3), bool)
    cell_mask[2, 2] = True
    values[:, 2, 2] = 1000  # genuine signal must not supply background
    values[2:4, 0, 0] = np.nan
    methods[2:4, 0, 0] = MISSING
    # At frame 3 no eligible OBSERVED donors remain; cell stays observed.
    methods[3, :2] = CURATED
    methods[3, 2, :2] = CURATED
    methods[3, 0, 0] = MISSING
    # Entire second phase has no background support, and must stay unsupported.
    methods[6:, :2] = CURATED
    methods[6:, 2, :2] = CURATED
    raw = RawWell(values.copy(), np.ones(values.shape, np.uint16), time, (6,),
                  np.ones((3, 3), bool), 'A1', cell_mask=cell_mask)
    state = StageState(raw, values, methods, raw.allowed, {})
    config = PreprocessConfig.profile('observed_only')
    config = replace(config, background=replace(config.background, method=aggregation, coordinates=((0, 0),)))
    result = stages.background(state, config)
    receipt = result.receipts['background']
    assert receipt['fallback_pixels'] == [dict(frame=2, pixel_count=7)]
    assert receipt['interpolated_frames'] == [dict(frame=3, anchors=[2, 4])]
    assert receipt['unavailable_frames'] == list(range(6, 12))
    np.testing.assert_allclose(receipt['trace_pm'][:6], 10 + time[:6])
    assert result.values[2, 2, 2] == 988
    assert result.values[3, 2, 2] == 986
    assert result.method[3, 2, 2] == OBSERVED
    np.testing.assert_array_equal(result.method[:6], state.method[:6])
    assert np.isnan(result.values[6:]).all()


def test_default_background_behavior_unchanged_when_pick_unavailable():
    values = np.full((5, 2, 2), 10.0)
    values[2, 0, 0] = np.nan
    methods = np.full(values.shape, OBSERVED, np.uint8)
    methods[2, 0, 0] = MISSING
    raw = RawWell(values, np.ones(values.shape, np.uint16), np.arange(5, dtype=float),
                  (), np.ones((2, 2), bool), 'C1')
    config = PreprocessConfig()
    config = replace(config, background=replace(config.background, coordinates=((0, 0),)))
    result = stages.background(StageState(raw, values, methods, raw.allowed, {}), config)
    assert np.isnan(result.values[2]).all()
    assert result.receipts['background']['fallback_pixels'] == []


def test_observed_only_without_background_selection_preserves_values():
    values = np.full((5, 2, 2), 100.0)  # no low-signal background candidates
    methods = np.full(values.shape, OBSERVED, np.uint8)
    raw = RawWell(values, np.ones(values.shape, np.uint16), np.arange(5, dtype=float),
                  (), np.ones((2, 2), bool), 'C1')
    state = StageState(raw, values, methods, raw.allowed, {})
    result = stages.background(state, PreprocessConfig.profile('observed_only'))
    np.testing.assert_array_equal(result.values, state.values)
    np.testing.assert_array_equal(result.method, state.method)
    assert not result.receipts['background']['available']
