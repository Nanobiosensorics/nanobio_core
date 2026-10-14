from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest

from nanobio_core.preprocessing import (
    PreprocessConfig, RawWell, Cache, preprocess_well, preprocess_measurement, weighted_region_sum,
    OBSERVED, TEMPORAL, SPATIAL, CURATED, MISSING, EXCLUDED,
)
from nanobio_core.preprocessing import stages, pipeline
from nanobio_core.preprocessing.curation import load_curation, curation_hash
from nanobio_core.epic_cardio import data_correction as correction


def raw_well():
    time = np.arange(60, dtype=float)
    values = np.broadcast_to(800 + time[:, None, None] * 0.001, (60, 20, 20)).copy()
    values += np.indices((20, 20))[0][None] * 0.0001 * time[:, None, None]
    return RawWell(values, np.ones(values.shape, np.uint16), time, (30,), np.ones((20, 20), bool), 'A1')


def config_without_background():
    config = PreprocessConfig()
    return replace(config, background=replace(config.background, method='none'),
                   registration=replace(config.registration, enabled=False))


def test_config_roundtrip_hash_profiles_and_frozen():
    config = PreprocessConfig()
    assert PreprocessConfig.from_dict(config.to_dict()) == config
    restored = PreprocessConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored.config_hash() == config.config_hash()
    assert not PreprocessConfig.profile('observed_only').recovery.enabled
    assert PreprocessConfig.profile('observed-only') == PreprocessConfig.profile('observed_only')
    assert PreprocessConfig.profile('legacy').phase_offset.mode == 'single'
    with pytest.raises(AttributeError):
        config.version = 2
    with pytest.raises(ValueError):
        PreprocessConfig.profile('unknown')


def test_default_nan_signed_values_coverage_determinism_and_excluded_bytes():
    raw = raw_well()
    raw.codes[:, 8, 8] = 0
    raw.values_nm[:, 7, 7] -= np.arange(60) * 0.003
    config = config_without_background()
    a = preprocess_well(raw, config)
    b = preprocess_well(raw, config)
    np.testing.assert_array_equal(a.values_pm, b.values_pm)
    np.testing.assert_array_equal(a.method, b.method)
    assert np.isnan(a.values_pm[:, 8, 8]).all()
    assert np.nanmin(a.values_pm[:, 7, 7]) < 0
    assert a.coverage[8, 8] == 0
    excluded = np.all(a.method == EXCLUDED, axis=0)
    assert a.values_pm[:, excluded].tobytes() == raw.values_nm[:, excluded].tobytes()
    assert len(a.receipts) == 9
    assert a.provenance == b.provenance


def legacy_expected(raw, config):
    settings = config.background
    np.random.seed(config.legacy.random_seed)
    aligned = correction.correct_interphase_well_shifts(
        raw.values_nm, raw.phases, settings.threshold_pm, list(settings.coordinates), settings.method)
    values, _, _ = correction.correct_well(aligned, settings.threshold_pm,
                                          list(settings.coordinates), settings.method)
    values, _ = correction.interpolate_frame_jumps(values, phases=raw.phases)
    return correction.correct_phase_offsets(values, raw.phases)[0]


@pytest.mark.parametrize('manual', [True, False])
def test_legacy_synthetic_equivalence_and_rng_isolation(manual):
    raw = raw_well()
    config = PreprocessConfig.profile('legacy')
    if manual:
        config = replace(config, background=replace(config.background, coordinates=((5, 5), (6, 6))))
    expected = legacy_expected(raw, config)
    saved = np.random.get_state()
    actual = preprocess_well(raw, config)
    np.testing.assert_allclose(actual.values_pm, expected, rtol=0, atol=1e-9)
    after = np.random.get_state()
    np.testing.assert_array_equal(saved[1], after[1])
    assert saved[2:] == after[2:]


def fixture_path():
    return Path(__file__).resolve().parents[2] / 'test/data/20210526_MCF7_fn/epic_data'


def test_legacy_fixture_equivalence():
    path = fixture_path()
    if not (path / 'DMR').exists():
        pytest.skip('Raw fixture not installed')
    raw, _ = pipeline.load_raw_measurement(path)
    config = PreprocessConfig.profile('legacy')
    config = replace(config, background=replace(config.background, coordinates=((20, 20), (60, 60))))
    for name in ('A1', 'C1'):
        np.testing.assert_allclose(preprocess_well(raw[name], config).values_pm,
                                   legacy_expected(raw[name], config), rtol=0, atol=1e-9)


def test_real_fixture_holdout_reports_temporal_and_spatial_error():
    path = fixture_path()
    if not (path / 'DMR').exists():
        pytest.skip('Raw fixture not installed')
    raw, _ = pipeline.load_raw_measurement(path)
    raw = raw['A1']
    from nanobio_core.preprocessing.recovery import recover_pixels
    valid = (raw.codes != 0) & np.isfinite(raw.values_nm)
    # Select a genuinely intact interior trace with compatible temporal brackets.
    compatible = valid.all(axis=0) & raw.allowed
    compatible &= np.abs(raw.values_nm[15] - raw.values_nm[11]) * 1000 < 75
    points = np.argwhere(compatible)
    assert len(points), 'Fixture has no intact holdout candidates'
    y, x = points[len(points) // 2]
    masked = valid.copy()
    masked[12:15, y, x] = False
    masked[-15:, y, x] = False
    recovered = recover_pixels(raw.values_nm, masked, raw.time_s, raw.phases, raw.allowed)
    temporal_error = np.abs(recovered.values[12:15, y, x] - raw.values_nm[12:15, y, x]) * 1000
    spatial = recovered.methods[-15:, y, x] == SPATIAL
    assert spatial.any(), 'Selected holdout must have spatial support'
    spatial_error = np.abs(recovered.values[-15:, y, x][spatial] - raw.values_nm[-15:, y, x][spatial]) * 1000
    print(dict(temporal_median_error_pm=float(np.median(temporal_error)),
               spatial_median_error_pm=float(np.median(spatial_error))))
    assert np.isfinite(temporal_error).all() and np.isfinite(spatial_error).all()
    assert np.median(temporal_error) < 75
    assert np.median(spatial_error) < 500


def test_robust_phase_offset_known_offset_and_noise():
    raw = raw_well()
    time = raw.time_s.copy() * 9
    time[30:] += 100
    rng = np.random.default_rng(7)
    trace = 800 + time[:, None, None] * 0.002
    values = np.broadcast_to(trace, raw.values_nm.shape).copy()
    values[30:] += 0.4
    values += rng.normal(0, 0.0005, values.shape)
    raw = replace(raw, values_nm=values, time_s=time)
    config = config_without_background()
    state = stages.validity(raw, config)
    state = stages.reference(state, config)
    corrected = stages.phase_offset(state, config)
    offsets = np.asarray(corrected.receipts['phase_offset']['maps'][0]['offset_pm'])
    assert abs(np.nanmedian(offsets) + 400) < 2
    np.testing.assert_array_equal(corrected.values[:30], state.values[:30])
    assert np.nanmedian(np.abs(offsets + 400)) < 2


def test_phase_offset_modes_and_insufficient_anchors():
    raw = raw_well()
    config = config_without_background()
    state = stages.reference(stages.validity(raw, config), config)
    none = replace(config, phase_offset=replace(config.phase_offset, mode='none'))
    np.testing.assert_array_equal(stages.phase_offset(state, none).values, state.values)
    state.method[30:, 5, 5] = MISSING
    corrected = stages.phase_offset(state, config)
    np.testing.assert_array_equal(corrected.values[:, 5, 5], state.values[:, 5, 5])
    single = replace(config, phase_offset=replace(config.phase_offset, mode='single'))
    actual = stages.phase_offset(state, single).values
    np.testing.assert_allclose(actual[30], state.values[29])


def entry():
    return dict(well='A1', start_frame=12, end_frame=18, reviewer='Test reviewer',
                date='2026-10-14', reason='Confirmed returning acquisition pulse')


@pytest.mark.parametrize('pulse_length', [3, 6])
def test_curation_loading_repair_phase_guard_and_byte_identity(tmp_path, pulse_length):
    interval = {**entry(), 'end_frame': 12 + pulse_length}
    path = tmp_path / 'curation.yml'
    path.write_text('entries:\n  - ' + '\n    '.join(f'{k}: {v}' for k, v in interval.items()) + '\n')
    assert load_curation(path) == [interval]
    raw = raw_well()
    end = interval['end_frame']
    raw.values_nm[12:end, 5:8, 5:8] += 0.4
    config = config_without_background()
    state = stages.active_area(stages.validity(raw, config), config)
    result = stages.curated_repairs(state, config, path)
    assert np.all(result.method[12:end, 6, 6] == CURATED)
    recovered = stages.within_phase_recovery(state, config)
    reviewed = stages.curated_repairs(recovered, config, path)
    assert np.all(reviewed.method[12:end, 6, 6] == CURATED)
    np.testing.assert_array_equal(result.values[30:], state.values[30:])
    assert result.values[:, ~state.allowed].tobytes() == state.values[:, ~state.allowed].tobytes()
    with pytest.raises(ValueError, match='phase'):
        stages.curated_repairs(state, config, [{**entry(), 'start_frame': 27, 'end_frame': 33}])
    with pytest.raises(ValueError, match='missing'):
        load_curation([{'well': 'A1'}])


def test_cache_keys_raw_config_curation_code_and_roundtrip(tmp_path, monkeypatch):
    raw = raw_well()
    config = config_without_background()
    original = pipeline.provenance_for(raw, config)
    changed = replace(config, reference=replace(config.reference, frames=3))
    alternatives = [pipeline.provenance_for(raw, changed),
                    pipeline.provenance_for(raw, config, [entry()])]
    raw.values_nm[10, 5, 5] += 0.01
    alternatives.append(pipeline.provenance_for(raw, config))
    monkeypatch.setattr(pipeline, 'code_version', lambda: 'changed')
    alternatives.append(pipeline.provenance_for(raw, config))
    assert len({pipeline.cache_key(p) for p in [original, *alternatives]}) == 5
    result = preprocess_well(raw, config)
    cache = Cache(tmp_path)
    assert cache.get(result.provenance) is None
    cache.put(result)
    saved = cache.get(result.provenance)
    np.testing.assert_array_equal(saved.values_pm, result.values_pm)
    np.testing.assert_array_equal(saved.method, result.method)
    assert saved.provenance == result.provenance


def test_measurement_one_vs_many_workers_and_cache(tmp_path, monkeypatch):
    raw = raw_well()
    wells = {'A1': raw, 'A2': replace(raw, well='A2')}
    monkeypatch.setattr(pipeline, 'load_raw_measurement', lambda path: (wells, {}))
    config = config_without_background()
    a = preprocess_measurement('synthetic', config, workers=1, cache=tmp_path)
    b = preprocess_measurement('synthetic', config, workers=2)
    for name in a:
        np.testing.assert_array_equal(a[name].values_pm, b[name].values_pm)
        np.testing.assert_array_equal(a[name].method, b[name].method)
        assert a[name].provenance == b[name].provenance
    assert len(list(tmp_path.glob('*.npz'))) == 2


def test_weighted_export_fractional_example_and_missing_support():
    raw = raw_well()
    result = preprocess_well(raw, config_without_background())
    result.values_pm[:, 5, 5] = 100
    result.values_pm[:, 5, 6] = 200
    result.method[:, 5, 5:7] = OBSERVED
    region = [(5, 5, 0.25), (6, 5, 0.75)]
    values, observed, reconstructed = weighted_region_sum(result, region)
    np.testing.assert_array_equal(values * 625, np.full(60, 175 * 625))
    assert np.all(observed == 1) and np.all(reconstructed == 0)
    result.method[0, 5, 6] = SPATIAL
    values, observed, reconstructed = weighted_region_sum(result, region)
    assert np.isnan(values[0]) and observed[0] == 0.25 and reconstructed[0] == 0.75
    result.values_pm[1, 5, 6] = np.nan
    assert np.isnan(weighted_region_sum(result, region)[0][1])


def test_active_area_bad_frame_zero_fails_explicitly_and_trims_never_expand():
    raw = raw_well()
    raw.allowed[:, :5] = False
    config = config_without_background()
    state = stages.active_area(stages.validity(raw, config), config)
    assert not state.allowed[:, :5].any()
    raw.values_nm[0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        stages.active_area(stages.validity(raw, config), config)


def test_validity_is_sample_local_without_dilation_and_ignores_phase_gaps():
    raw = raw_well()
    raw.codes[10, 5, 5] = 0
    raw.values_nm[:, 7, 7] = 800
    raw.values_nm[15, 8, 8] = np.nan
    raw.values_nm[30:] += 10
    config = config_without_background()
    state = stages.validity(raw, config)
    assert state.method[10, 5, 5] == MISSING
    assert state.method[10, 5, 6] == OBSERVED
    assert state.method[9, 5, 5] == OBSERVED
    assert state.method[15, 8, 8] == MISSING
    assert state.method[30, 5, 5] == OBSERVED
    # A separate truly flat pixel is unavailable for the entire acquisition.
    raw.values_nm[:, 7, 7] = 800
    state = stages.validity(raw, config)
    assert np.all(state.method[:, 7, 7] == MISSING)


def test_reference_first_observed_frames_and_no_5000_pm_cutoff():
    raw = raw_well()
    raw.values_nm[:] = 800 + np.arange(60)[:, None, None] * 0.2
    raw.codes[0:2, 5, 5] = 0
    config = config_without_background()
    config = replace(config, recovery=replace(config.recovery, enabled=False),
                     phase_offset=replace(config.phase_offset, mode='none'))
    state = stages.reference(stages.validity(raw, config), config)
    assert abs(state.receipts['reference']['baseline_nm'][5][5] - 800.8) < 1e-10
    result = preprocess_well(raw, config)
    assert np.isnan(result.values_pm[:2, 5, 5]).all()
    assert result.values_pm[-1, 5, 5] > 5000
    assert result.values_pm[2, 5, 5] < 0


def test_nan_aware_background_manual_picks_and_cell_exclusion():
    raw = raw_well()
    config = config_without_background()
    settings = replace(config.background, method='median', coordinates=((5, 5), (6, 6)))
    config = replace(config, background=settings)
    state = stages.reference(stages.validity(raw, config), config)
    state.values[:, 5, 5] = 10
    state.values[:, 6, 6] = 20
    state.values[10, 5, 5] = np.nan
    state.method[10, 5, 5] = MISSING
    result = stages.background(state, config)
    assert result.receipts['background']['trace_pm'][10] == 20
    assert result.receipts['background']['trace_pm'][9] == 15
    assert np.isnan(result.values[10, 5, 5])
    cell_mask = np.zeros(raw.allowed.shape, bool)
    cell_mask[6, 6] = True
    state.raw = replace(raw, cell_mask=cell_mask)
    result = stages.background(state, config)
    assert result.receipts['background']['coordinates'] == [(5, 5)]
    assert np.isnan(result.receipts['background']['trace_pm'][10])


def test_registration_moves_validity_and_preserves_excluded_pixels(monkeypatch):
    raw = raw_well()
    raw.allowed[:, :5] = False
    raw.codes[35, 6, 6] = 0
    config = config_without_background()
    config = replace(config, registration=replace(config.registration, enabled=True))
    state = stages.validity(raw, config)
    monkeypatch.setattr(stages, 'find_local_maxima', lambda *args: ([6], [6]))
    monkeypatch.setattr(stages, '_phase_translation', lambda *args: np.array([1, 0]))
    result = stages.interphase_registration(state, config)
    assert result.method[35, 6, 7] == MISSING
    assert result.method[35, 6, 5] == MISSING
    assert np.isnan(result.values[35, 6, 5])
    assert result.values[:, ~raw.allowed].tobytes() == state.values[:, ~raw.allowed].tobytes()
    np.testing.assert_array_equal(result.method[:30], state.method[:30])
