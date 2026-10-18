"""Trimmed low-rank repair of distinct instrument readout pulse shapes."""

from dataclasses import replace
import json

import numpy as np
import pytest

from nanobio_core.preprocessing import Cache, PreprocessConfig, RawWell, preprocess_well
from nanobio_core.preprocessing import cli, stages
from nanobio_core.preprocessing.curation import load_curation, pulse_basis_similarities
from nanobio_core.preprocessing.model import CURATED, EXCLUDED, MISSING


def interval(**overrides):
    entry = dict(well='C1', start_frame=12, end_frame=18, reviewer='Reviewer', date='2026-10-14',
                 reason='Reviewed segmented instrument pulse')
    entry.update(overrides)
    return entry


def segmented_pulse(noise_pm=0.5, special=True):
    rng = np.random.default_rng(31)
    time = np.cumsum(rng.uniform(0.8, 1.2, 36))
    slopes = np.linspace(0.0001, 0.002, 400).reshape(20, 20)
    trend = 800 + time[:, None, None] * slopes
    values = trend + rng.normal(0, noise_pm / 1000, trend.shape)
    shapes = np.array([[0.2, 1, 0.4, 0, 0, 0], [0, 0, 0, 0.3, 1, 0.2]])
    scales = np.linspace(0.2, 1.6, 400).reshape(20, 20)
    scales[10, 10] = 0.8
    artifact = np.broadcast_to(shapes[0, :, None, None], (6, 20, 20)).copy()
    artifact[:, :, 15:] = shapes[1, :, None, None]
    values[12:18] += artifact * scales
    codes = np.ones(values.shape, np.uint16)
    orthogonal = np.array([1, -0.2, 0, 0, 0, 0])
    if special:
        values[12:18, 0, 0] = trend[12:18, 0, 0] + orthogonal * 0.03
        values[12:18, 0, 1] = trend[12:18, 0, 1] + orthogonal * 0.3
        values[12:18, 0, 2] += orthogonal * 0.05
        values[12:18, 0, 3] = trend[12:18, 0, 3] + orthogonal * 20
        codes[6, 1, 0] = 0
        codes[14, 1, 1] = 0
    allowed = np.ones((20, 20), bool)
    allowed[-1, -1] = False
    return RawWell(values, codes, time, (28,), allowed, 'C1'), trend, shapes, orthogonal


@pytest.mark.parametrize('profile', ['default', 'observed_only'])
def test_low_rank_repairs_both_segments_and_retains_unmodelled_signal(profile):
    raw, trend, _, orthogonal = segmented_pulse()
    config = PreprocessConfig.profile(profile)
    state = stages.validity(raw, config, [interval()])
    state = stages.within_phase_recovery(state, config)
    state.values[:, raw.allowed] += 0.1  # earlier stitching must survive
    result = stages.curated_repairs(state, config, [interval()])
    receipt = result.receipts['curated_repairs'][0]
    assert receipt['chosen_rank'] == 2
    assert receipt['basis_trimmed_pixels'] == 4
    assert not receipt['basis_fit_mask'][0, 3]  # 20,000 pm genuine outlier is trimmed
    assert receipt['coefficient_maps'].shape == (2, 20, 20)
    mask = np.all(result.method[12:18] == CURATED, axis=0)
    ordinary = mask.copy()
    ordinary[0, :4] = False
    assert ordinary[:, :15].any() and ordinary[:, 15:].any()
    assert mask[10, 10]  # 800 pm
    assert mask[19, 18]  # greater than the automatic jump threshold
    assert np.max(np.abs(result.values[12:18, ordinary] - trend[12:18, ordinary] - 0.1)) * 1000 < 3
    assert receipt['outcomes'][0, 0] == 'near_noise'
    np.testing.assert_allclose(result.values[12:18, 0, 0] - trend[12:18, 0, 0] - 0.1,
                               orthogonal * 0.03, atol=0.002)
    np.testing.assert_allclose(result.values[12:18, 0, 2] - trend[12:18, 0, 2] - 0.1,
                               orthogonal * 0.05, atol=0.002)
    for y, x, outcome in [(0, 1, 'unexplained_pulse'), (0, 3, 'unexplained_pulse'),
                          (1, 0, 'no_anchor'), (1, 1, 'unexplained_pulse')]:
        assert receipt['outcomes'][y, x] == outcome
        assert np.all(result.method[12:18, y, x] == MISSING)
        assert np.isnan(result.values[12:18, y, x]).all()
    assert sum(receipt['counts'].values()) == raw.allowed.sum()
    assert receipt['repaired_pixels'] == mask.sum()
    assert 45 < receipt['max_residual_after_pm'] < 55  # residual is kept even above anchor noise
    np.testing.assert_array_equal(result.values[:12], state.values[:12])
    np.testing.assert_array_equal(result.values[18:], state.values[18:])
    assert result.values[:, ~raw.allowed].tobytes() == raw.values_nm[:, ~raw.allowed].tobytes()
    assert np.all(result.method[:, ~raw.allowed] == EXCLUDED)
    common = stages.curated_repairs(state, config, [interval(method='common_mode')])
    assert np.all(common.method[12:18, 10, 18] == MISSING)
    assert np.all(result.method[12:18, 10, 18] == CURATED)


@pytest.mark.parametrize('max_rank,basis_variance,rank', [(3, 0.95, 2), (1, 0.95, 1), (3, 0.5, 1)])
def test_rank_selection_and_explained_variance(max_rank, basis_variance, rank):
    raw, _, _, _ = segmented_pulse(noise_pm=0, special=False)
    config = PreprocessConfig()
    entry = interval(max_rank=max_rank, basis_variance=basis_variance)
    result = stages.curated_repairs(stages.validity(raw, config, [entry]), config, [entry])
    receipt = result.receipts['curated_repairs'][0]
    assert receipt['chosen_rank'] == rank
    variance = receipt['explained_variance']
    assert variance.sum() == pytest.approx(1)
    if max_rank > 1:
        assert variance[:rank].sum() >= basis_variance
        assert variance[:rank - 1].sum() < basis_variance
    basis = receipt['basis_vectors']
    np.testing.assert_allclose(basis @ basis.T, np.eye(rank), atol=1e-12)
    if rank == 2:
        assert receipt['max_residual_after_pm'] < 1e-6


def test_low_rank_no_anchors_phase_guard_and_zero_energy():
    raw, _, _, _ = segmented_pulse(noise_pm=0, special=False)
    config = PreprocessConfig()
    state = stages.validity(raw, config, [interval()])
    with pytest.raises(ValueError, match='phase'):
        stages.curated_repairs(replace(state, raw=replace(raw, phases=(20,))), config, [interval()])
    with pytest.raises(ValueError, match='clean windows'):
        stages.curated_repairs(state, config, [interval(clean_frames=2)])
    state.values[6:12, raw.allowed] = np.nan
    result = stages.curated_repairs(state, config, [interval()])
    receipt = result.receipts['curated_repairs'][0]
    assert receipt['counts']['no_anchor'] == raw.allowed.sum()
    assert receipt['chosen_rank'] == 0
    assert receipt['max_residual_after_pm'] is None
    assert receipt['coefficient_maps'].shape == (0, 20, 20)
    # A constant acquisition has exactly zero residual and a valid rank-zero repair.
    raw = replace(raw, values_nm=np.zeros_like(raw.values_nm))
    config = replace(config, validity=replace(config.validity, flat_trace_is_missing=False))
    result = stages.curated_repairs(stages.validity(raw, config, [interval()]), config, [interval()])
    receipt = result.receipts['curated_repairs'][0]
    assert receipt['chosen_rank'] == 0
    assert receipt['max_residual_after_pm'] == 0
    assert np.all(receipt['explained_fraction_map'][raw.allowed] == 1)
    assert np.all(result.method[12:18, raw.allowed] == CURATED)


def test_low_rank_yaml_defaults_overrides_and_config(tmp_path):
    path = tmp_path / 'curation.yml'
    entry = interval(method='low_rank', max_rank=2, min_explained=0.9, basis_variance=0.99)
    lines = '\n    '.join(f'{key}: {value}' for key, value in entry.items())
    path.write_text('entries:\n  - ' + lines + '\n')
    assert load_curation(path) == [entry]
    config = PreprocessConfig()
    assert config.curation.method == 'low_rank'
    defaults = (config.curation.max_rank, config.curation.basis_variance, config.curation.min_explained)
    assert defaults == (3, 0.95, 0.8)
    changed = replace(config, curation=replace(config.curation, max_rank=2, min_explained=0.9))
    assert PreprocessConfig.from_dict(changed.to_dict()) == changed
    assert changed.config_hash() != config.config_hash()
    raw, _, _, _ = segmented_pulse()
    result = stages.curated_repairs(stages.validity(raw, config, [entry]), config, path)
    for key in ('method', 'max_rank', 'min_explained', 'basis_variance'):
        assert result.receipts['curated_repairs'][0]['parameters'][key] == entry[key]


@pytest.mark.parametrize('settings', [{'max_rank': 0}, {'max_rank': 1.5}, {'max_rank': True},
                                     {'basis_variance': 0}, {'basis_variance': 1.1},
                                     {'basis_variance': float('nan')}, {'min_explained': -0.1},
                                     {'min_explained': 1.1}, {'min_explained': float('nan')}])
def test_invalid_low_rank_settings(settings):
    with pytest.raises(ValueError):
        load_curation([interval(**settings)])
    with pytest.raises(ValueError):
        replace(PreprocessConfig().curation, **settings)


def test_cross_well_principal_angles():
    basis = np.eye(6)[:2]
    rotated = np.array([[1, 1], [1, -1]]) @ basis
    result = pulse_basis_similarities({'A1': basis, 'A2': rotated, 'A3': np.eye(6)[2:4],
                                      'A4': basis[:1], 'B1': np.empty((0, 6)),
                                      'B2': np.full((1, 6), np.nan)})
    assert result['A1']['A2']['similarity'] == pytest.approx(1)
    np.testing.assert_allclose(result['A1']['A2']['principal_angles_rad'], 0, atol=3e-8)
    assert result['A1']['A3']['similarity'] == 0
    assert result['A1']['A4']['similarity'] == pytest.approx(0.5)
    assert result['A1']['B1'] is None and result['A1']['B2'] is None
    with pytest.raises(ValueError, match='matching'):
        pulse_basis_similarities({'A1': basis, 'A2': basis[:, :3]})
    with pytest.raises(ValueError, match='two-dimensional'):
        pulse_basis_similarities({'A1': basis[0]})


def test_low_rank_pipeline_cache_and_cli_receipts(tmp_path, monkeypatch):
    raw, _, _, _ = segmented_pulse()
    config = PreprocessConfig.profile('observed_only')
    config = replace(config, background=replace(config.background, method='none'),
                     registration=replace(config.registration, enabled=False))
    monkeypatch.setattr(stages, 'detect_active_area', lambda frame, margin: (None, raw.allowed, 0))
    result = preprocess_well(raw, config, [interval()])
    cache = Cache(tmp_path / 'cache')
    cache.put(result)
    saved = cache.get(result.provenance)
    np.testing.assert_array_equal(saved.values_pm, result.values_pm)
    receipt = result.receipts['curated_repairs'][0]
    for key, value in receipt.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(saved.receipts['curated_repairs'][0][key], value)
    monkeypatch.setattr(cli, 'preprocess_measurement', lambda *args, **kwargs: {'C1': result})
    cli.main(['review', str(tmp_path / 'measurement'), '--out', str(tmp_path / 'review')])
    report = json.loads((tmp_path / 'review' / 'review-receipts.json').read_text())
    assert np.shape(report['C1']['receipts']['curated_repairs'][0]['basis_vectors']) == (2, 6)
