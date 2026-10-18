"""Reordered recovery regressions with combined disturbances and original evidence."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from nanobio_core.preprocessing import (
    RawWell, PreprocessConfig, preprocess_well, preprocess_measurement, Cache,
    OBSERVED, TEMPORAL, SPATIAL, STITCHED, CURATED, MISSING, EXCLUDED,
)
from nanobio_core.preprocessing import pipeline, v2
from nanobio_core.preprocessing.model import StageState


def raw_well():
    time = np.arange(180, dtype=float)
    time[90:] += 100
    values = np.broadcast_to(800 + np.arange(180)[:, None, None] * 0.0001, (180, 20, 20)).copy()
    values += np.indices((20, 20))[0][None] * 0.000001 * np.arange(180)[:, None, None]
    allowed = np.ones((20, 20), bool)
    allowed[4, 4] = False
    return RawWell(values, np.ones(values.shape, np.uint16), time, (90,), allowed, 'C1')


def config():
    settings = PreprocessConfig.profile('default_v2')
    return replace(settings, registration=replace(settings.registration, enabled=False),
                   background=replace(settings.background, method='none'),
                   phase_offset=replace(settings.phase_offset, mode='none'))


def recovery(raw, settings=None, entries=()):
    settings = settings or config()
    state = v2.register(v2.sample_mask(raw, settings), settings)
    return v2.recover(v2.well_events(state, settings, entries), settings)


def entry(start, end, **kwargs):
    return dict(well='C1', start_frame=start, end_frame=end, reviewer='test',
                date='2026-10-14', reason='synthetic event', **kwargs)


def test_combined_ten_frame_pulse_and_neighboring_real_step():
    raw = raw_well()
    clean = raw.values_nm.copy()
    raw.values_nm[35:45, 9, 9] += 0.8
    for y, x in [(8, 8), (8, 9), (9, 8)]:
        raw.values_nm[35:90, y, x] += 0.3
    result = recovery(raw)
    assert (result.method[35:45, 9, 9] == TEMPORAL).all()
    np.testing.assert_allclose(result.values[35:45, 9, 9], clean[35:45, 9, 9], atol=1e-10)
    assert (result.method[35:90, 8, 8] == STITCHED).all()
    np.testing.assert_allclose(result.values[35:90, 8, 8], clean[35:90, 8, 8], atol=1e-10)


@pytest.mark.parametrize('length', [14, 25, 40])
def test_returning_pulses_never_become_steps(length):
    raw = raw_well()
    for y, x in [(8, 8), (8, 9), (9, 8)]:
        raw.values_nm[25:25 + length, y, x] += 0.8
    result = recovery(raw)
    assert not (result.method[:, 8, 8] == STITCHED).any()
    method = TEMPORAL if length <= 15 else MISSING
    assert (result.method[25:25 + length, 8, 8] == method).all()
    np.testing.assert_array_equal(result.values[25 + length:90, 8, 8], raw.values_nm[25 + length:90, 8, 8])


def test_phase_edge_pulse_cannot_use_other_phase_anchors():
    raw = raw_well()
    raw.values_nm[84:90] += 0.8
    result = recovery(raw)
    assert (result.method[84:90, 8, 8] == MISSING).all()
    assert result.receipts['well_events']['events'][0]['counts']['no_anchor'] > 0
    np.testing.assert_array_equal(result.values[90:, 8, 8], raw.values_nm[90:, 8, 8])


@pytest.mark.parametrize('start,end', [(70, 90), (0, 20), (84, 90)])
def test_pixel_phase_edge_pulse_is_unresolved(start, end):
    raw = raw_well()
    raw.values_nm[start:end, 8, 8] += 0.8
    result = recovery(raw)
    assert (result.method[start:end, 8, 8] == MISSING).all()
    assert not (result.method[:, 8, 8] == STITCHED).any()


def test_step_labelling_uses_original_partial_windows_before_gap_fixes():
    raw = raw_well()
    for y, x in [(8, 8), (8, 9), (9, 8)]:
        raw.values_nm[35:90, y, x] += 0.3
        raw.codes[38, y, x] = 0
    result = recovery(raw)
    assert (result.method[35:90, 8, 8] == STITCHED).all()
    np.testing.assert_allclose(result.values[35:90, 8, 8], raw.values_nm[35:90, 8, 8] - 0.3, atol=1e-10)


def test_step_after_pulse_fits_remaining_original_anchors():
    raw = raw_well()
    clean = raw.values_nm.copy()
    for y, x in [(8, 8), (8, 9), (9, 8)]:
        raw.values_nm[35:45, y, x] += 0.8
        raw.values_nm[50:90, y, x] += 0.3
    result = recovery(raw)
    assert (result.method[35:45, 8, 8] == TEMPORAL).all()
    assert (result.method[50:90, 8, 8] == STITCHED).all()
    np.testing.assert_allclose(result.values[50:90, 8, 8], clean[50:90, 8, 8], atol=1e-10)


def test_multiple_steps_translate_later_repairs_and_mark_them_stitched():
    raw = raw_well()
    clean = raw.values_nm.copy()
    for y, x in [(8, 8), (8, 9), (9, 8)]:
        raw.values_nm[25:90, y, x] += 0.3
        raw.values_nm[50:90, y, x] += 0.4
    raw.values_nm[70:76] += 0.8
    result = recovery(raw)
    assert (result.method[25:90, 8, 8] == STITCHED).all()
    np.testing.assert_allclose(result.values[25:90, 8, 8], clean[25:90, 8, 8], atol=1e-10)
    assert (result.method[70:76, 10, 10] == CURATED).all()


def test_regional_biological_step_is_preserved():
    raw = raw_well()
    raw.values_nm[35:90, 5:13, 5:15] += 0.3  # 80 / 255 allowed pixels.
    result = recovery(raw)
    assert not result.receipts['well_events']['events']
    assert (result.method[35:90, 8, 8] == OBSERVED).all()
    np.testing.assert_array_equal(result.values[35:90, 8, 8], raw.values_nm[35:90, 8, 8])


@pytest.mark.parametrize('curated', [False, True])
def test_well_event_replaces_two_readout_shapes_with_anchor_line(curated):
    raw = raw_well()
    clean = raw.values_nm.copy()
    raw.values_nm[35:41, :, :10] += np.array([0.8, 0.5, 0.4, 0.7, 0.6, 0.8])[:, None, None]
    raw.values_nm[35:41, :, 10:] += np.array([0.2, 0.8, 0.7, 0.3, 0.8, 0.4])[:, None, None]
    entries = [entry(35, 41)] if curated else []
    result = recovery(raw, entries=entries)
    allowed = result.allowed
    assert (result.method[35:41, allowed] == CURATED).all()
    np.testing.assert_allclose(result.values[35:41, allowed], clean[35:41, allowed], atol=1e-10)
    receipt = result.receipts['well_events']['events'][0]
    assert receipt['chosen_rank'] == 2
    assert receipt['source'] == ('curated' if curated else 'detected')
    assert receipt['max_residual_after_pm'] == 0


def test_long_well_event_is_detected_without_median_self_masking():
    raw = raw_well()
    clean = raw.values_nm.copy()
    raw.values_nm[35:60] += 0.8
    result = recovery(raw)
    assert (result.method[35:60, result.allowed] == CURATED).all()
    np.testing.assert_allclose(result.values[35:60, result.allowed], clean[35:60, result.allowed], atol=1e-10)


@pytest.mark.parametrize('bounds,expected', [((36, 40), [35, 41]), ((33, 43), [33, 43])])
def test_curation_unions_with_detection_without_shrinking(bounds, expected):
    raw = raw_well()
    raw.values_nm[35:41] += 0.8
    result = recovery(raw, entries=[entry(*bounds)])
    event = result.receipts['well_events']['events'][0]
    assert event['detected_bounds'] == [[35, 41]]
    assert event['final_bounds'] == expected


def test_well_median_extends_weak_edges_with_a_two_frame_cap():
    raw = raw_well()
    raw.values_nm[32:44, :, :12] += 0.1
    raw.values_nm[35:41] += 0.8
    settings = config()
    settings = replace(settings, v2=replace(settings.v2, event_fraction=0.9))
    result = recovery(raw, settings)
    event = result.receipts['well_events']['events'][0]
    assert event['detected_bounds'] == [[35, 41]]
    assert event['final_bounds'] == [33, 43]


def test_measurement_consensus_drops_one_outlier_and_preserves_exclusions(monkeypatch, tmp_path):
    raws = {}
    for name, bounds in [('A1', (35, 41)), ('A2', (34, 41)), ('A3', (35, 42)), ('A4', (25, 50))]:
        raw = replace(raw_well(), well=name)
        raw.values_nm[slice(*bounds)] += 0.8
        raws[name] = raw
    monkeypatch.setattr(pipeline, 'load_raw_measurement', lambda path: (raws, {}))
    results = preprocess_measurement('unused', config(), cache=tmp_path)
    for name, result in results.items():
        event = result.receipts['well_events']['events'][0]
        assert event['final_bounds'] == [34, 42]
        assert event['consensus'][0]['outlier_wells'] == ['A4']
        assert event['consensus'][0]['per_well_bounds']['A4']['detected_bounds'] == [[25, 50]]
    assert results['A1'].provenance != preprocess_well(raws['A1'], config()).provenance
    exclusion = dict(entry(37, 39, action='exclude'), well='A1')
    excluded = preprocess_measurement('unused', config(), curation=[exclusion])['A1']
    assert not (excluded.method[37:39, excluded.method[0] != EXCLUDED] == CURATED).any()


def test_consensus_uses_all_measurement_wells_even_for_one_requested_output(monkeypatch):
    raws = {name: replace(raw_well(), well=name) for name in ['A1', 'A2', 'A3', 'A4']}
    for name in ['A1', 'A2']:
        raws[name].values_nm[35:41] += 0.8
    monkeypatch.setattr(pipeline, 'load_raw_measurement', lambda path: (raws, {}))
    result = preprocess_measurement('unused', config(), wells=['A3'])['A3']
    event = result.receipts['well_events']['events'][0]
    assert event['final_bounds'] == [35, 41]
    assert event['detected_bounds'] == []
    assert event['source'] == 'consensus'
    assert event['consensus'][0]['supporting_wells'] == ['A1', 'A2']
    # Raising the quorum prevents a minority from creating an automatic event.
    settings = config()
    settings = replace(settings, v2=replace(settings.v2, consensus_fraction=0.75))
    assert not preprocess_measurement('unused', settings, wells=['A3'])['A3'].receipts['well_events']['events']


def test_consensus_cannot_merge_distinct_phases():
    proposals = {'A1': [dict(phase=[0, 90], bounds=[35, 41], detected_bounds=[[35, 41]])],
                 'A2': [dict(phase=[10, 90], bounds=[35, 41], detected_bounds=[[35, 41]])]}
    settings = config()
    settings = replace(settings, v2=replace(settings.v2, consensus_fraction=1))
    assert v2.plate_consensus(proposals, settings) == {'A1': [], 'A2': []}


@pytest.mark.parametrize('case', ['supported', 'insufficient', 'unexplained', 'ramp'])
def test_partial_event_requires_identifiable_basis_and_quiet_anchors(case):
    raw = raw_well()
    clean = raw.values_nm.copy()
    raw.values_nm[35:41] += np.array([0.8, 0.5, 0.4, 0.7, 0.6, 0.8])[:, None, None]
    raw.codes[36, 8, 8] = 0
    if case == 'insufficient':
        raw.codes[35:40, 8, 8] = 0
    elif case == 'unexplained':
        raw.values_nm[37:39, 8, 8] -= 1.5
    elif case == 'ramp':
        raw.values_nm[:, 8, 8] += np.arange(180) * 0.02
    result = recovery(raw)
    event = result.receipts['well_events']['events'][0]
    if case == 'supported':
        assert (result.method[35:41, 8, 8] == CURATED).all()
        np.testing.assert_allclose(result.values[35:41, 8, 8], clean[35:41, 8, 8], atol=1e-10)
        assert event['outcomes'][8, 8] == 'partial_event'
    else:
        assert (result.method[35:41, 8, 8] == MISSING).all()


def test_terminal_dropout_spanning_event_can_fill_only_outside_event():
    raw = raw_well()
    raw.codes[20:90, 8, 8] = 0
    raw.values_nm[35:41] += 0.8
    result = recovery(raw)
    assert (result.method[20:35, 8, 8] == SPATIAL).all()
    assert (result.method[35:41, 8, 8] == MISSING).all()
    assert (result.method[41:90, 8, 8] == SPATIAL).all()


def test_gap_cannot_interpolate_using_original_event_sample_as_anchor():
    raw = raw_well()
    raw.values_nm[35:41] += 0.8
    raw.codes[41, 8, 8] = 0
    result = recovery(raw)
    assert result.method[41, 8, 8] == MISSING
    assert np.isnan(result.values[41, 8, 8])


def test_dropout_spatial_fill_never_uses_stitched_donors(monkeypatch):
    raw = raw_well()
    raw.codes[60:90, 10, 10] = 0
    for y, x in [(9, 9), (9, 10), (10, 9)]:
        raw.values_nm[35:90, y, x] += 0.3
    seen = []
    original = v2.fill_spatial_frame

    def checked(values, donors, targets, allowed, settings, cache):
        assert not donors[9, 10] and not donors[10, 9]
        seen.append(True)
        return original(values, donors, targets, allowed, settings, cache)

    monkeypatch.setattr(v2, 'fill_spatial_frame', checked)
    result = recovery(raw)
    assert seen and (result.method[60:90, 10, 10] == SPATIAL).all()


def test_contaminated_background_pick_falls_back():
    raw = raw_well()
    raw.values_nm[40:65, 6, 6] += 0.8  # Returning, beyond pulse repair cap.
    settings = config()
    settings = replace(settings, background=replace(settings.background, method='median',
                                                   coordinates=((6, 6),)))
    result = preprocess_well(raw, settings)
    assert result.receipts['background']['fallback_pixels']
    assert np.max(np.abs(result.values_pm[40:65, 8, 8])) < 1


@pytest.mark.parametrize('method', [MISSING, STITCHED, SPATIAL])
def test_unanchored_phase_edge_is_flagged_without_skipping_bad_frames(method):
    raw = raw_well()
    settings = config()
    settings = replace(settings, phase_offset=replace(settings.phase_offset, mode='level'))
    values = raw.values_nm.copy()
    values[90:] += 10
    methods = np.full(values.shape, OBSERVED, np.uint8)
    methods[89, 8, 8] = method
    state = StageState(raw, values, methods, raw.allowed, {})
    result = v2.phase_offset(state, settings)
    assert result.receipts['phase_offset']['maps'][0]['offset_unanchored'][8, 8]
    np.testing.assert_array_equal(result.values[90:, 8, 8], values[90:, 8, 8])
    assert not result.receipts['phase_offset']['maps'][0]['offset_unanchored'][9, 9]


def test_curation_exclusion_wins_over_detected_and_curated_events():
    raw = raw_well()
    raw.values_nm[35:41] += 0.8
    entries = [entry(35, 41), entry(35, 41, action='exclude')]
    result = recovery(raw, entries=entries)
    assert not result.receipts['well_events']['events']
    np.testing.assert_array_equal(result.values[35:41], raw.values_nm[35:41])
    assert (result.method[35:41, result.allowed] == OBSERVED).all()


def test_no_anchor_and_unexplained_event_remain_missing():
    raw = raw_well()
    raw.values_nm[35:41] += np.array([0.8, 0.5, 0.4, 0.7, 0.6, 0.8])[:, None, None]
    raw.values_nm[35:41, 8, 8] += np.array([-0.8, 1, -1, 0, 1, -1])
    raw.codes[32, 9, 9] = 0
    result = recovery(raw)
    receipt = result.receipts['well_events']['events'][0]
    assert receipt['outcomes'][8, 8] == 'unexplained_event'
    assert receipt['outcomes'][9, 9] == 'no_anchor'
    assert (result.method[35:41, 8, 8] == MISSING).all()
    assert (result.method[35:41, 9, 9] == MISSING).all()


def test_determinism_cache_workers_and_excluded_bytes(tmp_path, monkeypatch):
    raw = raw_well()
    raw.values_nm[35:41] += 0.8
    settings = config()
    raws = {name: replace(raw, well=name) for name in ['C1', 'A1']}
    monkeypatch.setattr(pipeline, 'load_raw_measurement', lambda path: (raws, {}))
    serial = preprocess_measurement('unused', settings, workers=1)
    parallel = preprocess_measurement('unused', settings, workers=2)
    repeated = preprocess_well(raw, settings)
    cache = Cache(tmp_path)
    cache.put(repeated)
    cached = cache.get(repeated.provenance)
    for other in [parallel['C1'], serial['C1'], cached]:
        for field in ['values_pm', 'method', 'coverage']:
            assert getattr(other, field).tobytes() == getattr(repeated, field).tobytes()
    for other, expected in [(parallel['C1'], serial['C1']), (cached, repeated)]:
        assert other.provenance == expected.provenance
        assert json.dumps(other.receipts, default=pipeline._receipt_array, sort_keys=True) == json.dumps(
            expected.receipts, default=pipeline._receipt_array, sort_keys=True)
    excluded = repeated.method[0] == EXCLUDED
    assert repeated.values_pm[:, excluded].tobytes() == raw.values_nm[:, excluded].tobytes()
    assert PreprocessConfig.from_dict(settings.to_dict()) == settings


def test_registration_moves_mask_codes_and_values_together(monkeypatch):
    raw = raw_well()
    raw.codes[100, 7, 7] = 0
    settings = config()
    settings = replace(settings, registration=replace(settings.registration, enabled=True))
    monkeypatch.setattr(v2, '_phase_translation', lambda *args: np.array([1, 0]))
    masked = v2.sample_mask(raw, settings)
    registered = v2.register(masked, settings)
    assert registered.acquired_codes[100, 7, 8] == 0
    assert registered.method[100, 7, 8] == MISSING
    assert np.isnan(registered.values[100, 7, 8])
    assert registered.values[110, 7, 8] == masked.values[110, 7, 7]
    assert registered.receipts['interphase_registration']['projection'] == 'median'
    assert registered.values[:, ~masked.allowed].tobytes() == raw.values_nm[:, ~masked.allowed].tobytes()


def test_sample_mask_keeps_jump_evidence_and_constant_observations():
    raw = raw_well()
    raw.values_nm[35:41, 8, 8] += 2
    raw.values_nm[:, 9, 9] = 800
    state = v2.sample_mask(raw, config())
    assert (state.method[35:41, 8, 8] == OBSERVED).all()
    assert state.receipts['sample_mask']['suspect'][35, 8, 8]
    assert (state.method[:, 9, 9] == OBSERVED).all()


def test_stage_three_to_five_phase_isolation():
    raw = raw_well()
    raw.values_nm[35:41] += 0.8
    raw.codes[70:90, 8, 8] = 0
    combined = recovery(raw)
    for left, right in [(0, 90), (90, 180)]:
        phase = RawWell(raw.values_nm[left:right], raw.codes[left:right], raw.time_s[left:right],
                        (), raw.allowed, raw.well)
        independent = recovery(phase)
        assert combined.values[left:right].tobytes() == independent.values.tobytes()
        assert combined.method[left:right].tobytes() == independent.method.tobytes()


@pytest.mark.parametrize('profile', ['default', 'legacy', 'observed_only'])
def test_existing_profiles_keep_frozen_bytes(profile):
    fixture = Path(__file__).parent / 'fixtures/preprocessing-performance.npz'
    with np.load(fixture, allow_pickle=False) as bundle:
        raw = RawWell(bundle['values_nm'], bundle['codes'], bundle['time_s'],
                      tuple(map(int, bundle['phases'])), bundle['allowed'], 'A1')
        settings = PreprocessConfig.profile(profile)
        settings = replace(settings, registration=replace(settings.registration, enabled=False),
                           background=replace(settings.background, coordinates=((3, 3), (4, 3))))
        result = preprocess_well(raw, settings)
        for field in ['values_pm', 'method', 'coverage']:
            assert getattr(result, field).tobytes() == bundle[f'{profile}_{field}'].tobytes()
