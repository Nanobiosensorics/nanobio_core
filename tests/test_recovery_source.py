import json

import numpy as np
import pytest

from nanobio_core.epic_cardio.recovery_source import cube_digest, prepare_recovery_source
from nanobio_core.epic_cardio import processing


def source(tmp_path):
    time = np.arange(50) * 12.0
    raw = np.full((50, 12, 12), 830.0)
    recovered = raw.copy()
    recovered[:, 6, 6] += np.arange(50) * 0.004
    allowed = np.ones((12, 12), dtype=bool)
    metadata = dict(phases=[25], registration_threshold=75, registration_method='mean',
                    wells={'A1': {'input_sha256': cube_digest(raw)}})
    path = tmp_path / 'source.npz'
    np.savez(path, A1=recovered, A1_allowed=allowed, time_seconds=time,
             metadata=np.array(json.dumps(metadata)))
    return path, raw, time, allowed


def test_recovery_source_rejects_changed_raw_domain_and_registration(tmp_path):
    path, raw, _, allowed = source(tmp_path)
    drift = dict(threshold=75, filter_method='mean', phase_offset_correction=False)
    with pytest.raises(ValueError, match='raw data'):
        prepare_recovery_source(path, 'A1', raw + 1, drift, [(0, 0)], [25])
    with pytest.raises(ValueError, match='phase boundaries'):
        prepare_recovery_source(path, 'A1', raw, drift, [(0, 0)], [])
    changed = allowed.copy()
    changed[:, 0] = False
    with pytest.raises(ValueError, match='border/active-area'):
        prepare_recovery_source(path, 'A1', raw, drift, [(0, 0)], [25], changed)
    with pytest.raises(ValueError, match='threshold changed'):
        prepare_recovery_source(path, 'A1', raw, {**drift, 'threshold': 76}, [(0, 0)], [25])
    with pytest.raises(ValueError, match='timestamps changed'):
        prepare_recovery_source(path, 'A1', raw, drift, [(0, 0)], [25], current_time=np.arange(50))


def test_both_pipeline_stages_replay_same_recovery_without_legacy_masking(tmp_path, monkeypatch):
    path, raw, time, allowed = source(tmp_path)
    monkeypatch.setattr(processing, 'WELL_NAMES', ['A1'])
    params = dict(pixel_recovery_source=str(path), signal_range={'range_type': 0, 'ranges': [0, None]},
                  drift_correction=dict(threshold=75, filter_method='mean', background_selector=True,
                                        phase_offset_correction=False, frame_jump_correction=True))
    localization = dict(threshold_range=[150, 3000], neighbourhood_size=3, error_mask_filtering=True)
    before = processing.preprocessing(params, {'A1': raw}, time, [25], {'A1': [(0, 0)]})
    after = processing.localization(params, localization, {'A1': raw}, before[2], before[4],
                                     {'A1': [(0, 0)]})
    np.testing.assert_allclose(before[0]['A1'], after['A1'][0])
    assert before[0]['A1'][-1, 6, 6] == pytest.approx(196)


def test_unobserved_phase_endpoint_cannot_become_zero_to_signal_jump(tmp_path):
    path, raw, time, allowed = source(tmp_path)
    with np.load(path, allow_pickle=False) as bundle:
        fields = {key: bundle[key].copy() for key in bundle.files}
    fields['A1'][24, 6, 6] = np.nan
    fields['A1'][:, 7, 7] += np.arange(50) * .002
    np.savez(path, **fields)
    drift = dict(threshold=75, filter_method='mean', phase_offset_correction=True)
    cube, _, errors = prepare_recovery_source(path, 'A1', raw, drift, [(0, 0)], [25])
    assert errors[6, 6]
    assert np.all(cube[:, 6, 6] == 0)
    assert not errors[7, 7]
    assert cube[24, 7, 7] == pytest.approx(cube[25, 7, 7])
    np.testing.assert_allclose(np.diff(cube[25:, 7, 7]), 2., atol=1e-8)
    # Disabling phase correction preserves the previous missing-data policy.
    original, _, original_errors = prepare_recovery_source(
        path, 'A1', raw, {**drift, 'phase_offset_correction': False}, [(0, 0)], [25]
    )
    assert not original_errors[6, 6]
    assert original[25, 6, 6] > 0
