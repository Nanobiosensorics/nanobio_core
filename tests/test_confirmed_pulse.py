import numpy as np
import pytest

from nanobio_core.epic_cardio.confirmed_pulse import repair_confirmed_pulse


def recording():
    time = np.cumsum(np.r_[0, np.tile([12, 17, 23], 10)])
    values = 850 + time[:, None, None] * np.ones((1, 7, 7)) * 0.00001
    return values, np.ones_like(values, dtype=bool), time, np.ones((7, 7), dtype=bool)


def test_complete_six_sample_pulse_uses_clean_original_anchors():
    raw, valid, time, allowed = recording()
    damaged = raw.copy()
    damaged[10:16, 2:5, 2:5] += np.array([-0.08, 1.8, .5, .6, .7, .2])[:, None, None]
    allowed[:, 0] = False
    damaged[10:16, :, 0] += 2
    result, accepted, receipt = repair_confirmed_pulse(damaged, valid, time, [], allowed, 10, 16)
    assert receipt['repaired_pixels'] == 9
    np.testing.assert_allclose(result[:, 2:5, 2:5], raw[:, 2:5, 2:5], atol=1e-12)
    assert np.array_equal(result[:, 0, :], damaged[:, 0, :])
    assert np.array_equal(result[:, :, 0], damaged[:, :, 0])
    assert np.array_equal(result[:10], damaged[:10])
    assert np.array_equal(result[16:], damaged[16:])
    assert not accepted[:, 0].any()


def test_no_repair_for_persistent_step_or_missing_anchor():
    raw, valid, time, allowed = recording()
    raw[10:, 2:5, 2:5] += .5
    result, accepted, _ = repair_confirmed_pulse(raw, valid, time, [], allowed, 10, 16)
    assert not accepted.any()
    np.testing.assert_array_equal(result, raw)
    raw[16:, 2:5, 2:5] -= .5
    valid[9, 3, 3] = False
    _, accepted, _ = repair_confirmed_pulse(raw, valid, time, [], allowed, 10, 16)
    assert not accepted[3, 3]


def test_phase_gap_cannot_be_interpolated_and_isolated_excursion_stays():
    raw, valid, time, allowed = recording()
    raw[10:16, 3, 3] += .5
    _, accepted, _ = repair_confirmed_pulse(raw, valid, time, [], allowed, 10, 16)
    assert not accepted.any()
    with pytest.raises(ValueError, match='phase boundary'):
        repair_confirmed_pulse(raw, valid, time, [13], allowed, 10, 16)


def test_contaminated_clean_window_is_not_an_anchor():
    raw, valid, time, allowed = recording()
    raw[10:16, 2:5, 2:5] += .5
    raw[9, 3, 3] += 1
    result, accepted, _ = repair_confirmed_pulse(raw, valid, time, [], allowed, 10, 16)
    assert not accepted[3, 3]
    np.testing.assert_array_equal(result[:, 3, 3], raw[:, 3, 3])


def test_sensor_edge_cannot_supply_reflected_neighbor_evidence():
    raw, valid, time, allowed = recording()
    raw[10:16, 0, 0] += .5
    _, accepted, _ = repair_confirmed_pulse(raw, valid, time, [], allowed, 10, 16)
    assert not accepted.any()
