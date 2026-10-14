import numpy as np
import pytest

from nanobio_core.epic_cardio.pixel_recovery import (
    EXCLUDED, MISSING, SPATIAL, STITCHED, TEMPORAL, RecoveryConfig,
    fill_spatial_frame, recover_pixels,
)


def fixture():
    time = np.arange(50) * 12.0
    yy, xx = np.indices((12, 12))
    values = 830 + time[:, None, None] * 0.0002 + yy * 0.002 + xx * 0.003
    allowed = (xx >= 2) & (xx < 10) & (yy >= 2) & (yy < 10)
    return values, np.ones_like(values, dtype=bool), time, allowed


def test_short_dropout_and_nonzero_spike_preserve_borders_and_observations():
    values, valid, time, allowed = fixture()
    damaged = values.copy()
    valid[10:13, 5, 5] = False
    damaged[10:13, 5, 5] = 0
    damaged[20, 6, 6] += 0.3
    result = recover_pixels(damaged, valid, time, [], allowed)
    np.testing.assert_allclose(result.values[10:13, 5, 5], values[10:13, 5, 5])
    np.testing.assert_allclose(result.values[20, 6, 6], values[20, 6, 6])
    assert np.all(result.methods[10:13, 5, 5] == TEMPORAL)
    assert result.methods[20, 6, 6] == TEMPORAL
    np.testing.assert_array_equal(result.values[:, ~allowed], damaged[:, ~allowed])
    assert np.all(result.methods[:, ~allowed] == EXCLUDED)


def test_terminal_cluster_uses_original_anchors_and_preserves_gradient():
    values, valid, time, allowed = fixture()
    damaged = values.copy()
    valid[25:, 4:7, 4:7] = False
    damaged[25:, 4:7, 4:7] = 0
    result = recover_pixels(damaged, valid, time, [], allowed)
    np.testing.assert_allclose(result.values[25:, 4:7, 4:7], values[25:, 4:7, 4:7], atol=1e-9)
    assert np.all(result.methods[25:, 4:7, 4:7] == SPATIAL)
    assert np.max(result.spatial_radius) == 2
    assert np.all(result.donor_count[25:, 4:7, 4:7] > 0)


def test_long_returning_gap_is_not_interpolated_or_extrapolated_across_phases():
    values, valid, time, allowed = fixture()
    valid[15:25, 5, 5] = False
    valid[28:30, 6, 6] = False
    damaged = values.copy()
    damaged[~valid] = 0
    result = recover_pixels(damaged, valid, time, [30], allowed)
    assert np.isnan(result.values[15:25, 5, 5]).all()
    assert np.isnan(result.values[28:30, 6, 6]).all()
    assert np.all(result.methods[15:25, 5, 5] == MISSING)


def test_coherent_offsets_are_stitched_without_removing_within_phase_gradient():
    values, valid, time, allowed = fixture()
    damaged = values.copy()
    damaged[20:, allowed] += 0.3
    result = recover_pixels(damaged, valid, time, [], allowed)
    np.testing.assert_allclose(result.values[:, allowed], values[:, allowed], atol=1e-8)
    assert result.summary['applied_steps'] > 0
    assert np.all(result.methods[20:, allowed] == STITCHED)


def test_short_coherent_returning_pulse_does_not_create_a_stitched_plateau():
    values, valid, time, allowed = fixture()
    # Recorded 0724 C1 pulse shape, including its noisy return to baseline.
    trace_pm = [0, 4.8, -3.6, 1.4, 3.6, 10.4, 2.8, -2.4, -6.2, 3.4,
                2.6, -3.6, -4.6, -114.8, 369.2, 302.8, 683, -36.8, -46.6, -4.2,
                -42.8, -0.4, -4.2, 1.2, -0.6, -0.4, -6.6, -2.8, 5.8, -5.8,
                0, 2, -8.8, 3, -11.2, -4, -11.6, -6.4, -24.2, -1.4]
    pulse = np.pad(np.array(trace_pm), (5, 5), mode='edge') / 1000
    damaged = values[0][None] + np.broadcast_to(pulse[:, None, None], values.shape)
    result = recover_pixels(damaged, valid, time, [], allowed)
    assert result.summary['applied_steps'] == 0
    assert np.all(result.methods[18:22, allowed] == TEMPORAL)
    np.testing.assert_allclose(result.values[22:, allowed], damaged[22:, allowed], atol=1e-8)


def test_returning_pulse_across_phase_boundary_is_not_repaired():
    values, valid, time, allowed = fixture()
    damaged = values.copy()
    damaged[23:27, allowed] += 0.7
    result = recover_pixels(damaged, valid, time, [25], allowed)
    assert not np.any(result.methods[23:27, allowed] == TEMPORAL)
    assert result.summary['applied_steps'] == 0


@pytest.mark.parametrize('trace_pm', [
    [0, 6.4, 3, 7.6, -4, -31.2, 417, 339.2, 596.6, 880, 97, 4.6, -25.6, 3.4, -1],
    [0, -0.2, 0.2, 2.2, 5.8, 10, 335.6, 474.6, 683.4, 1442.6, 150.6, 12.8, -26.2, 11.8, 7],
])
def test_five_frame_recorded_pulse_is_not_masked_by_step_noise(trace_pm):
    values, valid, time, allowed = fixture()
    pulse = np.pad(np.array(trace_pm), (8, 27), mode='edge') / 1000
    damaged = values[0][None] + np.broadcast_to(pulse[:, None, None], values.shape)
    result = recover_pixels(damaged, valid, time, [], allowed)
    assert result.summary['applied_steps'] == 0
    assert np.all(result.methods[14:19, allowed] == TEMPORAL)
    np.testing.assert_allclose(result.values[19:, allowed], damaged[19:, allowed], atol=1e-8)


def test_long_returning_ramp_is_not_converted_to_a_permanent_offset():
    values, valid, time, allowed = fixture()
    damaged = values.copy()
    damaged[20:28, allowed] += np.arange(1, 9)[:, None] * 0.1
    result = recover_pixels(damaged, valid, time, [], allowed)
    assert result.summary['applied_steps'] == 0
    np.testing.assert_allclose(result.values[28:, allowed], damaged[28:, allowed], atol=1e-8)


@pytest.mark.parametrize('trace_pm', [
    [0, 6.6, 4.2, -8.6, -10, -3.4, -9.2, -1.4, -3.6, -5.2, -11.6, 2.2,
     -3.4, -88.2, 1748.6, 607.8, 455.2, 549.4, 231.2, -13, -33, -8.8,
     -9.2, -6.2, -11, -1.8, -4, -4.8, -2.4, -9.2, -1.2],
    [0, -4.6, -9.6, -6.2, -2.4, -11, -10.6, -8.4, -11.8, -4.8, -7.2, 1.6,
     -8.6, -81.2, 919.2, 329.8, 293.8, 283.8, 61.4, -1.6, -20.2, -4.6,
     -5.6, 1.2, 7.8, -7.2, -15, -2.8, 3.8, -6.6, -1],
    [0, 21.6, 32.6, 31.8, 39.8, 55.4, 45, 50.8, 46.2, 48.4, 64.2, 52.4,
     85.4, 7.2, 892.2, 490.6, 977.4, 1262.4, 206, 117.8, 90.4, 124.6,
     136.6, 139.2, 137.6, 128.4, 159.6, 152.8, 166.6, 178.2, 183],
])
def test_recorded_overshoot_and_noisy_adhesion_trend_do_not_create_a_plateau(trace_pm):
    values, valid, time, allowed = fixture()
    pulse = np.pad(np.array(trace_pm), (8, 11), mode='edge') / 1000
    damaged = values[0][None] + np.broadcast_to(pulse[:, None, None], values.shape)
    result = recover_pixels(damaged, valid, time, [], allowed)
    assert result.summary['applied_steps'] == 0
    np.testing.assert_allclose(result.values[35:, allowed], damaged[35:, allowed], atol=1e-8)


def test_isolated_directional_drop_requires_evidence_and_phase_jump_is_preserved():
    values, valid, time, allowed = fixture()
    damaged = values.copy()
    damaged[20:, 5, 5] -= 0.3
    result = recover_pixels(damaged, valid, time, [], allowed)
    assert not result.summary['applied_steps']
    np.testing.assert_allclose(result.values[:, 5, 5], damaged[:, 5, 5])
    phase = values.copy()
    phase[25:, allowed] += 0.3
    result = recover_pixels(phase, valid, time, [25], allowed)
    np.testing.assert_allclose(result.values[:, allowed], phase[:, allowed])


def test_expanding_radius_reaches_distant_original_donors_without_using_border():
    yy, xx = np.indices((20, 20))
    original = xx * 3.0 + yy * 2
    allowed = (xx >= 3) & (xx < 17) & (yy >= 3) & (yy < 17)
    targets = (xx >= 9) & (xx <= 10) & (yy >= 9) & (yy <= 10)
    missing = (xx >= 7) & (xx <= 12) & (yy >= 7) & (yy <= 12)
    damaged = original.copy()
    damaged[missing] = np.nan
    damaged[~allowed] = 1e6
    result, radius, donors = fill_spatial_frame(damaged, allowed & ~missing, targets, allowed)
    np.testing.assert_allclose(result[targets], original[targets])
    assert np.min(radius[targets]) >= 4
    assert np.min(donors[targets]) > 0
    np.testing.assert_array_equal(result[~allowed], damaged[~allowed])
    assert np.isnan(result[missing & ~targets]).all()


def test_unsupported_and_oversized_regions_remain_missing():
    values, valid, time, allowed = fixture()
    image = values[0].copy()
    image[allowed] = np.nan
    result, radius, _ = fill_spatial_frame(image, ~allowed, allowed, allowed,
                                          RecoveryConfig(max_cluster_area=3))
    assert np.isnan(result[allowed]).all()
    assert not radius.any()


def test_one_sample_phases_do_not_fail_or_bridge_boundaries():
    values, valid, time, allowed = fixture()
    result = recover_pixels(values, valid, time, [1, 2], allowed)
    np.testing.assert_array_equal(result.values, values)


def test_invalid_zero_majority_does_not_reject_interleaved_recorded_signal():
    values, valid, time, allowed = fixture()
    valid[10:30, 5, 5] = False
    valid[[10, 14, 18, 22, 26], 5, 5] = True
    damaged = values.copy()
    damaged[~valid] = 824
    result = recover_pixels(damaged, valid, time, [], allowed)
    recorded = valid[:, 5, 5]
    np.testing.assert_allclose(result.values[recorded, 5, 5], values[recorded, 5, 5])
    assert np.all(result.methods[[14, 18, 22, 26], 5, 5] == 0)
