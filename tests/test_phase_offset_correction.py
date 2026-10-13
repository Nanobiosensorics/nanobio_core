import numpy as np
import pytest

from epic_cardio.data_correction import correct_phase_offsets


def test_offsets_preserve_first_phase_and_kinetics_across_three_phases():
    raw = np.array([
        [0, 0], [10, 20], [100, -50], [105, -57], [300, 100], [298, 110],
    ], float)[:, None, :]
    corrected, summary = correct_phase_offsets(raw, [2, 4])
    np.testing.assert_allclose(
        corrected[:, 0], [[0, 0], [10, 20], [10, 20], [15, 13], [15, 13], [13, 23]],
    )
    np.testing.assert_array_equal(raw[:2], corrected[:2])
    np.testing.assert_array_equal(
        raw[:, 0], [[0, 0], [10, 20], [100, -50], [105, -57], [300, 100], [298, 110]],
    )
    assert [row['frame'] for row in summary] == [2, 4]
    weights = np.array([.25, .75])
    iws = corrected[:, 0] @ weights * 625
    np.testing.assert_allclose(iws[[2, 4]], iws[[1, 3]])
    np.testing.assert_allclose(corrected.max((1, 2))[[2, 4]], corrected.max((1, 2))[[1, 3]])


def test_nonfinite_endpoints_are_not_propagated_to_other_pixels():
    raw = np.array([[0, 0], [10, np.nan], [30, 40], [32, 42]], float)[:, None, :]
    corrected, summary = correct_phase_offsets(raw, [2])
    np.testing.assert_allclose(corrected[2:, 0], [[10, 40], [12, 42]])
    assert summary[0]['valid_pixels'] == 1


def test_single_phase_and_negative_changes_are_preserved():
    raw = np.array([0, 10, 100, 80], float)[:, None, None]
    corrected, _ = correct_phase_offsets(raw, [2])
    np.testing.assert_array_equal(corrected[:, 0, 0], [0, 10, 10, -10])
    np.testing.assert_array_equal(correct_phase_offsets(raw, [])[0], raw)


def test_rejects_non_cube_input():
    with pytest.raises(ValueError, match='signal cube'):
        correct_phase_offsets(np.zeros((2, 3)), [1])
