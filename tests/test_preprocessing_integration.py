from dataclasses import replace
import numpy as np
import pytest

from nanobio_core.epic_cardio import processing
from nanobio_core.preprocessing import PreprocessConfig, SPATIAL, EXCLUDED
from nanobio_core.preprocessing.cli import main
from nanobio_core.tests.test_preprocessing_pipeline import raw_well


@pytest.mark.parametrize('profile', ['default', 'legacy', 'observed_only'])
def test_opt_in_entrypoints_share_identical_values_and_config(profile, monkeypatch):
    raw = raw_well()
    raw.values_nm[40:, 8, 8] = 0
    codes = raw.codes.copy()
    codes[40:, 8, 8] = 0
    config = PreprocessConfig.profile(profile)
    config = replace(config, background=replace(config.background, method='none')
                     if profile != 'legacy' else config.background)
    params = dict(pipeline_profile=profile, pipeline_config=config.to_dict(),
                  pipeline_codes={'A1': codes},
                  signal_range=dict(range_type=processing.RangeType.INDIVIDUAL_POINT,
                                    ranges=[0, len(raw.time_s)]),
                  drift_correction=dict(threshold=75, filter_method='mean', background_selector=True))
    localization = dict(threshold_range=[0, 5000], neighbourhood_size=3, error_mask_filtering=True)
    monkeypatch.setattr(processing, 'WELL_NAMES', ['A1'])
    captured = []

    def maxima(cube, **kwargs):
        captured.append(cube)
        return []

    monkeypatch.setattr(processing, 'calculate_cell_maximas', maxima)
    from nanobio_core.preprocessing import pipeline
    original = pipeline.preprocess_well
    results = []
    configurations = []

    def record(raw, config, curation=None):
        result = original(raw, config, curation)
        results.append(result)
        configurations.append(config)
        return result

    import nanobio_core.preprocessing as package
    monkeypatch.setattr(package, 'preprocess_well', record)
    values, time, phases, picks, selected = processing.preprocessing(
        params, {'A1': raw.values_nm}, raw.time_s, raw.phases, {'A1': [(5, 5)]})
    localized = processing.localization(params, localization, {'A1': raw.values_nm}, phases,
                                        selected, {'A1': [(5, 5)]})
    np.testing.assert_array_equal(values['A1'], localized['A1'][0])
    np.testing.assert_array_equal(results[0].values_pm, results[1].values_pm)
    np.testing.assert_array_equal(results[0].method, results[1].method)
    assert configurations[0] == configurations[1]
    assert results[0].provenance == results[1].provenance
    if profile == 'default':
        assert np.all(values['A1'][results[0].method == EXCLUDED] == 0)
        spatial = results[1].method == SPATIAL
        assert spatial.any()
        assert np.all(captured[0][spatial] == -np.inf)


@pytest.mark.parametrize('arguments', [[], ['--out', 'inside'], ['--out', '.']])
def test_cli_run_requires_explicit_external_output(tmp_path, monkeypatch, arguments):
    measurement = tmp_path / 'measurement'
    measurement.mkdir()
    monkeypatch.chdir(measurement)
    with pytest.raises(SystemExit) as error:
        main(['run', str(measurement), *arguments])
    assert error.value.code == 2
    assert not list(measurement.iterdir())


def test_cli_run_review_diff_outputs(tmp_path, monkeypatch):
    from nanobio_core.preprocessing import cli, preprocess_well
    from nanobio_core.tests.test_preprocessing_pipeline import config_without_background
    result = preprocess_well(raw_well(), config_without_background())
    calls = []

    def measurement(path, config, **kwargs):
        calls.append((config.profile_name, kwargs))
        if kwargs.get('cache'):
            kwargs['cache'].put(result)
        return {'A1': result}

    monkeypatch.setattr(cli, 'preprocess_measurement', measurement)
    root = tmp_path / 'measurement'
    root.mkdir()
    output = tmp_path / 'output'
    main(['run', str(root), '--out', str(output)])
    assert len(list(output.glob('*.npz'))) == 1
    assert (output / 'run-receipts.json').exists()
    main(['review', str(root), '--out', str(output)])
    assert (output / 'review-receipts.json').exists()
    main(['diff', str(root), '--out', str(output)])
    assert (output / 'diff-receipts.json').exists()
    assert not list(root.iterdir())
    main(['run', str(root), '--in-place'])
    assert list((root / 'result/.cache/preprocess').glob('*.npz'))
