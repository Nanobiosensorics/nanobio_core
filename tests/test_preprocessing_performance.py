"""Exact pre-optimization oracles and bounded execution/storage regressions."""

from concurrent.futures import Future
from dataclasses import replace
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from scipy import ndimage

from nanobio_core.preprocessing import (
    Cache, PreprocessConfig, RawWell, preprocess_measurement, preprocess_well,
)
from nanobio_core.preprocessing import pipeline, recovery, stages
from nanobio_core.preprocessing.model import EXCLUDED, MISSING, StageState

FIXTURE = Path(__file__).parent / 'fixtures/preprocessing-performance.npz'


def fixture_raw():
    with np.load(FIXTURE, allow_pickle=False) as bundle:
        return RawWell(bundle['values_nm'], bundle['codes'], bundle['time_s'],
                       tuple(map(int, bundle['phases'])), bundle['allowed'], 'A1')


def fixture_config(profile):
    config = PreprocessConfig.profile(profile)
    return replace(config, registration=replace(config.registration, enabled=False),
                   background=replace(config.background, coordinates=((3, 3), (4, 3))))


@pytest.mark.parametrize('profile', ['default', 'legacy', 'observed_only'])
def test_synthetic_preoptimization_bytes(profile):
    result = preprocess_well(fixture_raw(), fixture_config(profile))
    with np.load(FIXTURE, allow_pickle=False) as expected:
        for name in ('values_pm', 'method', 'coverage'):
            actual, baseline = getattr(result, name), expected[f'{profile}_{name}']
            assert actual.dtype == baseline.dtype
            assert actual.shape == baseline.shape
            assert actual.tobytes() == baseline.tobytes(), name


@pytest.mark.parametrize('length,radius', [(2, 5), (11, 1), (11, 5), (30, 7)])
def test_centered_median_nearest_edges_and_nonfinite_traces(length, radius):
    values = np.random.default_rng(73).normal(size=(length, 3, 4))
    values[:, 0, 0] = 4
    values[length // 2, 1, 1] = np.nan
    values[-1, 2, 2] = np.inf
    values[0, 2, 3] = -np.inf
    expected = ndimage.median_filter(values, size=(2 * radius + 1, 1, 1), mode='nearest')
    actual = recovery._centered_median(values, radius)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('radius', [1, 5])
def test_centered_median_preserves_signed_zero_bytes(radius):
    values = np.random.default_rng(12).choice([0.0, -0.0, 1.0, -1.0], size=(30, 3, 4))
    expected = ndimage.median_filter(values, size=(2 * radius + 1, 1, 1), mode='nearest')
    assert recovery._centered_median(values, radius).tobytes() == expected.tobytes()


@pytest.mark.parametrize('count', [1, 2, 5, 20])
def test_level_offsets_sparse_anchors_multiple_phases_and_receipts(count):
    raw = fixture_raw()
    config = fixture_config('default')
    config = replace(config, phase_offset=replace(config.phase_offset, anchor_frames=count))
    rng = np.random.default_rng(24)
    values = rng.normal(size=raw.values_nm.shape)
    methods = rng.integers(0, 7, size=values.shape, dtype=np.uint8)
    values[rng.random(values.shape) < 0.15] = np.nan
    methods[60:95, 5, 5] = MISSING
    methods[95:, 6, 6] = EXCLUDED
    allowed = raw.allowed.copy()
    allowed[:2] = False
    state = StageState(raw, values, methods, allowed, {})
    expected = state.copy()
    maps = []
    bounds = [0, *raw.phases, len(values)]
    for left, start, end in zip(bounds, bounds[1:], bounds[2:]):
        offset = np.full(allowed.shape, np.nan)
        for y, x in zip(*np.nonzero(allowed)):
            known = np.isfinite(expected.values[:, y, x])
            known &= ~np.isin(methods[:, y, x], [MISSING, EXCLUDED])
            before = np.flatnonzero(known[left:start]) + left
            after = np.flatnonzero(known[start:end]) + start
            if min(len(before), len(after)) < count:
                continue
            delta = (np.median(expected.values[before[-count:], y, x])
                     - np.median(expected.values[after[:count], y, x]))
            offset[y, x] = delta
            expected.values[start:end, y, x] += delta
        maps.append(dict(frame=start, offset_pm=offset.tolist(),
                         unsupported_pixels=int((allowed & ~np.isfinite(offset)).sum())))
    actual = stages.phase_offset(state, config)
    assert actual.values.tobytes() == expected.values.tobytes()
    assert actual.method.tobytes() == methods.tobytes()
    assert state.values.tobytes() == values.tobytes()
    assert json.dumps(actual.receipts['phase_offset']['maps']) == json.dumps(maps)
    state.allowed[:] = False
    assert stages.phase_offset(state, config).values.tobytes() == values.tobytes()


def test_cache_uncompressed_writes_and_old_compressed_reads(tmp_path):
    result = preprocess_well(fixture_raw(), fixture_config('default'))
    cache = Cache(tmp_path)
    path = cache.put(result)
    with zipfile.ZipFile(path) as archive:
        assert all(member.compress_type == zipfile.ZIP_STORED for member in archive.infolist())
    with np.load(path, allow_pickle=False) as bundle:
        payload = {name: bundle[name] for name in bundle.files}
    np.savez_compressed(path, **payload)
    with zipfile.ZipFile(path) as archive:
        assert all(member.compress_type == zipfile.ZIP_DEFLATED for member in archive.infolist())
    restored = cache.get(result.provenance)
    for name in ('values_pm', 'method', 'coverage'):
        assert getattr(restored, name).tobytes() == getattr(result, name).tobytes()
    assert json.dumps(restored.receipts) == json.dumps(result.receipts)
    assert restored.provenance == result.provenance


@pytest.mark.parametrize('profile', ['default', 'legacy', 'observed_only'])
def test_parallel_profiles_equal_bytes_and_preserve_order(tmp_path, monkeypatch, profile):
    raw = fixture_raw()
    wells = {'A1': raw, 'A2': replace(raw, well='A2'), 'A3': replace(raw, well='A3')}
    monkeypatch.setattr(pipeline, 'load_raw_measurement', lambda path: (wells, {}))
    config = fixture_config(profile)
    names = ['A3', 'A1', 'A2']
    serial = preprocess_measurement('fixture', config, wells=names)
    parallel = preprocess_measurement('fixture', config, wells=names, workers=2, cache=tmp_path)
    assert list(parallel) == names
    for name in names:
        for attribute in ('values_pm', 'method', 'coverage'):
            assert getattr(parallel[name], attribute).tobytes() == getattr(serial[name], attribute).tobytes()
        assert json.dumps(parallel[name].receipts) == json.dumps(serial[name].receipts)
        assert parallel[name].provenance == serial[name].provenance


@pytest.mark.parametrize('fail', [False, True])
def test_parallel_submission_and_staging_bound_and_cleanup(monkeypatch, fail):
    directories = []
    raw = fixture_raw()

    class Executor:
        def __init__(self, max_workers):
            assert max_workers == 2

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, directory, metadata, config, curation, cache):
            directories.append(directory)
            assert len(list(directory.parent.iterdir())) <= 2
            values = np.load(directory / 'values_nm.npy', mmap_mode='r')
            assert isinstance(values, np.memmap)
            assert not values.flags.writeable
            future = Future()
            if fail:
                future.set_exception(RuntimeError('Worker failed'))
            else:
                future.set_result((metadata['well'], metadata['well']))
            return future

    monkeypatch.setattr(pipeline, 'ProcessPoolExecutor', Executor)
    monkeypatch.setattr(pipeline, 'wait', lambda pending, **kwargs: ([next(reversed(pending))], []))
    tasks = [(replace(raw, well=f'A{i}'), fixture_config('default'), [], None) for i in range(5)]
    if fail:
        with pytest.raises(RuntimeError, match='Worker failed'):
            pipeline._parallel_tasks(tasks, 2)
    else:
        assert list(pipeline._parallel_tasks(tasks, 2)) == [f'A{i}' for i in range(5)]
    assert directories
    assert not directories[0].parent.exists()


@pytest.mark.parametrize('workers', [0, -1])
def test_invalid_worker_count_rejected_before_loading(monkeypatch, workers):
    def load(path):
        pytest.fail('Invalid workers must fail before loading')

    monkeypatch.setattr(pipeline, 'load_raw_measurement', load)
    with pytest.raises(ValueError, match='positive integer'):
        preprocess_measurement('fixture', fixture_config('default'), workers=workers)
