"""Deterministic well orchestration and content-addressed result storage."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import re
import tempfile
import threading

import numpy as np

from ..epic_cardio import data_correction, measurement_load, confirmed_pulse, pixel_recovery
from ..processing.border_filter import allowed_pixel_mask_for_well
from . import stages, recovery
from .config import PreprocessConfig
from .curation import load_curation, curation_hash
from .model import RawWell, PreprocessResult, OBSERVED, TEMPORAL, MISSING

_LEGACY_LOCK = threading.Lock()


def code_version():
    """Hash implementation content, including legacy/shared numerical dependencies."""
    digest = hashlib.sha256()
    files = set(Path(__file__).parent.glob('*.py'))
    for module in (data_correction, measurement_load, confirmed_pulse, pixel_recovery):
        files.add(Path(inspect.getfile(module)))
    files.update(Path(__file__).parents[1].joinpath('processing').glob('*.py'))
    files.add(Path(inspect.getfile(data_correction.find_local_maxima)))
    for path in sorted(files):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def raw_hash(raw):
    digest = hashlib.sha256()
    for name in ('values_nm', 'codes', 'time_s', 'allowed', 'cell_mask'):
        value = getattr(raw, name)
        if value is None:
            digest.update(b'none')
            continue
        array = np.ascontiguousarray(value)
        digest.update(str((name, array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    digest.update(json.dumps([raw.phases, raw.well, raw.phase_source]).encode())
    return digest.hexdigest()


def provenance_for(raw, config, curation=None):
    return dict(raw_sha256=raw_hash(raw), config_hash=config.config_hash(),
                code_version=code_version(), curation_hash=curation_hash(curation))


def cache_key(provenance):
    return hashlib.sha256(json.dumps(provenance, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class Cache:
    """Explicit writable directory; arrays and JSON, never pickle."""

    def __init__(self, path):
        self.path = Path(path)

    def get(self, provenance):
        path = self.path / f'{cache_key(provenance)}.npz'
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as bundle:
            saved = json.loads(str(bundle['provenance']))
            if saved != provenance:
                raise ValueError('Cache provenance mismatch')
            return PreprocessResult(bundle['values_pm'], bundle['method'], bundle['coverage'],
                                    json.loads(str(bundle['receipts'])), saved)

    def put(self, result):
        self.path.mkdir(parents=True, exist_ok=True)
        path = self.path / f'{cache_key(result.provenance)}.npz'
        with tempfile.NamedTemporaryFile(dir=self.path, suffix='.npz', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                np.savez_compressed(stream, values_pm=result.values_pm, method=result.method,
                                    coverage=result.coverage, receipts=json.dumps(result.receipts),
                                    provenance=json.dumps(result.provenance))
                stream.flush()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return path


def _legacy(raw, config, provenance):
    settings = config.background
    coords = list(settings.coordinates)
    # select_indices uses global RNG; isolate it to keep calls repeatable and caller state intact.
    with _LEGACY_LOCK:
        saved_rng = np.random.get_state()
        np.random.seed(config.legacy.random_seed)
        try:
            aligned, translations = data_correction.correct_interphase_well_shifts(
                raw.values_nm, raw.phases, settings.threshold_pm, coords, settings.method,
                return_translations=True)
            values, coordinates, mask = data_correction.correct_well(
                aligned, threshold=settings.threshold_pm, coords=coords, mode=settings.method,
                pixel_filtering=config.legacy.pixel_filtering)
        finally:
            np.random.set_state(saved_rng)
    method = np.full(values.shape, OBSERVED, dtype=np.uint8)
    method[:, mask] = MISSING
    jumps = {}
    if config.legacy.frame_jump_correction:
        corrected, jumps = data_correction.interpolate_frame_jumps(
            values, config.legacy.frame_jump_threshold, raw.phases)
        method[corrected != values] = TEMPORAL
        values = corrected
    offsets = []
    if config.legacy.phase_offset_correction:
        values, offsets = data_correction.correct_phase_offsets(values, raw.phases)
    receipts = dict(legacy=dict(profile='legacy', mask=mask.tolist()),
                    interphase_registration=dict(shifts=translations),
                    background=dict(coordinates=coordinates),
                    within_phase_recovery=jumps, phase_offset=offsets)
    return PreprocessResult(values, method, np.mean(method == OBSERVED, axis=0), receipts, provenance)


def preprocess_well(raw, config, curation=None):
    """Return signed pm values, per-sample methods, stage receipts and provenance."""
    curation = load_curation(curation)
    provenance = provenance_for(raw, config, curation)
    if config.profile_name == 'legacy':
        if curation:
            raise ValueError('Legacy regression profile does not support curation')
        return _legacy(raw, config, provenance)
    state = stages.validity(raw, config)
    for stage in (stages.active_area, stages.interphase_registration, stages.within_phase_recovery):
        state = stage(state, config)
    state = stages.curated_repairs(state, config, curation)
    for stage in (stages.reference, stages.background, stages.phase_offset):
        state = stage(state, config)
    return stages.finalize(state, config, provenance)


def _process_task(task):
    raw, config, curation, cache = task
    provenance = provenance_for(raw, config, curation)
    result = cache.get(provenance) if cache is not None else None
    if result is None:
        result = preprocess_well(raw, config, curation)
        if cache is not None:
            cache.put(result)
    return raw.well, result


def load_raw_measurement(path):
    """Use the existing wavelength/time decoder and retain original DMR codes."""
    path = Path(path)
    values, time = measurement_load.load_measurement(str(path))
    if values is None or time is None:
        raise ValueError(f'Cannot load measurement: {path}')
    files = sorted((path / 'DMR').iterdir(), key=lambda p: int(re.sub(r'\D', '', p.name)))
    power = next(p for p in path.iterdir() if '_wl_power' in p.name.lower())
    dtype = np.uint16 if power.stat().st_size == 614400 else np.uint8
    codes = np.stack([np.fromfile(p, dtype=dtype).reshape(240, 320) for p in files])
    if len(time) != len(values):
        raise ValueError('Measurement time/cube length mismatch')
    parameters = {}
    for candidate in (path / 'result/.metadata/parameters.json', path / '.metadata/parameters.json'):
        if candidate.exists():
            parameters = json.loads(candidate.read_text())
            break
    preprocessing = parameters.get('preprocessing', {})
    saved = parameters.get('phases', preprocessing.get('phases'))
    full_phases = (tuple(saved) if saved is not None
                   else tuple((np.flatnonzero(np.diff(time) > 60) + 1).tolist()))
    source = 'project' if saved is not None else 'time_gaps_over_60_s'
    signal_range = preprocessing.get('signal_range', {})
    left, right = signal_range.get('ranges', [0, None])
    if signal_range.get('range_type', 0) == 0:
        left = 0 if left == 0 or left > len(full_phases) else full_phases[left - 1]
        right = len(time) if right is None else full_phases[right - 1]
    selected = slice(left, right)
    start, stop, _ = selected.indices(len(time))
    phases = tuple(int(p - start) for p in full_phases if start < p < stop)
    flip = preprocessing.get('flip', [True, False])
    value_wells = measurement_load.wl_map_to_wells(values[selected], flip)
    code_wells = measurement_load.wl_map_to_wells(codes[selected], flip)
    time = np.asarray(time)[selected]
    localization = parameters.get('localization', {})
    raw = {name: RawWell(cube, code_wells[name], np.asarray(time), phases,
                         allowed_pixel_mask_for_well(localization, name, cube.shape[1:]), name, source)
           for name, cube in value_wells.items()}
    return raw, parameters


def preprocess_measurement(path, config, curation=None, wells=None, workers=None, cache=None):
    """Process selected wells independently; cache=None performs no writes."""
    raw_wells, parameters = load_raw_measurement(path)
    curation = load_curation(curation)
    names = list(raw_wells) if wells is None else list(wells)
    if set(names) - raw_wells.keys():
        raise ValueError(f'Unknown wells: {sorted(set(names) - raw_wells.keys())}')
    if cache is not None and not isinstance(cache, Cache):
        cache = Cache(cache)
    tasks = []
    for name in names:
        well_config = config
        picks = parameters.get('filter_ptss', {}).get(name, [])
        if picks and not config.background.coordinates:
            background = replace(config.background, coordinates=tuple(map(tuple, picks)))
            well_config = replace(config, background=background)
        tasks.append((raw_wells[name], well_config, curation, cache))
    if workers == 1 or len(tasks) < 2:
        return dict(map(_process_task, tasks))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return dict(executor.map(_process_task, tasks))
