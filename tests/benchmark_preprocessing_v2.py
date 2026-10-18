"""Read-only C1 benchmark using frozen verifier NPZ inputs (not a pytest test).

Run with PYTHONPATH=. python nanobio_core/tests/benchmark_preprocessing_v2.py --inputs PERF_DIR.
Loading and output comparisons are excluded from alternating warm-process timings.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from nanobio_core.preprocessing import PreprocessConfig, RawWell, preprocess_well


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--data', type=Path,
                        default=Path('/home/balint/data/nanobio/cardio-fluidfm-data/additional'))
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    summary = {}
    for date, project in [('20260724', 'result1'), ('20260323', 'result')]:
        input_path = next((args.inputs / f'inputs-{date}' / '_cache').glob('C1-raw-*.npz'))
        with np.load(input_path, allow_pickle=False) as bundle:
            raw = RawWell(bundle['raw'], bundle['codes'], bundle['time'],
                          tuple(map(int, bundle['phases'])), bundle['allowed'], 'C1',
                          'project_range_with_time_gap_boundaries')
        parameters = json.loads((args.data / date / project / '.metadata/parameters.json').read_text())
        entries = json.loads((args.inputs / 'frozen-curation.json').read_text()) if date == '20260724' else []
        timings = {profile: [] for profile in ['default', 'default_v2']}
        outputs = {}
        for trial in range(args.repeats):
            for profile in (['default', 'default_v2'] if trial % 2 == 0 else ['default_v2', 'default']):
                settings = PreprocessConfig.profile(profile)
                settings = replace(settings, background=replace(
                    settings.background, coordinates=tuple(map(tuple, parameters['filter_ptss']['C1']))))
                started = time.perf_counter()
                result = preprocess_well(raw, settings, entries)
                timings[profile].append(time.perf_counter() - started)
                if profile in outputs:
                    for field in ['values_pm', 'method', 'coverage']:
                        assert getattr(result, field).tobytes() == getattr(outputs[profile], field).tobytes()
                outputs[profile] = result
                print(json.dumps(dict(date=date, profile=profile, trial=trial,
                                      seconds=timings[profile][-1])), flush=True)
        summary[date] = dict(input=str(input_path), shape=raw.values_nm.shape,
                             seconds=timings,
                             median_seconds={p: float(np.median(t)) for p, t in timings.items()},
                             events=[dict(start=e['start_frame'], end=e['end_frame'], source=e['source'],
                                          counts=e['counts'])
                                     for e in outputs['default_v2'].receipts['well_events']['events']])
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
