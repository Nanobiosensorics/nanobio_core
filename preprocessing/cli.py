"""Run, review receipts, or compare profiles without implicit measurement writes."""

import argparse
import json
from pathlib import Path
import numpy as np

from .config import PreprocessConfig
from .pipeline import Cache, preprocess_measurement


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'review', 'diff'):
        command = commands.add_parser(name)
        command.add_argument('measurement', type=Path)
        command.add_argument('--curation', type=Path)
        command.add_argument('--workers', type=int, default=1)
        command.add_argument('--wells', nargs='+')
        command.add_argument('--profile', default='default')
        command.add_argument('--out', '--output', dest='out', type=Path)
        command.add_argument('--in-place', action='store_true')
        if name == 'diff':
            command.add_argument('--a', default='legacy')
            command.add_argument('--b', default='default')
    args = parser.parse_args(argv)
    root = args.measurement.resolve()
    output = args.out.resolve() if args.out else None
    if args.command in ('run', 'review') and output is None:
        if not args.in_place:
            parser.error('run/review requires --out unless --in-place is passed')
        output = root / 'result/.cache/preprocess'
    if output is not None and (output == root or root in output.parents) and not args.in_place:
        parser.error('An output inside the measurement requires --in-place')
    options = dict(curation=args.curation, wells=args.wells, workers=args.workers)
    if args.command == 'diff':
        a = preprocess_measurement(root, PreprocessConfig.profile(args.a), **options)
        b = preprocess_measurement(root, PreprocessConfig.profile(args.b), **options)
        report = {}
        for name in a:
            difference = np.abs(a[name].values_pm - b[name].values_pm)
            finite = difference[np.isfinite(difference)]
            report[name] = dict(max_difference_pm=float(finite.max()) if finite.size else None,
                                method_changes=int((a[name].method != b[name].method).sum()),
                                availability_changes=int((np.isfinite(a[name].values_pm)
                                                          != np.isfinite(b[name].values_pm)).sum()))
    else:
        results = preprocess_measurement(root, PreprocessConfig.profile(args.profile),
                                         cache=Cache(output) if args.command == 'run' else None, **options)
        report = {name: dict(receipts=result.receipts, provenance=result.provenance)
                  for name, result in results.items()}
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / f'{args.command}-receipts.json').write_text(json.dumps(report, indent=2))
    else:
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
