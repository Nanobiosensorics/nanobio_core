# Unified preprocessing performance

Recovery's two nearest-edge temporal rank filters use Bottleneck moving medians
for finite traces. SciPy remains the oracle for nonfinite traces and negative
zero ties. Returning-pulse eligibility uses integer missing-sample prefix counts
and batched right-anchor medians, retaining sequential candidate suppression and
the original scalar fits, noise estimates, interpolation and threshold decisions.

Level phase offsets select the same first/last finite non-MISSING/non-EXCLUDED
anchors in time order, including repaired samples. Selection uses partitioning
in batches of 256 pixels. Boundaries are corrected sequentially, using the already
corrected preceding phase. Single and extrapolate arithmetic remains unchanged.

Optional well parallelism defaults to serial execution. Positive `workers`
limits both active submissions and temporary well snapshots; workers open NPY
arrays read-only instead of receiving raw cubes through pickle. Temporary storage
is removed after each completed well and on failure. Final arrays/receipts must
still return to the parent, and the dictionary retains every requested result.
Full measurement decoding and total output memory are not bounded by `workers`.

Cache writes use atomic temporary-file replacement and uncompressed NPZ, retaining
float64, receipts and content/provenance checks. NPZ auto-detection reads existing
compressed entries without migration. Uncompressed storage uses more disk space.
Receipt list/JSON construction remains a significant memory and read-time cost.

## Frozen validation, 2026-10-14

Python: `/home/balint/anaconda3/envs/nanobio-py311/bin/python`.
Read-only review inputs were the `scratchpad/perf/inputs-20260724` and
`inputs-20260323` C1 NPZ files, six `*-baseline.npz` files and
`frozen-curation.json` supplied with the performance review. Saved project
background coordinates and legacy drift settings were used. July default used
the original reviewed interval; other profiles had no curation. Explicit
`curation.method='per_pixel'` reproduces b6305df's frozen configuration while
preserving separate uncommitted curation changes (the default advanced from
common-mode to low-rank during this work).

Times below are medians of three alternating pre-edit/optimized `preprocess_well`
calls in the same process, without result caching or profiling. Loading and
comparisons are excluded; original implementation snapshots reproduced all
baseline bytes before optimization. Every trial matched values, methods and
coverage byte for byte, with maximum finite difference 0 pm and identical
availability. Optimized receipts also matched the pre-edit receipts exactly.

| C1 input | Profile | Before (s) | After (s) |
| --- | --- | ---: | ---: |
| 20260724/result1 | default | 6.483 | 4.905 |
| 20260724/result1 | legacy | 0.416 | 0.418 |
| 20260724/result1 | observed_only | 0.878 | 0.429 |
| 20260323/result | default | 3.199 | 1.910 |
| 20260323/result | legacy | 0.469 | 0.468 |
| 20260323/result | observed_only | 0.782 | 0.554 |

A separate two-well default run, using the same raw cubes/configuration with
unique well names, took 6.921 s serial and 5.721 s with two workers, including
staging and result transfer. Values/methods/coverage bytes, receipts, provenance
and order matched. Linux high-water RSS was 1264 MiB in the parent and 1447 MiB
for the largest child; the parent retained the serial oracle during this check.
These are per-process high-water marks, not a simultaneous summed memory figure.

Complete Cache API single-trial comparisons, including receipts and warm OS pages:

| C1 input | Compressed write/read (s) | Plain write/read (s) | Disk compressed/plain (MiB) |
| --- | ---: | ---: | ---: |
| 20260724 | 2.568 / 0.955 | 0.574 / 0.768 | 16.6 / 191.4 |
| 20260323 | 2.852 / 1.052 | 0.677 / 0.897 | 16.7 / 223.9 |

Both storage variants round-tripped arrays and receipts exactly. These results
do not measure crash durability, cold disk reads or a full verifier/PDF rebuild.

## Committed synthetic oracle

`tests/fixtures/preprocessing-performance.npz` is a 435 KiB synthetic fixture,
not measurement data. It contains a 120-frame 16×16 float64 cube, original codes,
timestamps, allowed mask, phase starts 60/95 and pre-optimization output arrays
for default, legacy and observed_only. Expected values/methods/coverage were
captured from the pre-edit implementation, independently of the optimized paths.

The cube has small spatially varying ramps/noise, a four-frame returning pulse,
a longer pulse, a neighboring persistent step, short and phase-straddling gaps,
a wholly missing pixel phase, and supported terminal losses in two phases.
It also includes an excluded interior pixel and phase level shifts. Registration
is disabled for the new profiles and background picks are `(3,3)` and `(4,3)`.
The default oracle contains OBSERVED, TEMPORAL, SPATIAL, STITCHED, MISSING and
EXCLUDED samples. Tests require byte equality, with no numeric tolerance.

Additional tests cover nearest-edge windows longer than the phase, infinities,
NaNs, signed zeros, sparse/repaired anchors, insufficient support, sequential
boundaries, receipt equality, old compressed cache entries, all-profile serial/
parallel equality, deterministic result order, bounded staging and error cleanup.
The complete working-tree suite passed 154 tests, including 21 new performance
regression cases and the separate concurrent curation tests. Whitespace and
new Python line-length checks passed (110-character limit).

Run from the evaluator root:

```bash
PYTHONPATH=. /home/balint/anaconda3/envs/nanobio-py311/bin/python -m pytest -q nanobio_core/tests
```

## Reordered profile comparison, 2026-10-14

`tests/benchmark_preprocessing_v2.py --inputs PERF_DIR` loads the same frozen C1 NPZ inputs
and saved project background picks read-only. Three alternating calls per profile used the
documented Python with `PYTHONPATH=. OPENBLAS_NUM_THREADS=1`; loading/comparison/cache are excluded.
July now uses the current `default` low-rank curation, rather than the earlier frozen per-pixel policy.
Median `default` / `default_v2`: July result1 **5.827 / 1.952 s**, March result **2.124 / 1.903 s**.
Repeated output arrays matched bytes within each profile. July `[233,239)` had 5,109 strict
anchor-line repairs (stage-3 residual 0 pm), 120 `unexplained_event` pixels and two `no_anchor` pixels.
V2 reuses baseline estimates when no events/exclusions occur; baseline/label reductions use Bottleneck.
These timings cover selected C1 cubes (938/1,100 × 80 × 80), rather than full measurement loading.
All six frozen real-input profile comparisons also matched values/methods/coverage bytes;
July default used the frozen `per_pixel` policy for those compatibility checks.
