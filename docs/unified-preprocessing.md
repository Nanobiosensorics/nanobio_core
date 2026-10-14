# Unified Cardio preprocessing

`nanobio_core.preprocessing` implements the design's nine ordered, pure stages:
validity, active area, registration, phase-local recovery, reviewed repairs,
reference, background, phase offset and finalization. Quantitative allowed
samples remain signed float64 pm with NaN for unavailable data. Excluded pixels
retain their raw nm bytes as an invariant; their method is EXCLUDED and exports
ignore them. No neighbor dilation, clipping or numeric missing-data fill occurs
in the default quantitative path.

Scoping: core README, pixel-recovery and phase-offset docs, the authoritative
`cardio-fluidfm-evaluation/docs/unified-preprocessing-design.md`, publication
report §2, existing correction/active-area code and the two recovery scripts.

## API and profiles

```python
from nanobio_core.preprocessing import PreprocessConfig, RawWell, Cache
from nanobio_core.preprocessing import preprocess_well, preprocess_measurement, weighted_region_sum

config = PreprocessConfig.profile('default')
result = preprocess_well(raw, config, curation=None)
results = preprocess_measurement(path, config, curation=None, wells=None, workers=1,
                                 cache=Cache(output_directory))
value, observed_fraction, reconstructed_fraction = weighted_region_sum(
    result, [(x, y, fraction)], min_observed_weight=config.export.min_observed_weight)
```

`RawWell(values_nm, codes, time_s, phases, allowed, well='')` carries original
instrument codes (zero means dropout), strictly increasing seconds and phase
start indices. `PreprocessResult` provides `values_pm`, uint8 `method`, observed
`coverage`, stage `receipts` and `provenance`. Existing method integers 0–5 are
unchanged; CURATED is 6. Frozen nested config sections include detector and
reviewed-repair constants. Use `to_dict()`, `from_dict(data)` and `config_hash()`
for JSON project parameters; use `dataclasses.replace` for overrides.

`default` uses five observed reference anchors, deterministic spaced low-signal
background picks, and `phase_offset.mode='level'`: the median of the last five
finite observed or repaired samples before each gap minus the median of the first
five after it. `anchor_frames` controls that count; there is no slope extrapolation.
Insufficient support is recorded and left uncorrected.
Gap correction assumes unobserved differences are offsets;
it can remove biological responses. `phase_offset.mode='none'` permits sensitivity
analysis; `single` uses the two boundary samples. `extrapolate` fits lines to the
same anchors, evaluating toward the gap midpoint at most `max_extrapolation_s`
(default 300 seconds) beyond the nearest valid sample on each side. Saved `robust`
configs are accepted as a deprecated alias and serialize/hash as `extrapolate`.
`observed_only` also uses `level`; the separate `legacy` numerical path is unchanged.
Damaged frame-zero active-area detection fails explicitly instead of silently
expanding the usable area.

Background aggregation accepts finite observed or reconstructed samples,
including CURATED picks; only MISSING/EXCLUDED and nonfinite values are omitted.
Recovery eligibility, clean anchors and spatial donors are restricted to the
current phase. A wholly missing phase is not filled using prior-phase support.

The phase-offset receipt includes `background_checks` for every boundary, using
the same picks and aggregation as background subtraction. `background_step_pm`
is the after-minus-before difference of the first/last `anchor_frames` finite background medians;
`passed` compares its magnitude with three times the within-phase background noise.
Noise is 1.4826 × MAD of first differences / √2 in the two adjacent phases of the
original background trace; a 1e-9 pm floor handles arithmetic roundoff. Missing
picks/support give null step/pass values. Checks audit actual levels, including
when offsets are disabled; they do not certify extrapolated midpoint agreement.

Read-only March/July verification found that uncapped extrapolation, applied after
background subtraction, moved March pick (43,35) by −24.65 pm and introduced a
−20.20 pm aggregate background step. Level matching preserves the stage order:
the pick's boundary step is now 0 pm and its later-phase median is +3.20 pm
(production 0 pm). The measured aggregate step is +0.20 pm, within background
noise. Background subtraction itself was not applied after the offset stage.

`legacy` executes the original registration → correct_well → jump interpolation
→ single-frame offset functions, including dilation and clipping. Its automatic
background uses an isolated configurable random seed (default 0), with global
RNG state restored. Equivalence holds against the original path with that seed
or identical explicit background coordinates. `observed_only` (also
`observed-only`) disables automatic reconstruction; reviewed curation remains
explicitly opt-in. Spatial estimates never supply recovery donors or maxima.

Weighted sums do not rescale missing weights. They require the configured
fraction of total requested weight to be OBSERVED, and also report reconstructed
weight. For `(0.25, 0.75)` at `(100, 200)` pm the sum is 175 pm (multiply by
625 µm² for physical IWS). Entirely unavailable regions always return NaN.

## Curation and cache

Pass a YAML path (e.g. `curation/20260724.yml`) or a list of entries with `well`,
`start_frame`, exclusive `end_frame`, `reviewer`, `date`, `reason`, and optional
`clean_frames`/`threshold_pm` (`threshold` alias). `entries:` is accepted. Clean
anchors and neighbor evidence must be original observations within one phase;
the existing confirmed-pulse function applies the repair and records CURATED.
Registered acquisition evidence is retained separately from automatic estimates,
so automatic interpolation cannot erase the evidence for a reviewed interval.

`Cache(directory)` writes atomic NPZ arrays plus JSON, without pickle. The key
hashes raw array content/shape/dtype and input metadata, config, implementation
content (including shared dependencies), and normalized curation. Cached
provenance must exactly match the request. With `cache=None` the API never writes.
Callers may explicitly choose `<measurement>/result/.cache/preprocess`; no
measurement-local cache is created implicitly. Each well is independent in a
process pool; one and multiple workers yield identical arrays and provenance.
The existing measurement decoder supplies wavelengths/time; DMR files additionally
supply instrument codes. Saved top-level or preprocessing `phases` take precedence
over gaps >60 s, with source recorded. Saved background picks and trims are used.
Loading honors `preprocessing.flip` (app default `[True, False]`) and
`preprocessing.signal_range`: phase ranges resolve against full-acquisition
boundaries, frame ranges slice directly, and values/codes/timestamps are sliced
together. Remaining phase indices are rebased to the selected frame range.

```bash
python -m nanobio_core.preprocessing.cli run MEASUREMENT --profile default --out OUTPUT
python -m nanobio_core.preprocessing.cli review MEASUREMENT --out REVIEW
python -m nanobio_core.preprocessing.cli diff MEASUREMENT --a legacy --b default
```

`run` requires an output directory; writing inside a measurement requires
`--in-place`. `review` exports receipts JSON; plotting is external. `diff` reports
finite differences, availability changes and method changes.

## Migration status

The legacy evaluator cube entrypoints opt in via
`preprocessing_params['pipeline_profile']`, with optional `pipeline_config`,
`pipeline_curation`, `pipeline_allowed` and `pipeline_codes`. Preprocessing saves
selected actual timestamps as JSON `pipeline_time_s`; localization requires
those timestamps and reuses the same config, raw cube and background picks.
Localization alone requires callers to supply `pipeline_time_s`. Because this
old API carries only wavelength cubes, codes default to nonzero when omitted;
use `preprocess_measurement` or explicitly provide codes for instrument dropout
semantics. Localization border filters still restrict candidate centers.
NaN and excluded raw wavelengths become zero only at this compatibility boundary; reconstruction method
filters exclude spatial and curated samples from localization maxima. Direct
quantitative exports must consume `PreprocessResult`, not the legacy finite cube.
Without a profile the existing app behavior and GUI remain unchanged.

Historical recovery sources, external July scripts, existing projects and GUI
profile controls are not migrated automatically. Reprocessing datasets,
re-exporting and retraining are separate work. Core tests include detector
benchmarks, phase/border/donor invariants, fixture legacy comparison, real-fixture
temporal/spatial holdouts, level/capped offsets, background continuity, cache
invalidation and worker equality; holdout errors are printed with `pytest -s`
and are not publication validation of
every measurement or detector.
