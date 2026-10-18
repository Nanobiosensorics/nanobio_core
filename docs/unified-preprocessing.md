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
    result, [(x, y, fraction)], min_available_weight=config.export.min_available_weight)
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
In `observed_only`, background donors must be OBSERVED. If all selected picks
are unavailable in a frame, aggregation falls back to other allowed, non-cell,
low-signal background pixels (median absolute observed trace below `threshold_pm`).
Without donors, the background trace is interpolated at actual timestamps between
supported frames in the same phase. No extrapolation or cross-phase filling is
performed. Receipts record fallback frame/pixel counts, interpolation frames and
their anchors, and remaining unavailable frames. Other profiles retain their
background policy; `legacy` retains its separate numerical path.
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

Weighted sums do not rescale missing weights. They require `min_available_weight`
(default 0.5) of total requested weight to be finite OBSERVED, TEMPORAL, STITCHED,
SPATIAL or CURATED samples; MISSING and EXCLUDED never count. This applies to all
profiles, so repaired pulse frames remain exportable. Observed and reconstructed
fractions remain separate. `return_method_fractions=True` adds a fourth return
value mapping method names to weight fractions. `min_observed_weight` remains a
deprecated argument/config alias with the new availability semantics; saved
configs serialize the new name. For `(0.25, 0.75)` at `(100, 200)` pm the sum is
175 pm (multiply by 625 µm² for physical IWS). Entirely unavailable or zero
available-weight regions always return NaN.

## `default_v2`

`PreprocessConfig.profile('default_v2')` opts into reordered recovery; existing profiles keep their arrays.
Order: sample/area mask → median-projection registration → well events → joint pixel labels → fixes
→ five-observation median reference/background → checked level offsets → finalize.
Zero/nonfinite/outside samples are unavailable; impossible jumps stay as suspect evidence; flat traces stay valid.
Phase-local events require ≥50% pixel deviations from anchor lines and a well-median robust z ≥6.
Returning well changes are paired to avoid median self-masking; reviewed intervals confirm or add events.
Curation unions overlapping intervals and never narrows detection, except explicit exclusions.
Adjacent well-median residuals above the robust pixel detection threshold extend bounds
by at most `v2.max_edge_extension=2` frames per side, within the phase and outside exclusions.
`preprocess_measurement` first detects events independently in every loaded well, even when
only some outputs are requested. Same-phase overlapping events supported by at least
`v2.consensus_fraction=0.5` of measurement wells share the union of their intervals.
Endpoint outliers beyond three scaled MADs (two-frame minimum tolerance) do not contribute
to that union or its support count. Matching-phase wells receive the shared interval;
curation may widen it, and exclusions still win. Receipts retain each well's detected bounds,
the shared bounds, supporting/outlier wells and final bounds. The event plan is hashed into
cache provenance. `preprocess_well` alone uses its own detection plus curation and edge extension.
Curation `action: exclude` wins over detection/confirmation and protects that interval from automatic fixes.
Trimmed low-rank fits (rank ≤3, variance 0.95) replace accepted intervals with the anchor line, using CURATED=6.
Explained fraction <0.8 becomes MISSING `unexplained_event`; incomplete anchors become MISSING `no_anchor`.
For well-event replacement only, partial pulse evidence can qualify as `partial_event`:
complete anchors must have detrended deviations and fitted drift across the event ≤75 pm,
and at least `max(rank + 1, minimum_clean_frames)` original pulse observations must identify
the entire learned basis with explained fraction ≥0.8. Unavailable pulse samples never
train the basis. The qualified interval uses the anchor line; insufficient evidence,
rank deficiency, anchor variation/ramps or poor fits remain missing. Receipts expose
`partial_event_rejections` and explained fractions. Reviewed default low-rank subtraction
retains its original completeness rule. See the [C4 audit](preprocessing-c4-missing-audit.md)
for the read-only 0724 investigation: 83,790 → 76,890 missing samples, with the remaining
differences and exact pixel/frame runs recorded.
Two-pass baseline/MAD estimates precede immutable gap/pulse/step/dropout/unresolved labels; fixes use those labels.
Gaps ≤5 and returning pulses ≤15 frames interpolate together; longer/unanchored disturbances stay unavailable.
Stitching needs history/neighbours; regional steps ≥10% stay observed; translated repairs become STITCHED.
Spatial donors are OBSERVED/TEMPORAL only; background also accepts CURATED, with observed-only fallback logic.
Level offsets require all five immediate edge frames on each side to be OBSERVED/TEMPORAL/CURATED; otherwise
leave that pixel unshifted and record `offset_unanchored` plus boundary counts. `config.v2` exposes event/pulse defaults.

## Curation and cache

Pass a YAML path (e.g. `curation/20260724.yml`) or a list of entries with `well`,
`start_frame`, exclusive `end_frame`, `reviewer`, `date`, `reason`, and optional
`clean_frames`/`threshold_pm` (`threshold` alias), `method`, `min_correlation`,
`max_rank`, `basis_variance` and `min_explained`. `entries:` is accepted.
The default method is `low_rank`, with `max_rank=3`, `basis_variance=0.95`,
`min_explained=0.8`, six clean frames on each side and a 75 pm threshold.
Entries may override these config settings. Curation remains opt-in.

`low_rank` uses the same original, complete phase-local anchor lines and pm pulse
residuals as `common_mode`. It fits an uncentered SVD to pixels with finite valid
pulse samples, excluding the largest 1% of residual norms (rounded up, keeping at
least one donor). The smallest rank explaining `basis_variance` of trimmed energy
is selected, capped by `max_rank` and numerical rank. A zero-energy fit has rank
zero. Rank caps can leave the requested variance unmet; receipts expose this.
Per-pixel least-squares coefficients give post-fit residual `e` and explained
fraction `f = 1 - ||e||² / ||r||²` (one for a zero residual). Complete pixels with
`f >= min_explained`, or maximum original residual at most `threshold_pm`, become
CURATED with corrected anchor line plus `e`. All unmodelled signal is retained,
even above anchor noise; earlier stitched baselines are honored. Low-explained,
larger or unavailable pulses become MISSING with `unexplained_pulse`; incomplete
original or unavailable corrected anchors become MISSING with `no_anchor`.
The reviewed validity jump-gate bypass also applies to `low_rank`; dropout codes
and nonfinite pulse samples remain unavailable. Excluded pixels stay untouched.

Low-rank receipts contain `chosen_rank`, `singular_values_pm`, `explained_variance`
(per-component fractions of trimmed energy), `basis_vectors` (rank × frames,
unitless orthonormal rows), `coefficient_maps` (rank × height × width, pm),
`explained_fraction_map`, `residual_norm_pm`, `post_fit_residual_norm_pm`,
`basis_fit_mask`, donor/trim counts, outcome counts and `max_residual_after_pm`
over repaired pixels against corrected anchor lines. Outcomes distinguish
`low_rank_component`, `near_noise`, `unexplained_pulse` and `no_anchor`.
Cache and CLI serialization preserve these arrays. Use
`curation.pulse_basis_similarities({well: basis_vectors, ...})` on aligned
intervals for pairwise principal angles in radians, cosines and similarity
`sum(cosines²) / max(rank_a, rank_b)`. This is invariant to basis signs/rotations
and penalizes unmatched dimensions; empty/nonfinite bases give `None`.
Normalized-residual clustering is deferred; SVD handles multiple segment shapes
without cluster assignment. A biological component lying inside the learned
basis cannot be distinguished from artifact by this model.

`common_mode` fits a linear anchor trend at actual timestamps per pixel from
complete valid original clean windows in the same phase. The well pulse shape
is the median residual across allowed pixels with complete anchors. Each pixel's
least-squares scale and Pearson correlation are fitted against that shape. For
correlated pixels, only the fitted common component is removed if every remaining
residual is within `noise_multiplier` (default 6) times anchor MAD noise
(`noise_scale=1.4826`, 1e-6 pm numerical floor); otherwise the anchor line replaces
the interval. Both outcomes are CURATED and honor earlier stitched baselines.
Low-correlation pixels with maximum anchor residual ≤ `threshold_pm` retain
their original signal; larger or unavailable residuals become MISSING with
`uncorrelated_pulse`. Incomplete original or unavailable corrected anchors become
MISSING with `no_anchor`. Constant/unsupported shapes cannot establish correlation.
The 1000 pm validity jump gate is bypassed inside reviewed common-mode intervals
and on their falling-edge transition, while dropout codes and nonfinite samples
remain unavailable. No excursion, anchor-noise or neighbor qualification gate
rejects otherwise complete common-mode anchors.

Per-well interval receipts contain NumPy arrays `shape_pm`, `amplitude_map`
(dimensionless scales), `correlation_map`, `noise_limit_pm` and `outcomes`, plus
`shape_amplitude_pm` (max absolute shape), counts by outcome, `repaired_pixels`
and `max_residual_after_pm` against corrected anchor lines. Cache round trips
preserve arrays without pickle; CLI JSON exports them as lists.
`curation.pulse_shape_correlations({well: shape_pm, ...})` returns pairwise
correlations for aligned intervals; unsupported pairs return `None`.

Set `method: common_mode` to retain the single-shape algorithm above, with
`min_correlation=0.8`. Set `method: per_pixel` to retain the previous confirmed-pulse algorithm and
validity gates exactly, including clean-trend, excursion and neighbor checks.
Registered acquisition evidence is retained separately from automatic estimates,
so automatic interpolation cannot erase the evidence for a reviewed interval.

`Cache(directory)` writes atomic uncompressed NPZ arrays plus JSON, without pickle;
old compressed entries remain readable. The key
hashes raw array content/shape/dtype and input metadata, config, implementation
content (including shared dependencies), and normalized curation. Cached
provenance must exactly match the request. With `cache=None` no persistent cache is written.
Callers may explicitly choose `<measurement>/result/.cache/preprocess`; no
measurement-local cache is created implicitly. Each well is independent in a
process pool; one and multiple workers yield identical arrays and provenance.
`workers=None` or `1` runs serially. A positive worker count opts into a pool
capped by the selected well count; start with two workers. Only that many wells
are staged/submitted at once, using temporary read-only NPY mappings instead of
pickling raw cubes. Staging is removed on success or failure. Results retain the
requested order; the returned result dictionary necessarily holds all outputs.
The loader still decodes the full measurement. See
[performance validation](preprocessing-performance.md) for timing and memory limits.
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
